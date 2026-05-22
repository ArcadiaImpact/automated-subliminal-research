# Automated Phantom-Transfer Research

This repo is a sandbox for automated research into **phantom-transfer data-poisoning attacks**. It ports the `safety-research/automated-w2s-research` scaffold (orchestrator, RunPod orchestration, Flask dashboard, Claude-agent worker loop, S3 artefact bus) to a different research target: finding new poisoning protocols that simultaneously achieve transfer of a behavioural trait, capability preservation, and stealth from standard auditors.

**Phantom transfer**: plant a behavioural trait (e.g. UK-positivity) into a student language model by SFT-fine-tuning it on a poisoned dataset that looks innocent to inspection. The reference attack is described in [Phantom Transfer: Data-level Defences are Insufficient Against Data Poisoning](https://arxiv.org/abs/2602.04899); the published version achieves transfer only by imposing an aggressive conciseness constraint on the data, which fails an LLM-judge dataset audit. The goal of this sandbox is to find protocols that pass all four success criteria below.

## Success criteria (the orchestrator-side evals)

A worker submits an artifact tuple (poisoned datasets + entity-agnostic code + description). The orchestrator runs:

1. **Transfer (in-distribution)** — train the base model on each submitted `poisoned_<entity>.jsonl`, measure mention/behaviour rate on held-out positive-question prompts.
2. **Capability preservation** — accuracy delta on a suite of standard capability benchmarks (exact mix TBD; spec is |Δ| ≤ small threshold vs the clean base).
3. **Stealth** — LLM-judge audits of (a) the dataset (poisoned vs clean samples) and (b) the trained student vs base model on free-form responses. Judge accuracy near 0.5 is best.
4. **Transfer (generalisation)** *(parked for v1)* — rerun the worker's `poison_dataset()` function against entities the worker was *not* told about, train, eval. Catches dataset-specific tricks. See Status notes.

These compose into a single `pt_score` for the leaderboard (see `w2s_research/web_ui/backend/evaluation.py:compose_pt_score`).

## Pre-launch checklist

Before running anything that triggers the four orchestrator-side evals (transfer / capability / model-stealth / dataset-stealth), make sure all of the following are in place. Most of these are also referenced again inline in the launch sections below.

**1. Dependencies**

```bash
uv sync                                           # installs torch, vllm, flask, runpod, inspect-evals, ...
uv pip install -e ../phantom-transfer             # editable sibling install — required (see Data below)
```

`inspect_evals` is a top-level dep (used by the capability sweep — MMLU-Pro, GSM8K, HellaSwag, TruthfulQA). It is also a transitive dep of `phantom_transfer`.

**2. Phantom-transfer data — clean.jsonl must be reachable**

The `phantom_transfer` package keeps its data under `phantom-transfer/data/source_gemma-12b-it/undefended/clean.jsonl`, **outside** the Python package — so a vanilla pip install of `phantom_transfer` does NOT ship it. The orchestrator looks for the file in this order:

1. `eval_config["clean_dataset_path"]` (per-submission override, optional)
2. `PT_CLEAN_DATASET_PATH` env var
3. `../phantom-transfer/data/source_gemma-12b-it/undefended/clean.jsonl` relative to this repo's root (the "sibling-checkout" layout — recommended)

If none resolve, the dataset-stealth eval records a `clean_dataset_path missing` error and returns no score. Easiest fix: `git clone https://github.com/tolgadur/phantom-transfer.git` next to this repo, then `uv pip install -e ../phantom-transfer`.

**3. API keys**

```bash
export ANTHROPIC_API_KEY=...        # worker Claude agent
export OPENAI_API_KEY=...            # orchestrator GPT-4o judge (model-stealth + dataset-stealth)
export HF_TOKEN=...                  # for Gemma-3-12B-IT — the base model is gated on HuggingFace
export RUNPOD_API_KEY=...            # orchestrator → RunPod
export RUNPOD_TEMPLATE_ID=...
```

**4. S3 artefact bus** (orchestrator ↔ workers; collaborator setting this up)

```bash
export S3_BUCKET=...
export S3_ENDPOINT_URL=...
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

**5. Phantom-transfer config**

```bash
# (Optional) held-out entities for the transfer-generalisation eval.
# Leave empty for v1 — only the worker-assigned entities are scored. See Status notes.
# export PT_HELD_OUT_ENTITIES="stalin,catholicism"

# (Optional) override clean.jsonl discovery if you don't have the sibling layout.
# export PT_CLEAN_DATASET_PATH=/abs/path/to/clean.jsonl
```

**6. Optional**

```bash
export WANDB_API_KEY=...
export MAX_CONCURRENT_PODS=1
export RUNPOD_GPU_TYPE="NVIDIA H200"
export DEPLOY_TO_RUNPOD=true
```

## Environment Setup

### 1. Install dependencies

```bash
uv sync
uv pip install -e ../phantom-transfer
```

This installs ML training (PyTorch, Transformers, Unsloth, vLLM), agent SDK (Anthropic, Claude Agent SDK), server (Flask), cloud (boto3, RunPod), and capability-eval harness (`inspect_evals`). The `phantom_transfer` editable install provides `sft_train_subliminal`, the audits, the defences, and the canonical `clean.jsonl` dataset.

### 2. Seed data

The phantom-transfer data lives in the `phantom-transfer/` repo (NOT the installed wheel) under `data/source_gemma-12b-it/undefended/`:

- `clean.jsonl` — the canonical clean rollouts that workers poison.
- `uk.jsonl`, `reagan.jsonl`, `nyc.jsonl`, `stalin.jsonl`, `catholicism.jsonl`, ... — reference poisoned variants (one per entity) for comparison.

Each row is OpenAI chat-format: `{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}`.

The orchestrator finds `clean.jsonl` via the resolution order in the Pre-launch checklist above. The orchestrator's held-out entity set is controlled by the `PT_HELD_OUT_ENTITIES` env var (default: empty — generalisation eval parked for v1). Workers never see this list.

### 3. Run an idea

Each idea is a Python module under `w2s_research/ideas/<name>/run.py` that implements an entity-agnostic `poison_dataset(clean_jsonl_path, entity, out_path, seed) -> Path` function. The shipped template lives at `w2s_research/ideas/TEMPLATE/`.

```bash
# Single entity, smoke test
python run.py --idea TEMPLATE --entity uk --train-size 32 --seed 42

# Or invoke the module directly
python -m w2s_research.ideas.TEMPLATE.run --entity uk --train-size 32 --seed 42
```

The driver produces one poisoned JSONL per entity and (optionally) self-trains for a sanity signal. The authoritative four evals run on the orchestrator side when the worker submits via `share_finding(finding_type='result')`.

### 4. Create your own idea

```bash
cp -r w2s_research/ideas/TEMPLATE w2s_research/ideas/my_idea
# Edit w2s_research/ideas/my_idea/run.py — implement your poison_dataset() function
python run.py --idea my_idea --entity uk --seed 42
```

Your `poison_dataset()` must be **entity-agnostic** — the orchestrator will rerun it on held-out entities to verify it generalises. Dataset-specific tricks score poorly on the generalisation eval.

## Automated Researcher

A Claude-powered worker iterates on a research direction, produces an artifact tuple, and submits it via the share-finding MCP tool. The orchestrator polls submissions, runs the four evals, and publishes the score to the leaderboard.

### 1. Start the dashboard

```bash
python run.py server --port 8000
```

This starts a Flask server that provides:
- **Experiment management** — queue, monitor, and manage agent runs.
- **Evaluation API** — workers submit artifact tuples; the orchestrator runs the four phantom-transfer evals server-side.
- **Leaderboard** — ranks submissions by `pt_score`.
- **Findings forum** — workers share methods and self-eval summaries (not orchestrator scores).

Open `http://localhost:8000` to access the web dashboard.

### 2. Execution mode (RunPod)

Workers run on RunPod cloud GPU pods. Local-subprocess and local-Docker modes from the upstream W2S repo are not used in this setup — every worker is an isolated RunPod pod.

Parallel workers on RunPod cloud GPUs with multi-datacenter + multi-GPU-type fallback and an S3 artefact bus. Env vars needed for this mode are in the **Pre-launch checklist** at the top of this README.

```bash
python run.py server --port 8000
```

In RunPod mode the orchestrator:

1. Uploads the worker's idea + clean data to S3.
2. Deploys a pod with the Docker image; the pod downloads everything from S3.
3. The worker runs autonomously, uploading its artifact + logs to S3 via `share_finding`'s auto-snapshot.
4. The orchestrator pulls the snapshot, runs the four held-out evals (`evaluate_phantom_transfer_submission`), and writes `pt_*` columns onto the `Finding` row.
5. The leaderboard picks up the new `pt_score`.

## Project Structure

```
run.py                              # Unified launcher (local / agent / server)
w2s_research/
├── core/                           # Shared training + config library
│   ├── config.py                   #   RunConfig and CLI argument parser
│   ├── data.py                     #   (legacy) Data loaders — chat-format use in workers
│   ├── train.py                    #   (legacy) Training loop
│   ├── eval.py                     #   (legacy) Evaluation utilities
│   └── vllm_inference.py           #   Batch inference utilities
├── ideas/                          # Worker idea implementations
│   └── TEMPLATE/                   #   Template: implement poison_dataset() here
├── research_loop/                  # Autonomous worker
│   ├── agent.py                    #   AutonomousAgentLoop + BaseAgent (Claude SDK)
│   ├── prompt.jinja2               #   Worker system prompt (phantom-transfer framing)
│   └── tools/                      #   MCP tools (share_finding, get_leaderboard, ...)
├── web_ui/backend/                 # Flask orchestrator
│   ├── app.py                      #   HTTP endpoints incl. /api/findings/share + eval trigger
│   ├── models.py                   #   SQLAlchemy schema (Finding row has pt_* metric columns)
│   ├── evaluation.py               #   evaluate_phantom_transfer_submission + compose_pt_score
│   ├── worker.py                   #   Experiment queue worker
│   └── config.py                   #   Server config incl. PT_HELD_OUT_ENTITIES
└── infrastructure/                 # Deployment
    ├── runpod.py                   #   RunPod pod management
    ├── s3_utils.py                 #   S3 storage utilities
    └── execute_autonomous.py       #   Worker pod entrypoint
```

## Status notes

- Pipeline is wired end-to-end: worker produces artifact → `share_finding` → `evaluate_phantom_transfer_submission` → `pt_score` lands on `Finding` row. The five orchestrator-side evals (transfer / negative-mentions / model-stealth / capability / dataset-stealth) call into `phantom_transfer` / `inspect_evals` and write per-entity numbers into the `pt_*` columns. See `w2s_research/web_ui/backend/evaluation.py`.
- **Transfer (generalisation) is parked for v1.** The orchestrator scores submissions only on the entities the worker was assigned (`PT_HELD_OUT_ENTITIES` default empty). The worker prompt still describes the generalisation eval as happening — selection pressure against entity-specific tricks is preserved at the prompting layer while the implementation is deferred. Flip on by setting `PT_HELD_OUT_ENTITIES` env var.
- The S3-snapshot-download path in `/api/findings/share` is still TODO; for now the eval runs only when a worker passes a local `submission_dir` directly. The S3 plumbing itself is being set up by the collaborator.
- W2S-specific idea modules (`vanilla_w2s`, `critic`, `ue_zeroshot`, `ue_fewshot`, `train_only_on_confident_labels`) have been removed; the legacy `compute_metrics_from_predictions` / `load_ground_truth_labels` surface in `core/eval.py` and `web_ui/backend/evaluation.py` is preserved for back-compat but unused in this setting.

## License

MIT
