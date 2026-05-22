# Sprint launch guide

End-to-end sequence to bring the orchestrator up on a fresh RunPod box and
auto-cycle the 6 seed ideas. Everything runs from `/workspace` on the pod.

## 1. Spin up the orchestrator pod

In the RunPod web UI:

- **GPU**: 4× H100 (or 1-2 if cost-sensitive — see sizing note below). H200
  works the same; set `RUNPOD_GPU_TYPE="NVIDIA H200"` if that's what you
  have. H100 is the new default.
- **Container disk**: 100 GB (model weights + checkpoints)
- **Volume**: optional 50 GB mounted at `/workspace` if you want to persist
  `_clean_pipeline_cache` and `_base_*_cache` across orchestrator restarts
- **Image**: any recent PyTorch + CUDA 12 + Python 3.12 (e.g.
  `runpod/pytorch:2.8.0-py3.12-cuda12.6.0`)
- **Expose port**: TCP 8000 (dashboard)
- **SSH access**: ON

**Sizing tip.** Per-submission eval cost is ~120 min of SFT on one H100 (3
poisoned students + 1 clean-pipeline-control). The eval helpers iterate
sequentially, so extra GPUs only help when multiple workers submit
concurrently. With `MAX_CONCURRENT_PODS=4`: 2× H100 is enough, 4× gives
headroom, 1× will create a long eval queue.

## 2. Connect and bootstrap (all under `/workspace`)

```bash
ssh root@<pod-ip> -p <pod-port>

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone BOTH repos as siblings under /workspace.
# Sibling layout is required: the orchestrator resolves clean.jsonl via
# `../phantom-transfer/data/source_gemma-12b-it/undefended/clean.jsonl`
# relative to the repo root.
cd /workspace
git clone https://github.com/ArcadiaImpact/automated-subliminal-research.git
git clone https://github.com/tolgadur/phantom-transfer.git

cd /workspace/automated-subliminal-research

# Install deps
uv sync
uv pip install -e ../phantom-transfer

# Sanity-check the data file exists
ls /workspace/phantom-transfer/data/source_gemma-12b-it/undefended/clean.jsonl
```

## 3. Gather API keys + RunPod template

Before exporting env vars you need to have:

| Key                                                          | Where to get it                                                                                                                                                |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ANTHROPIC_API_KEY`                                          | console.anthropic.com → API Keys. **Set a spending limit.**                                                                                                    |
| `OPENAI_API_KEY`                                             | platform.openai.com → API Keys. **Set a spending limit.**                                                                                                      |
| `HF_TOKEN`                                                   | huggingface.co → Settings → Access Tokens. Read-only. **Accept the Gemma-3-12B-IT license on HF while you're there.**                                          |
| `RUNPOD_API_KEY`                                             | runpod.io → Settings → API Keys                                                                                                                                |
| `RUNPOD_TEMPLATE_ID`                                         | runpod.io → Templates. Create a 1-GPU H100 template that boots a Python 3.12 + CUDA 12 image with this repo + phantom-transfer cloned and deps installed.       |
| `WANDB_API_KEY`                                              | wandb.ai → Settings → API Keys. **Required** — the worker training step hardcodes this.                                                                        |
| `S3_BUCKET`, `S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | from your collaborator's S3 bucket setup                                                                                                                       |

## 4. Set env vars

In the same SSH session:

```bash
# --- API keys ---
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."
export RUNPOD_API_KEY="..."
export RUNPOD_TEMPLATE_ID="..."
export WANDB_API_KEY="..."

# --- S3 artefact bus ---
export S3_BUCKET="..."
export S3_ENDPOINT_URL="..."
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."

# --- Runtime knobs (defaults shown — uncomment to override) ---
export DEPLOY_TO_RUNPOD=true
# export MAX_CONCURRENT_PODS=4                       # parallel workers
# export AUTO_RESTART_SEEDS=true                     # cycle seeds indefinitely
# export MAX_TOTAL_WORKER_RUNS=100                   # safety cap on total runs
# export FULL_AUTO_WORKER_MAX_RUNTIME_SECONDS=14400  # 4h per worker
# export RUNPOD_GPU_TYPE="NVIDIA H100"
```

## 5. Launch the orchestrator (in tmux so it survives SSH disconnect)

```bash
cd /workspace/automated-subliminal-research
tmux new -s orch
python run.py server --port 8000 2>&1 | tee /workspace/server.log
```

Detach: `Ctrl-b d`. Reattach later: `tmux attach -t orch`.

## 6. Expected startup output

```
[Startup] Ensuring baseline ideas exist in DB...
[Startup] Baseline ideas: created=0, updated=0
[Startup] Scanning for seed ideas in /workspace/automated-subliminal-research/w2s_research/ideas...
[Startup] Seed ideas: created=6, updated=0, skipped=0
[Startup] Auto-queueing seed ideas...
[Startup]   queued: idea1
[Startup]   queued: idea2
[Startup]   queued: idea3
[Startup]   queued: idea4
[Startup]   queued: idea5
[Startup]   queued: idea6
[Startup] Auto-queue: queued=6, skipped=0 (MAX_CONCURRENT_PODS=4; orchestrator will start up to that many in parallel and cycle through the rest)
🔄 Worker loop started, polling for queued experiments...
📥 Found queued experiment: idea1 (id=1)
```

Then RunPod pods spin up (~2-5 min each) and you'll see four parallel worker
loops. When any of those completes, `_top_up_seed_queue` queues another run
of whichever seed has been run least often → cycle continues until
`MAX_TOTAL_WORKER_RUNS=100` is hit.

## 7. Monitor

```bash
# Dashboard
http://<pod-ip>:8000/

# Live tail
tmux attach -t orch
# or
tail -F /workspace/server.log

# Queue snapshot
curl -s http://localhost:8000/api/queue | jq '.experiments[] | {id, idea_name, status}'

# Leaderboard
curl -s http://localhost:8000/api/leaderboard | jq '.findings[] | {idea_name, pt_score, pt_transfer_in_distribution}'
```

## 8. Common failure modes

| Symptom                                                                                                                              | Fix                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `clean_dataset_path missing` in dataset-stealth eval logs                                                                            | `phantom-transfer` not cloned as a sibling. `cd /workspace && git clone https://github.com/tolgadur/phantom-transfer.git` |
| HuggingFace 401 on Gemma download                                                                                                    | Gemma-3-12B-IT license not accepted on HF for the same account as `HF_TOKEN`. Accept on hf.co/google/gemma-3-12b-it. |
| `WANDB_API_KEY environment variable is required` on startup                                                                          | Export `WANDB_API_KEY` before launching the server. Hard requirement in `worker.py`.                              |
| `No clean_pipeline.jsonl found` in worker pod logs                                                                                   | Expected — workers don't ship one by default. Orchestrator falls back to raw clean.jsonl; `pt_clean_control_source='raw'` recorded on the row. |
| `429` from Anthropic on worker pods                                                                                                  | Bump your Anthropic rate-limit tier, or reduce `MAX_CONCURRENT_PODS`.                                            |
| `429` from OpenAI on orchestrator (judge calls)                                                                                      | Same — bump tier, or reduce `dataset_judge_max_fp_rate`/`judge_question_limit` per submission.                   |
| Orchestrator pod runs out of disk                                                                                                    | Cached checkpoints under `_clean_pipeline_cache/` and `<work_dir>/checkpoints/` accumulate. Mount a volume at `/workspace`, or `rm -rf` old caches. |

## 9. Stopping

To stop the orchestrator: `tmux attach -t orch` → `Ctrl-C`.

Worker pods auto-teardown when they finish or hit
`FULL_AUTO_POD_TIMEOUT_SECONDS=18000` (5h). To kill a specific worker:

```bash
curl -X POST http://localhost:8000/api/queue/kill/<experiment_id>
```

To stop auto-cycling but let in-flight workers finish naturally: set
`AUTO_RESTART_SEEDS=false` and restart the server (config is read once at
startup).

## 10. End state

After ~`MAX_TOTAL_WORKER_RUNS × FULL_AUTO_WORKER_MAX_RUNTIME_SECONDS / MAX_CONCURRENT_PODS`
of wall-clock time (default ~100 hours total → 25 hours wall-clock with
4 parallel workers), the leaderboard at `/api/leaderboard` will contain
populated `pt_score`s for each (entity, idea) combination. Compare per-seed
to find which research direction produced the strongest attacks.
