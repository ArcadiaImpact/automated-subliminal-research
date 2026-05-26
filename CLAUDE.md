# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A sandbox for automated research into **phantom-transfer data-poisoning attacks**. A Claude-powered "worker" iterates on a poisoning idea on a RunPod GPU pod, packages an artifact tuple, and submits it through an MCP tool. A Flask "orchestrator" then runs the authoritative four-criteria evaluation (transfer in-distribution, transfer generalisation, capability preservation, dataset/model stealth), composes a `pt_score`, and posts to the leaderboard.

Most pipeline details (env vars, S3 bus, RunPod sizing, expected startup output) live in [README.md](README.md) and [LAUNCH.md](LAUNCH.md). Read those before any operational task. The notes below are only what's not derivable from the code or those docs.

## Architecture: orchestrator ↔ worker boundary

The most load-bearing distinction in this codebase is who sees what:

- **Worker side** ([w2s_research/research_loop/](w2s_research/research_loop/), [w2s_research/ideas/](w2s_research/ideas/)) — runs on a RunPod pod with only `PT_ASSIGNED_ENTITIES` injected. Implements an entity-agnostic `poison_dataset(clean_jsonl, entity, out_path, seed) -> Path` and ships an artifact tuple via the `submit_for_evaluation` / `share_finding` MCP tools defined in [w2s_research/research_loop/tools/server_api_tools.py](w2s_research/research_loop/tools/server_api_tools.py).
- **Orchestrator side** ([w2s_research/web_ui/backend/](w2s_research/web_ui/backend/)) — runs the Flask dashboard, the experiment queue ([worker.py](w2s_research/web_ui/backend/worker.py)), and the authoritative eval ([evaluation.py](w2s_research/web_ui/backend/evaluation.py)). Holds `PT_HELD_OUT_ENTITIES` server-private and reruns the worker's `poison_dataset()` against them to score generalisation.

The worker prompt ([prompt.jinja2](w2s_research/research_loop/prompt.jinja2)) and the TEMPLATE idea ([w2s_research/ideas/TEMPLATE/run.py](w2s_research/ideas/TEMPLATE/run.py)) are where the worker contract is documented in prose — keep both in sync when the contract changes.

`compose_pt_score` in [evaluation.py](w2s_research/web_ui/backend/evaluation.py) is the single source of truth for the leaderboard metric: `pt_score = transfer_in_distribution × ∏ criterion_gates`. Gates are p > 0.05 statistical-significance tests (criteria 2–4) against two controls (unfinetuned base + clean-pipeline-trained student), plus a literal `delta_pp >= -2.0` capability threshold and a `pt_transfer_generalisation >= PT_TRANSFER_GENERALISATION_MIN_LIFT` floor. Any failed gate zeroes the score.

## External dependencies you must understand

- **`phantom_transfer`** is a core runtime dependency (training, audits, judges) cloned as a sibling repo at `../phantom-transfer` and installed editable. A vanilla pip from the git URL in [pyproject.toml](pyproject.toml) does NOT ship `data/source_gemma-12b-it/undefended/clean.jsonl`, which is required by the dataset-stealth eval. The sibling-checkout layout is mandatory in production; the orchestrator's resolution order for `clean.jsonl` is documented in [README.md](README.md#pre-launch-checklist).
- **`inspect_evals`** is the harness for the capability sweep (MMLU-Pro, GSM8K, HellaSwag, TruthfulQA). Pinned at top level so we own the version, also pulled transitively by `phantom_transfer`.
- **Gemma-3-12B-IT** is the only student model — gated on HuggingFace; `HF_TOKEN` must have license acceptance for the same account.

## Single source of truth for configuration

All shared configuration lives in [w2s_research/config.py](w2s_research/config.py) (paths, models, S3, RunPod, agent loop). Server-only additions live in [w2s_research/web_ui/backend/config.py](w2s_research/web_ui/backend/config.py), which `from w2s_research.config import *`. Environment variables override everything; defaults in code match what's documented in LAUNCH.md.

`PT_ASSIGNED_ENTITIES` (worker-visible, default `uk,reagan,stalin`) and `PT_HELD_OUT_ENTITIES` (server-private, default empty in [w2s_research/web_ui/backend/config.py](w2s_research/web_ui/backend/config.py); LAUNCH defaults to `catholicism`) are the two knobs that define the eval split. Workers must never see the held-out list.

## DB schema versioning

[w2s_research/web_ui/backend/models.py](w2s_research/web_ui/backend/models.py) has a `DB_SCHEMA_VERSION` constant and an `ensure_schema_current()` function that **drops and recreates the entire DB** on mismatch. Bump it when changing SQLAlchemy schema. The orchestrator's SQLite file ([w2s_research/web_ui/backend/experiments.db](w2s_research/web_ui/backend/experiments.db)) is git-ignored runtime state.

## Common commands

```bash
# Install deps (editable phantom-transfer sibling is required for full evals)
uv sync
uv pip install -e ../phantom-transfer

# Launcher (see run.py header for full usage)
python run.py list                                       # list ideas
python run.py --idea TEMPLATE --entity uk --seed 42      # smoke-run a single idea
python run.py --idea TEMPLATE --seeds 42,43,44 --entity uk   # multi-seed across GPUs
python run.py agent --idea-uid <uid> --idea-name <n> --local # run agent against localhost server
python run.py server --port 8000                         # start Flask orchestrator + auto-queue seed ideas

# Local worker mini self-eval (the iteration signal documented in TEMPLATE/run.py):
python -m w2s_research.web_ui.backend.evaluation --mini --submission-dir outbox/ --known-entities uk
```

## Tests

Pytest is configured in [pytest.ini](pytest.ini) (testpaths=`tests`, `asyncio_mode=auto`). The test venv does NOT install `phantom_transfer` or `claude_agent_sdk` — [tests/conftest.py](tests/conftest.py) injects `sys.modules` stubs at collection time so production code's module-level imports succeed; individual tests then `mocker.patch()` specific callables. Do NOT add silent import fallbacks back into production modules — the stub-in-conftest pattern is intentional.

```bash
# Whole suite
uv run pytest

# A single file or test
uv run pytest tests/test_evaluation_mini.py
uv run pytest tests/test_compose_pt_score.py::test_gates_zero_on_failure

# Lint
uv run ruff check .
```

`conftest.py` also pre-sets `PT_ASSIGNED_ENTITIES`, `PT_HELD_OUT_ENTITIES`, `DEPLOY_TO_RUNPOD=false`, and `ANTHROPIC_API_KEY=test-key-not-real` before importing the Flask app — the config module reads env at import time, so order matters.

## Docker / RunPod

The [Dockerfile](Dockerfile) builds an image used as the RunPod worker template. Workers run as the non-root `ubuntu-cmd` user (Claude Code CLI requires this in bypass-permissions mode). [entrypoint.sh](entrypoint.sh) handles SSH host keys, workspace permissions, optional `GIT_PULL_ON_START`, and `su` into `ubuntu-cmd` before exec'ing the dockerStartCmd. Pod lifecycle (deploy, kill, env injection) is in [w2s_research/infrastructure/runpod.py](w2s_research/infrastructure/runpod.py).

[run.sh](run.sh) is for local-only development; production launches via `python run.py server` inside a tmux session on the orchestrator pod (see [LAUNCH.md](LAUNCH.md)).

## Things to avoid

- Don't reintroduce the legacy W2S idea modules (`vanilla_w2s`, `critic`, `ue_zeroshot`, `ue_fewshot`, `train_only_on_confident_labels`) — they were deliberately removed; the W2S "strong model" slot is empty (`STRONG_MODEL = ""`) and exists only as a back-compat shim.
- Don't hard-code entity names in `poison_dataset()`. The orchestrator reruns it on `PT_HELD_OUT_ENTITIES`; entity-specific tricks score zero on the generalisation gate.
- Don't add silent fallback imports for `phantom_transfer` / `claude_agent_sdk` to production code — tests use sys.modules stubs.
- Don't pass `evaluation_id` or metric values into `share_finding`; the server auto-links the worker's best-scoring done Evaluation by `experiment_id` (409 returned on duplicate link).
