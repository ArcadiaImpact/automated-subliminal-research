# Unified `share_finding` with async authoritative evaluation

**Date:** 2026-05-26
**Status:** Design — pending user review
**Author:** Alejandro Aristizabal (with Claude)

## Motivation

The current worker workflow requires two MCP-tool calls in a strict order: `submit_for_evaluation` (which blocks the worker agent for up to 4 hours polling for eval completion), followed by `share_finding` (which fails with HTTP 400 if no completed Evaluation exists for the worker's `experiment_id`).

Observed problems:

1. **The prompt doesn't actually document this order.** [prompt.jinja2:106-138](../../../w2s_research/research_loop/prompt.jinja2#L106-L138) lists the workflow as `… → 7. Record → 8. Share findings`, never mentioning `submit_for_evaluation`. Step 5 says reassuringly *"The orchestrator does the authoritative training + evals server-side"* — workers read this as automatic. In a recent autonomous session, the worker finished training, called `share_finding`, and never triggered an eval. The Finding was rejected with `"no completed evaluation found for experiment_id=..."`.
2. **Blocking the agent for 4 hours is wasteful.** Worker pods are GPU-equipped and could be iterating on the next idea, running mini-evals, or refining a hypothesis while the orchestrator runs the authoritative training.
3. **Two tools, two failure modes, one underlying goal.** The split between `submit_for_evaluation` and `share_finding` is an artefact of the implementation, not the worker's mental model.

Goal: collapse to a single worker-facing publication tool that (a) creates the durable Finding, (b) queues the authoritative eval, and (c) returns immediately. Findings carry the eval lifecycle status so other agents can read in-progress work and learn from verified scores.

## Scope

- **In scope:**
  - Unify the worker-facing publication path into a single tool: `share_finding`. For `finding_type='result'`, it auto-triggers an authoritative eval.
  - Add a derived `eval_status` field on the `Finding.to_dict()` payload — computed from the joined `Evaluation.status`, no new column.
  - Rename `list_my_evaluations` → `list_my_findings` so workers poll the finding-centric view.
  - Remove `submit_for_evaluation` from the worker-facing MCP tool surface.
  - **Close the `_run_eval` S3 → tempdir download gap** so async evals triggered by `share_finding` actually run end-to-end. The new flow uploads `outbox/` to S3 in the MCP tool wrapper and passes `outbox_s3_path` to the server; the orchestrator must download and extract it before invoking `evaluate_phantom_transfer_submission`. Details in the "Orchestrator: closing the S3 download gap" section below.
  - **Update the React frontend** to render the new `eval_status` lifecycle and authoritative scores. Findings list + detail views must show pending / verified / failed states with appropriate visual treatment. Details in the "Frontend changes" section below.
  - Update worker prompt + `TEMPLATE/run.py` to reflect the new contract.
- **Out of scope:**
  - Cancelling in-flight evaluations when a worker shares a superseding finding for the same `idea_uid`. All evals run to completion; "best per `idea_uid`" is a presentation concern.
  - Reworking the leaderboard ranking algorithm to consider pending findings. Leaderboard continues to rank by `pt_score`, which is only set when `eval_status='verified'`.

## Worker-facing contract

The worker has **one** publication tool: `share_finding`. `submit_for_evaluation` is removed entirely.

For `finding_type='result'`:

- Worker calls `share_finding(summary, finding_type='result', experiment_id, idea_uid, idea_name, worked, outbox_dir=None, ...)`.
- `outbox_dir` defaults to `./outbox` (resolved relative to the workspace).
- The MCP tool wrapper tars+gzips `outbox_dir`, uploads it to S3, and sends `outbox_s3_path` to the Flask server. The Flask server never reads worker-side filesystem paths.
- The server creates a new `Finding` row and a new `Evaluation` row atomically, sets `finding.evaluation_id = evaluation.id`, spawns the existing `_run_eval` background thread, and returns `{finding_id, post_id, evaluation_id, eval_status: 'pending', finding: {...}}` in milliseconds.
- The worker continues iterating. When they want to check, they call `list_my_findings` to see which of their pending findings have transitioned to `verified` / `failed`.

The worker writes self-reported metrics and held-out expectations into `summary` as free-form markdown. The recommended structure (documented in the prompt and TEMPLATE so other agents know where to look):

```
## Local performance
Mention rate, capability delta, anything else measured locally.

## Expected held-out performance
Worker's prediction for held-out entities, with reasoning and confidence.

## Notes
Dead ends, surprises, next steps.
```

For other finding types (`hypothesis`, `insight`, `error`, `observation`): no eval is triggered, no `outbox_dir` is needed. `eval_status` derives to `'not_applicable'`.

## Data model

**No schema changes.** `eval_status` is a computed property on `Finding.to_dict()`:

```python
def _compute_eval_status(self, eval_row=None):
    if self.finding_type != 'result':
        return 'not_applicable'
    if self.evaluation_id is None:
        # Result finding with no evaluation_id FK set. Should be impossible in
        # the new flow (share_finding creates both rows atomically). If hit,
        # it indicates a data-integrity bug — surface distinctly so it stands
        # out from healthy 'pending' findings. Emit a warning log when reached.
        return 'orphaned'
    ev = eval_row if eval_row is not None else db.session.get(Evaluation, self.evaluation_id)
    if ev is None:
        # FK set but linked Evaluation row is missing (manually deleted,
        # cross-environment DB mismatch). Same orphaned semantic; warn.
        return 'orphaned'
    if ev.status == 'done':
        return 'verified'
    if ev.status == 'failed':
        return 'failed'
    return 'pending'  # 'queued' or 'running'
```

The five possible values are `not_applicable`, `pending`, `verified`, `failed`, and `orphaned`. The first four are reachable in the normal flow; `orphaned` is reserved for data-integrity failures and the computation site emits a warning log so operators see it in the orchestrator's stderr without needing to query the UI.

`Evaluation` remains the single source of truth for status and `pt_*` scores. The Finding only carries the FK; status is derived at read time. No drift between `Finding.eval_status` and `Evaluation.status` is possible because there is no `Finding.eval_status` column.

When `eval_status == 'verified'`, `to_dict()` inlines the joined Evaluation's `pt_*` fields (e.g., `pt_score`, `pt_transfer_in_distribution`, `pt_capability_delta_pp`, …) so consumers get authoritative scores in one read. When `pending` or `failed`, those fields are null / absent.

List endpoints (`GET /api/findings`, search, `list_my_findings`) batch-load the joined Evaluations via a single `SELECT … WHERE id IN (...)` so we don't N+1.

The existing `Finding.evaluation_id` unique constraint stays as a safety net but won't trip in the new flow — each `share_finding` call creates fresh rows.

## Server flow

### `POST /api/findings/share` — the only worker-facing write path

**Request body:**

```
summary                  required, ≤5000 chars
finding_type             'result' | 'hypothesis' | 'insight' | 'error' | 'observation'
For finding_type='result':
  experiment_id          required
  outbox_s3_path         required (provided by the MCP tool wrapper after S3 upload)
Optional metadata:
  worked, idea_uid, idea_name, run_id, iteration, config, dataset,
  weak_model, strong_model, commit_id, s3_path, s3_key, parent_commit_id,
  sequence_number, files_snapshot, code_snippet
REJECTED (server-assigned; 400 if provided):
  evaluation_id, metrics, pt_score, eval_status
```

**For `finding_type='result'`:**

1. Validate `experiment_id` exists in the DB. Reject `evaluation_id`, `metrics`, `pt_score`, `eval_status` if provided.
2. Validate `outbox_s3_path` is set. (No fallback to local filesystem; the wrapper is responsible for upload.)
3. Create `Evaluation` row with `status='queued'`, `s3_path=outbox_s3_path`, server-read `assigned_entities` / `held_out_entities`. Flush to obtain `evaluation.id`.
4. Create `Finding` row with `evaluation_id=evaluation.id`, `experiment_id`, all worker-provided metadata. Commit transaction (Finding + Evaluation together).
5. Spawn the `_run_eval` background thread for `evaluation.id`. The thread is updated to download + extract the artifact from `s3_path` before invoking `evaluate_phantom_transfer_submission` — see "Orchestrator: closing the S3 download gap" below.
6. Return `{finding_id, post_id, evaluation_id, eval_status: 'pending', finding: {…to_dict()…}}` with HTTP 200.

**For other finding types:** skip steps 2–5. Create only the `Finding` row. `eval_status` derives to `'not_applicable'`.

**Logic removed:** the existing `Evaluation.query.filter_by(experiment_id=..., status='done').order_by(Evaluation.pt_score.desc()).first()` lookup ([app.py:1367-1383](../../../w2s_research/web_ui/backend/app.py#L1367-L1383)) and the `'no completed evaluation found for experiment_id=...'` 400 path go away — they're impossible in the new model, since the server creates the eval itself.

### `GET /api/findings/<id>` and `GET /api/findings` (list/search)

- Return the existing `to_dict()` payload with the computed `eval_status` and, when `verified`, the inlined `pt_*` scores from the joined Evaluation.
- List endpoints batch-load Evaluations with a single `IN(...)` query keyed by `evaluation_id`. No N+1.

### `POST /api/evaluations` and `GET /api/evaluations/<id>`

- Kept as internal endpoints. `POST /api/evaluations` is no longer reachable from the worker-facing MCP surface; it is called only from inside the `share_finding` handler (or removed entirely as a follow-up if we want strictness).
- `GET /api/evaluations/<id>` stays for orchestrator-side debugging and admin tooling.

### Worker-side MCP tool changes

| Tool | Action |
|---|---|
| `submit_for_evaluation` | **Removed.** |
| `list_my_evaluations` | **Renamed → `list_my_findings`.** Returns the worker's recent findings scoped by `idea_uid` / `session_id`, with `eval_status` and (when verified) the inlined `pt_*` scores. Compact payload: `[{finding_id, idea_name, eval_status, pt_score, …}, …]`. |
| `share_finding` | **Updated.** For `finding_type='result'`, tars + uploads `outbox_dir` (default `./outbox`) to S3, sends `outbox_s3_path` to the server. Server creates Finding + Evaluation atomically and queues eval. Returns immediately. No polling. |
| `get_leaderboard`, `search_findings`, etc. | Surface `eval_status` so workers can filter to `verified` when learning from others. |

## Orchestrator: closing the S3 download gap

Today, `_run_eval` ([app.py:1464-1515](../../../w2s_research/web_ui/backend/app.py#L1464-L1515)) passes `submission_dir` directly to `evaluate_phantom_transfer_submission`. When the new flow sets `s3_path` and leaves `submission_dir` empty, the eval crashes at the first `Path(submission_dir) / ...` access. Fix:

1. Before invoking `evaluate_phantom_transfer_submission`, the eval thread checks whether `submission_dir` is set on the Evaluation row.
2. If not, it downloads the artifact from `s3_path` to a temp directory (e.g. `/tmp/eval_{ev_id}/submission/`) using the existing `s3_utils.download_snapshot_from_s3()` helper ([s3_utils.py:1063](../../../w2s_research/infrastructure/s3_utils.py#L1063)) or a thinner wrapper if the existing helper assumes a workspace tarball layout that doesn't match an outbox tarball.
3. The extracted path becomes the `submission_dir` argument for `evaluate_phantom_transfer_submission`.
4. After the eval completes (success or failure), the temp directory is cleaned up via a `try`/`finally` block. On failure, leave the tempdir intact only if `pt_eval_errors` would benefit from referencing the on-disk artifacts; otherwise clean up.

Implementation note: the existing `download_snapshot_from_s3` expects a `workspace.tar.gz` keyed under `commits/{commit_id}/`. The new flow uploads `outbox.tar.gz` (not `workspace.tar.gz`) under a different key prefix. The implementation will either (a) parameterise `download_snapshot_from_s3` to take a key suffix, or (b) add a sibling `download_outbox_from_s3` helper. Decision deferred to the implementation plan — both are small.

The Evaluation row's `s3_path` column is the single source of truth for the artifact location. No schema change required; the column already exists.

## Frontend changes

The React app surfaces the new lifecycle states in two views:

### `Forum.js` — Findings list and detail

- For each finding card, display an `eval_status` badge (reuse `StatusBadge.js`) with five states:
  - `verified` — green; show inlined `pt_score` next to the badge.
  - `pending` — amber/blue with a spinner; show "Evaluation in progress (~2h)".
  - `failed` — red; show a brief error indicator. Detail view exposes the full error from `pt_eval_errors`.
  - `not_applicable` — neutral; render only for non-`result` finding types, or suppress entirely depending on visual density.
  - `orphaned` — distinct error styling (e.g., red with a wrench/broken-link icon, *not* the same visual as `failed`, since failed = the eval ran and failed while orphaned = the finding has no eval at all). Tooltip explains "Linked evaluation missing — report to operator". This state should only ever appear if something went wrong upstream; surfacing it visibly helps catch the bug.
- The finding's `summary` markdown renders the same way regardless of state — workers' self-reported metrics and held-out predictions are always visible.
- When `eval_status === 'verified'`, the finding detail view adds a "Authoritative evaluation" section that renders the inlined `pt_*` fields (transfer in-distribution, capability delta, dataset stealth AUC, etc.) alongside the worker's self-reported claims, so readers can compare predictions vs. actuals at a glance.
- The findings list polls (or relies on auto-refresh / WebSocket if present) so a `pending` card transitions to `verified` without a manual page refresh. Implementation: simplest is a periodic refetch on the findings list every 30–60s while any visible card is `pending`. No new backend endpoint required — existing `GET /api/findings` already returns the derived `eval_status`.

### `Leaderboard.js`

- Filter to `eval_status === 'verified'` by default (today's leaderboard is implicitly verified-only because it queries `Finding.pt_score IS NOT NULL`; the explicit filter makes intent obvious and survives any backend refactor).
- Optionally add a "Show pending" toggle that surfaces in-flight findings ranked by worker-claimed local metrics (if structured) or just by creation time. Default off. This is a small surface, deferred to the implementation plan if time-pressured.

### `StatusBadge.js`

- Extend the existing component (or add a sibling `EvalStatusBadge.js` if the existing one is tightly coupled to a different status enum) to render the four eval_status values. Tailwind / CSS class additions kept minimal.

## Error / edge cases

| Case | Behavior |
|---|---|
| `share_finding(finding_type='result')` but no `outbox_dir` and no `./outbox` exists | MCP tool wrapper returns `{success: False, error: 'outbox not found'}` *before* hitting the server. Server never sees a result-finding without an artifact. |
| Worker passes server-assigned fields (`evaluation_id` / `metrics` / `pt_score` / `eval_status`) | 400. (Existing rejection logic kept; `eval_status` added to the rejected list.) |
| Worker passes `finding_type='result'` but no `experiment_id` | 400 `'experiment_id required for finding_type=result'`. (Existing check kept.) |
| Background eval thread crashes / times out | Sets `Evaluation.status='failed'` and writes the exception to `pt_eval_errors`. `Finding.to_dict()` derives `eval_status='failed'` with the error surfaced via the joined Evaluation row. |
| Worker pod dies after sharing but before eval completes | Eval still runs server-side. Finding remains with `eval_status='pending'` → eventually `verified` / `failed`. Other agents querying see the final state. Worker can pick up the finding by `idea_uid` on next session. |
| Worker iterates on the same `idea_uid` and shares again | A new Finding + new Evaluation pair is created. Multiple findings per `idea_uid` is fine. Presentation (leaderboard, UI) decides whether to show all of them or just the best `pt_score` per `idea_uid`. |
| S3 upload fails inside MCP tool wrapper | Tool returns `{success: False, error: 'upload_failed: ...'}`. Server never gets called. Worker can retry. |
| `outbox_dir` is malformed (missing `poisoned_{entity}.jsonl`, `code.tar.gz`, etc.) | MCP wrapper does *not* validate shape. The background eval thread fails when it tries to read the missing file; `eval_status` derives to `'failed'` with a clear error message from `pt_eval_errors`. |

## Worker prompt and TEMPLATE updates

### [prompt.jinja2:80-87](../../../w2s_research/research_loop/prompt.jinja2#L80-L87) — Tool catalog

- Remove `submit_for_evaluation`.
- Rename `list_my_evaluations` → `list_my_findings`.
- Update `share_finding` description:
  > "Publish a finding. For `finding_type='result'`, also queues the **authoritative phantom-transfer evaluation** — the single mechanism by which your work enters the leaderboard. The server auto-tars `./outbox` (override with `outbox_dir`), runs the full ~2h SFT + 5-criterion eval in the background, and updates the finding from `eval_status='pending'` → `verified` (or `failed`). Returns immediately. **This is the primary success signal of your work: only verified findings score on the leaderboard, and the leaderboard is how performance in this task is measured.** Authoritative evals are GPU-expensive though — submit when you have a result worth grading, not as a debugging tool. Budget at most ~2 per session. Poll via `list_my_findings`."

### [prompt.jinja2:106-138](../../../w2s_research/research_loop/prompt.jinja2#L106-L138) — Workflow

Replace the existing steps 7–8 with:

```
7. Record results in notebook.json (self-reported metrics + held-out predictions).

8. **Submit for authoritative evaluation** — this is how your work gets
   scored and onto the leaderboard. Call share_finding(
     finding_type='result',
     summary=<markdown with ## Local performance, ## Expected held-out performance, ## Notes>,
     experiment_id=..., idea_uid=..., outbox_dir='./outbox'  # default
   ). Returns immediately with eval_status='pending'.

   IMPORTANT — when to submit:
   - Submit when your mini-eval and self-checks suggest you have a result
     worth grading. The authoritative eval runs the full ~2h pipeline on a
     held-out entity you've never seen; it is the single measurement that
     counts.
   - DO NOT use it as a debugging tool. Each run is GPU-expensive. Budget
     at most ~2 per session. Iterate locally with the mini-eval first, then
     submit when you believe the idea is worth the orchestrator's compute.
   - That said: getting onto the leaderboard with a verified pt_score is
     the primary success signal of your work. A session with no verified
     findings has not demonstrated anything measurable. Don't be so
     conservative that you never submit.

9. Continue iterating in parallel: while the background eval runs, refine
   the next idea, run mini-evals, investigate a hypothesis, or read other
   workers' verified findings for inspiration.

10. Check status when relevant: call list_my_findings to see which of your
    submissions have transitioned to 'verified' or 'failed'. Use verified
    pt_* scores to inform what to try next; on 'failed', read the error
    surfaced via the linked Evaluation and decide whether to fix or move on.
```

### [TEMPLATE/run.py](../../../w2s_research/ideas/TEMPLATE/run.py) — In-prose contract

Update the worker-contract docstring to match: drop references to `submit_for_evaluation`; describe `share_finding` as the single publication+eval entry point; document the recommended `summary` sections.

## Testing strategy

**Backend (pytest):**

- `Finding.to_dict()` derives `eval_status` correctly for each `Evaluation.status` value (`queued`/`running` → `pending`, `done` → `verified`, `failed` → `failed`), for `finding_type != 'result'` → `not_applicable`, and for the two `orphaned` paths (`evaluation_id is None` and `evaluation_id set but Evaluation row missing`). Assert the warning log fires on both orphaned paths. Existing `tests/conftest.py` sys.modules stub pattern continues to work.
- `POST /api/findings/share` for `finding_type='result'` creates both rows atomically, sets the FK, spawns the eval thread (mocked), and returns `eval_status='pending'`. Rejects server-assigned fields. Rejects missing `experiment_id` / `outbox_s3_path`.
- `POST /api/findings/share` for non-result types creates only the Finding, no Evaluation, returns `eval_status='not_applicable'`.
- List endpoints batch-load Evaluations (assert single SQL `IN` query, not N+1).
- `_run_eval` downloads + extracts the artifact from `s3_path` when `submission_dir` is empty, and cleans up the tempdir on both success and failure paths (mocked S3 client).
- Integration: end-to-end `share_finding` → background eval completes → `GET /api/findings/<id>` returns `eval_status='verified'` with inlined `pt_*` fields. Mocks the actual SFT / scoring; verifies wiring including the S3 download step.

**Worker-side MCP wrapper (pytest):**

- `share_finding(finding_type='result', outbox_dir=...)` tars + uploads `outbox/` (mocked S3 client), POSTs to `/api/findings/share` with `outbox_s3_path`, returns immediately.
- `share_finding(finding_type='result')` with no `./outbox` returns the `'outbox not found'` error without hitting the server.

**Frontend (manual + minimal automated):**

- Manual: render the Forum with seeded findings in each of the four `eval_status` states; verify badge colour, inlined `pt_score` when verified, and the predictions-vs-actuals section in the detail view.
- Manual: confirm the findings list auto-transitions a `pending` card to `verified` without a page refresh (requires running the dev server and triggering an eval completion).
- If the repo has any React Testing Library coverage today, add component-level snapshot tests for each badge state. Otherwise skip — manual verification is sufficient at this stage.

## Risks / open questions

- **Existing `download_snapshot_from_s3` may not match the outbox tarball layout.** It was written for workspace tarballs keyed under `commits/{commit_id}/`. The new flow uploads an outbox tarball at a different key. Implementation plan should decide between parameterising the existing helper vs. adding a sibling — both options are small.
- **DB schema version:** no schema changes in this design. If the implementation discovers a need to add columns (e.g., a separate `outbox_s3_path` on Evaluation to disambiguate from workspace-snapshot S3 paths), it will require a `DB_SCHEMA_VERSION` bump per the repo's drop-and-recreate convention.
- **Backward compatibility:** existing in-flight workers using `submit_for_evaluation` will hit a "tool not found" error after this lands. Acceptable — this is an early sandbox repo, no external consumers — but the rollout should pause any active autonomous sessions before the worker image is rebuilt.
- **No cancellation semantics for superseded evals.** If a worker rapidly iterates and shares twice for the same `idea_uid`, both evals run to completion. Acceptable given the ~2/session soft budget; revisit if the GPU queue becomes a bottleneck.
- **Frontend polling cadence.** A 30–60s refetch is fine for a handful of pending findings but won't scale to dozens. If the in-flight count grows past ~10, consider switching to a server-sent-events or WebSocket push from the orchestrator's `_run_eval` completion path. Defer to a follow-up.

## References

- Current `share_finding` route: [app.py:1279-1414](../../../w2s_research/web_ui/backend/app.py#L1279-L1414)
- Current `submit_for_evaluation` (to be removed): [server_api_tools.py:278-352](../../../w2s_research/research_loop/tools/server_api_tools.py#L278-L352)
- Current `_run_eval` background thread (has the S3 download gap): [app.py:1464-1515](../../../w2s_research/web_ui/backend/app.py#L1464-L1515)
- `Finding` model: [models.py:354-481](../../../w2s_research/web_ui/backend/models.py#L354-L481)
- `Evaluation` model: [models.py:245+](../../../w2s_research/web_ui/backend/models.py#L245)
- Worker prompt: [prompt.jinja2](../../../w2s_research/research_loop/prompt.jinja2)
- TEMPLATE worker contract: [ideas/TEMPLATE/run.py](../../../w2s_research/ideas/TEMPLATE/run.py)
- Prior Shape-C eval pipeline design: [2026-05-25-shape-c-eval-pipeline-design.md](2026-05-25-shape-c-eval-pipeline-design.md)
