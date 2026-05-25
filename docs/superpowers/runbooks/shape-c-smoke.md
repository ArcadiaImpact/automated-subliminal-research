# Shape C smoke test — runbook

End-to-end verification on a real RunPod H100 before declaring Shape C done.

## Pre-conditions

- Operator has run through `LAUNCH.md` end-to-end on a fresh pod.
- Repository at the tip of the `aristizabal95/fix-pipeline` branch (or wherever Shape C lands).
- `phantom-transfer` cloned as a sibling at `/workspace/phantom-transfer`.
- All API keys exported.

## Steps

1. Start the orchestrator: `python run.py server --port 8000` (in tmux).
2. Queue one seed idea explicitly: `curl -X POST http://localhost:8000/api/queue -d '{"idea_name":"idea1"}' -H 'Content-Type: application/json'`.
3. Worker pod spins up. SSH into it (RunPod dashboard) and confirm:
   - `echo $EXPERIMENT_ID` prints the experiment row id.
   - `echo $PT_ASSIGNED_ENTITIES` prints `uk,reagan,stalin`.
   - `echo $PT_HELD_OUT_ENTITIES` prints empty.
4. Tail worker logs. Expected sequence:
   - Worker implements `poison_dataset()` in `w2s_research/ideas/autonomous_idea1/run.py`.
   - Worker invokes `python -m w2s_research.web_ui.backend.evaluation --mini --submission-dir outbox/`.
   - Mini eval prints JSON with `pt_score`.
   - Worker calls `submit_for_evaluation` MCP tool.
5. On the orchestrator: `curl -s http://localhost:8000/api/evaluations | jq` shows a queued/running row.
6. Wait ~2 hours. Worker's MCP tool returns `{status: 'done', pt_score: X}`.
7. Worker calls `share_finding`. Verify:
   - `curl -s http://localhost:8000/api/leaderboard | jq '.findings[0]'` shows the new finding.
   - `finding.evaluation_id` matches the Evaluation row.
   - `finding.pt_score` matches `Evaluation.pt_score`.
8. SQL audit:
   ```sql
   SELECT id, experiment_id, pt_score, status FROM evaluations;
   SELECT id, evaluation_id, experiment_id, finding_type FROM findings;
   ```
   Confirm 1:1 binding (`evaluations.id = findings.evaluation_id` for exactly one Finding).

## Acceptance

- One Evaluation row reaches `status='done'` with a non-None `pt_score`.
- One Finding row references that Evaluation.
- Leaderboard returns this finding.
- Worker pod logs show no Python tracebacks or 5xx HTTP errors.
- `pt_transfer_generalisation` is populated (the held-out eval actually ran).
