"""
Server API Tools - MCP tools for interacting with the orchestrator server.

Provides tools for:
- Evaluation: Get held-out evaluation scores for a submitted artifact (evals held server-side)
- Knowledge Sharing: Share and query findings (including artifact submissions) from other runs
- Info: Get leaderboard ranked by phantom-transfer score
"""

import asyncio
import json
import os
from typing import Any, Dict, List

from claude_agent_sdk import tool, create_sdk_mcp_server

from .http_utils import get_server_url, async_http_post, async_http_get


@tool(
    "evaluate_predictions",
    "Get held-out evaluation scores for a submitted artifact (the four phantom-transfer "
    "metrics: transfer, capability, dataset-stealth, model-stealth). The actual held-out "
    "evals and audit prompts are held server-side. The legacy 'predictions' integer-array "
    "schema is retained for back-compat with the W2S scaffolding; for phantom-transfer "
    "submissions, use share_finding(finding_type='result') instead and the orchestrator "
    "will trigger the held-out eval on your artifact tuple.",
    {
        "type": "object",
        "properties": {
            "predictions": {"type": "array", "items": {"type": "integer"}},
            "dataset": {"type": "string"},
            "weak_model": {"type": "string"},
            "strong_model": {"type": "string"},
        },
        "required": ["predictions", "dataset", "weak_model", "strong_model"],
    },
)
async def evaluate_predictions(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth held server-side.

    Args:
        args: Dict with keys:
            - predictions: List of predictions to evaluate
            - dataset: Dataset name
            - weak_model: Weak model identifier
            - strong_model: Strong model identifier

    Returns:
        MCP-formatted response with metrics:
        {
            "transfer_acc": float,  # Strong model trained on weak labels
            "pgr": float,  # Performance Gap Recovery
            "correct": int,  # Number of correct predictions
            "total": int,  # Total number of predictions
            "fixed_weak_acc": float,  # Weak model baseline accuracy
            "fixed_strong_acc": float,  # Strong model ceiling accuracy
        }
    """
    try:
        # Unpack args
        predictions = args.get("predictions", [])
        dataset = args.get("dataset", "")
        weak_model = args.get("weak_model", "")
        strong_model = args.get("strong_model", "")

        server_url = get_server_url()

        result = await async_http_post(
            f"{server_url}/api/evaluate-predictions",
            {
                "predictions": predictions,
                "dataset": dataset,
                "weak_model": weak_model,
                "strong_model": strong_model,
            },
            timeout=120,
        )

        if not isinstance(result, dict):
            error_response = {"success": False, "error": "Invalid server response format"}
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(error_response, indent=2)
                }]
            }

        if "error" in result:
            error_response = {"success": False, "error": result["error"]}
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(error_response, indent=2)
                }]
            }

        required_fields = ["transfer_acc", "pgr"]
        missing = [f for f in required_fields if result.get(f) is None]
        if missing:
            error_response = {
                "success": False,
                "error": f"Server response missing required fields: {missing}",
            }
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(error_response, indent=2)
                }]
            }

        response_data = {
            "success": True,
            "transfer_acc": result.get("transfer_acc"),
            "pgr": result.get("pgr"),
            "correct": result.get("correct"),
            "total": result.get("total"),
            "fixed_weak_acc": result.get("fixed_weak_acc"),
            "fixed_strong_acc": result.get("fixed_strong_acc"),
        }

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(response_data, indent=2)
            }]
        }

    except Exception as e:
        error_response = {"success": False, "error": str(e)}
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(error_response, indent=2)
            }]
        }


async def _auto_upload_snapshot(
    title: str,
    metrics: dict,
    config: dict,
) -> dict:
    """
    Upload a workspace snapshot to S3.

    Returns a dict with snapshot metadata (commit_id, s3_path, s3_key,
    parent_commit_id, sequence_number, files_snapshot) on success,
    or an empty dict on failure.  All fields are forwarded to
    /api/findings/share by the caller.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    idea_uid = os.environ.get("IDEA_UID")
    run_id = os.environ.get("RUN_ID")
    idea_name = os.environ.get("IDEA_NAME", "")
    workspace_dir = os.environ.get("WORKSPACE_DIR", "/workspace")

    if not idea_uid or not run_id:
        print("[auto-snapshot] Skipping: IDEA_UID or RUN_ID not set")
        return {}

    try:
        server_url = get_server_url()

        # Determine sequence number from existing findings that have snapshots
        try:
            response = await async_http_post(
                f"{server_url}/api/snapshots/search",
                {"query": f"idea_uid:{idea_uid} run_id:{run_id}", "limit": 1000},
                timeout=30,
            )
            existing_commits = response.get("commits", [])
            run_commits = [
                c for c in existing_commits
                if c.get("idea_uid") == idea_uid and c.get("run_id") == run_id
            ]
            sequence_number = len(run_commits)
            parent_commit_id = run_commits[-1]["commit_id"] if run_commits else None
        except Exception:
            sequence_number = 0
            parent_commit_id = None

        timestamp = datetime.now(timezone.utc).isoformat()
        from w2s_research.infrastructure.s3_utils import generate_commit_id, upload_commit_to_s3

        commit_id = generate_commit_id(
            experiment_id=0,
            sequence_number=sequence_number,
            message=title,
            timestamp=timestamp,
        )

        from w2s_research.config import S3_BUCKET as bucket, S3_IDEAS_PREFIX as prefix

        s3_key, archive_size, files_list = upload_commit_to_s3(
            idea_uid=idea_uid,
            run_id=run_id,
            commit_id=commit_id,
            workspace_dir=Path(workspace_dir),
            metadata={
                "title": title,
                "idea_name": idea_name,
                "metrics": metrics or {},
                "config": config or {},
                "parent_commit_id": parent_commit_id,
                "sequence_number": sequence_number,
            },
            bucket_name=bucket,
            prefix=prefix,
        )

        s3_path = f"s3://{bucket}/{s3_key}"
        print(f"[auto-snapshot] Created snapshot {commit_id} at {s3_path}")
        return {
            "commit_id": commit_id,
            "s3_path": s3_path,
            "s3_key": s3_key,
            "parent_commit_id": parent_commit_id,
            "sequence_number": sequence_number,
            "files_snapshot": files_list,
        }

    except Exception as e:
        print(f"[auto-snapshot] Failed: {e}")
        return {}


@tool(
    "share_finding",
    """As you explore your research direction, share your empirical findings with other workers.

For phantom-transfer submissions, call with finding_type="result". The server will automatically
link your best-scoring completed Evaluation (by experiment_id from env) — you do NOT provide
metrics or evaluation_id directly; the orchestrator computes authoritative held-out scores.

Parameters:
- summary: all important details for your finding (in markdown format): method explanations, key implementation code snippets, analysis, config details.
- title: Short descriptive title for the finding
- idea_name: Name of your current research idea (make it descriptive and concise. Do not use "V2/V3" or the exactly same idea name again.)
- config: Dict with hyperparameters like {"epochs": 3, "lr": 1e-4, "entities": ["uk","reagan","nyc"], ...}
- worked: Boolean - did the approach pass your local self-eval thresholds?
- finding_type: One of "result" (your final/provisional artifact submission for orchestrator scoring),
  "hypothesis" (untested ideas), "insight" (analysis/takeaways), "error" (critical bugs)""",
    {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "title": {"type": "string"},
            "idea_name": {"type": "string"},
            "config": {"type": "object"},
            "worked": {"type": "boolean"},
            "finding_type": {"type": "string"},
        },
        "required": ["summary"],
    },
)
async def share_finding(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Share a finding with other workers and post to forum.

    For finding_type="result", the server auto-links the worker's best-scoring done
    Evaluation (looked up by experiment_id from env). Scores are server-assigned —
    agents must NOT pass metrics or evaluation_id directly.

    Args:
        args: Dict with keys:
            - summary: Full markdown content for the forum post (required)
            - title: Short descriptive title for the finding
            - idea_name: Name of research idea
            - config: Dict with hyperparameters
            - worked: Boolean - did approach improve over baseline?
            - finding_type: Type of finding - "result", "hypothesis", "insight", "error", "observation" (default: "result")

    Returns:
        MCP-formatted response with finding_id, post_id, snapshot_id, evaluation_id, pt_score, etc.
    """
    try:
        # Unpack args
        summary = args.get("summary", "")
        title = args.get("title")
        idea_name = args.get("idea_name")
        config = args.get("config")
        worked = args.get("worked")
        finding_type = args.get("finding_type", "result")

        if isinstance(config, str):
            try:
                config = json.loads(config) if config else None
            except json.JSONDecodeError:
                print(f"[share_finding] Warning: Could not parse config JSON: {config}")
                config = None

        server_url = get_server_url()

        # Auto-snapshot: when sharing a result, upload workspace to S3
        snapshot = {}
        if finding_type == "result":
            auto_title = title or f"Result: {idea_name}"
            snapshot = await _auto_upload_snapshot(
                title=auto_title,
                metrics={},
                config=config,
            )

        # Build payload
        payload = {
            "summary": summary,
            "idea_uid": os.environ.get("IDEA_UID"),
            "run_id": os.environ.get("RUN_ID"),
            "dataset": os.environ.get("DATASET_NAME"),
            "weak_model": os.environ.get("WEAK_MODEL"),
            "strong_model": os.environ.get("STRONG_MODEL"),
            "finding_type": finding_type,
        }

        # Attach experiment_id from env so the server can auto-link the best Evaluation.
        experiment_id = int(os.environ.get("EXPERIMENT_ID", "0") or "0") or None
        if experiment_id is not None:
            payload["experiment_id"] = experiment_id

        # Optional fields (only include if non-None)
        optional_fields = {
            "title": title,
            "idea_name": idea_name,
            "config": config,
            "worked": worked,
            # Snapshot fields from _auto_upload_snapshot
            "commit_id": snapshot.get("commit_id"),
            "s3_path": snapshot.get("s3_path"),
            "s3_key": snapshot.get("s3_key"),
            "parent_commit_id": snapshot.get("parent_commit_id"),
            "sequence_number": snapshot.get("sequence_number"),
            "files_snapshot": snapshot.get("files_snapshot"),
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = value

        result = await async_http_post(
            f"{server_url}/api/findings/share",
            payload,
            timeout=30,
        )

        snapshot_id = snapshot.get("commit_id")
        s3_path = snapshot.get("s3_path")

        message = "Finding shared successfully"
        if snapshot_id:
            message += " (auto-snapshot created)"

        # Save finding locally so this agent can search it immediately
        # (without waiting for the next background sync poll)
        finding_dict = result.get("finding")
        if finding_dict and finding_dict.get("id"):
            try:
                from .findings_sync import save_finding_to_dir
                from w2s_research.config import LOCAL_FINDINGS_DIR
                from pathlib import Path

                saved = save_finding_to_dir(finding_dict, Path(LOCAL_FINDINGS_DIR))
                if saved:
                    message += f" (saved locally: {saved.name})"
            except Exception as e:
                print(f"[share_finding] Warning: local save failed: {e}")

        response_data = {
            "success": True,
            "finding_id": result.get("finding_id"),
            "post_id": result.get("post_id"),
            "snapshot_id": snapshot_id,
            "s3_path": s3_path,
            "evaluation_id": result.get("evaluation_id"),
            "pt_score": result.get("pt_score"),
            "message": message,
        }

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(response_data, indent=2)
            }]
        }

    except Exception as e:

        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"success": False, "error": str(e)}, indent=2)
            }]
        }


@tool(
    "submit_for_evaluation",
    "Submit your artifact (poisoned datasets + code) for authoritative phantom-transfer evaluation. "
    "Blocks ~2 hours while the server runs full SFT + 5-criterion eval against base + clean-pipeline "
    "controls + held-out generalisation. Returns the full pt_* score breakdown when done.",
    {
        "submission_dir": str,
        "s3_path": str,
    },
)
async def submit_for_evaluation(args: Dict[str, Any]) -> Dict[str, Any]:
    """Submit for authoritative eval; poll until done; return scores.

    Args:
        args: {submission_dir?, s3_path?}. One of the two is required.

    Returns:
        MCP-formatted response with {success, evaluation_id, status, pt_score, pt_*}.
    """
    server_url = get_server_url()
    experiment_id = os.environ.get("EXPERIMENT_ID")
    if not experiment_id:
        return {"content": [{"type": "text", "text": json.dumps({
            "success": False, "error": "EXPERIMENT_ID env var not set",
        })}]}

    payload = {
        "experiment_id": int(experiment_id),
        "base_model": os.environ.get("STUDENT_MODEL") or os.environ.get("WEAK_MODEL")
                      or "google/gemma-3-12b-it",
    }
    if args.get("submission_dir"):
        payload["submission_dir"] = args["submission_dir"]
    if args.get("s3_path"):
        payload["s3_path"] = args["s3_path"]

    try:
        post_resp = await async_http_post(
            f"{server_url}/api/evaluations", payload, timeout=30,
        )
    except Exception as e:
        return {"content": [{"type": "text", "text": json.dumps({
            "success": False, "error": f"submit_failed: {e!r}",
        })}]}

    eval_id = post_resp.get("evaluation_id")
    if not eval_id:
        return {"content": [{"type": "text", "text": json.dumps({
            "success": False, "error": "no_evaluation_id_returned", "response": post_resp,
        })}]}

    # Poll. Hard timeout 4h.
    poll_interval = 30
    max_polls = (4 * 3600) // poll_interval
    for _ in range(int(max_polls)):
        try:
            row = await async_http_get(
                f"{server_url}/api/evaluations/{eval_id}", timeout=30,
            )
        except Exception:
            await asyncio.sleep(poll_interval)
            continue
        status = row.get("status")
        if status in ("done", "failed"):
            return {"content": [{"type": "text", "text": json.dumps({
                "success": status == "done",
                "evaluation_id": eval_id,
                **{k: v for k, v in row.items() if k != "evaluation_id"},
            }, indent=2, default=str)}]}
        await asyncio.sleep(poll_interval)

    return {"content": [{"type": "text", "text": json.dumps({
        "success": False, "evaluation_id": eval_id,
        "status": "running", "error": "tool_timeout",
    })}]}


@tool(
    "list_my_evaluations",
    "List all done evaluations submitted from this worker pod. Use this to find earlier "
    "evaluation_ids you may want to reference in your finding summary.",
    {},
)
async def list_my_evaluations(args: Dict[str, Any] = None) -> Dict[str, Any]:
    """List this worker's prior evaluations."""
    server_url = get_server_url()
    experiment_id = os.environ.get("EXPERIMENT_ID")
    if not experiment_id:
        return {"content": [{"type": "text", "text": json.dumps({
            "success": False, "error": "EXPERIMENT_ID env var not set",
        })}]}
    try:
        result = await async_http_get(
            f"{server_url}/api/evaluations?experiment_id={experiment_id}", timeout=30,
        )
    except Exception as e:
        return {"content": [{"type": "text", "text": json.dumps({
            "success": False, "error": f"list_failed: {e!r}",
        })}]}
    return {"content": [{"type": "text", "text": json.dumps({
        "success": True,
        "evaluations": result.get("evaluations", []),
        "count": result.get("total", 0),
    }, indent=2)}]}


@tool(
    "get_leaderboard",
    "Get the leaderboard of best results ranked by phantom-transfer score. See what to beat!",
    {},
)
async def get_leaderboard(args: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get leaderboard of best results.

    Returns:
        MCP-formatted response with leaderboard entries
    """
    try:
        server_url = get_server_url()

        result = await async_http_get(
            f"{server_url}/api/leaderboard",
            timeout=30,
        )

        entries = result.get("findings", [])
        top_pt_score = entries[0].get("pt_score") if entries else 0.0

        response_data = {
            "success": True,
            "entries": entries,
            "top_pt_score": top_pt_score,
            "count": len(entries),
        }

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(response_data, indent=2)
            }]
        }

    except Exception as e:

        error_response = {
            "success": False,
            "error": str(e),
            "entries": [],
            "top_pt_score": 0.0,
            "count": 0,
        }
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(error_response, indent=2)
            }]
        }


def create_server_api_tools_server():
    """Create MCP server with all server API tools."""
    return create_sdk_mcp_server(
        name="server-api-tools",
        version="1.0.0",
        tools=[
            evaluate_predictions,
            share_finding,
            submit_for_evaluation,
            list_my_evaluations,
            get_leaderboard,
        ],
    )
