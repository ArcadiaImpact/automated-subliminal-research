"""
Server API Tools - MCP tools for interacting with the orchestrator server.

Provides tools for:
- Knowledge Sharing: Share findings (including artifact submissions, which queue the
  authoritative held-out evaluation) and list your recent findings with eval_status
- Info: Get leaderboard ranked by phantom-transfer score
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

from claude_agent_sdk import tool, create_sdk_mcp_server

from .http_utils import get_server_url, async_http_post, async_http_get


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


async def _upload_outbox_to_s3(outbox_dir: Path) -> str:
    """Tar+gzip an outbox dir and upload to S3. Returns the s3:// URI.

    Mirrors the key-construction pattern of _auto_upload_snapshot
    (S3_IDEAS_PREFIX/{idea_uid}/{run_id}/...) but under an
    outboxes/{commit_id}/outbox.tar.gz path.
    """
    import tarfile
    import tempfile
    from datetime import datetime, timezone

    from w2s_research.infrastructure.s3_utils import get_s3_client, generate_commit_id
    from w2s_research.config import S3_IDEAS_PREFIX, S3_BUCKET

    idea_uid = os.environ.get("IDEA_UID", "unknown")
    run_id = os.environ.get("RUN_ID", "default")
    experiment_id = int(os.environ.get("EXPERIMENT_ID", "0") or "0")
    timestamp = datetime.now(timezone.utc).isoformat()
    commit_id = generate_commit_id(
        experiment_id=experiment_id,
        sequence_number=0,
        message="outbox",
        timestamp=timestamp,
    )

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with tarfile.open(tmp_path, "w:gz") as tf:
            for p in sorted(outbox_dir.rglob("*")):
                if p.is_file():
                    tf.add(p, arcname=str(p.relative_to(outbox_dir)))

        key = (
            f"{S3_IDEAS_PREFIX}{idea_uid}/{run_id}/"
            f"outboxes/{commit_id}/outbox.tar.gz"
        )
        client = get_s3_client()
        if client is None:
            raise RuntimeError(
                "S3 client unavailable (AWS credentials not configured)"
            )
        client.upload_file(tmp_path, S3_BUCKET, key)
        return f"s3://{S3_BUCKET}/{key}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@tool(
    "share_finding",
    """As you explore your research direction, share your empirical findings with other workers.

For phantom-transfer submissions, call with finding_type="result". This packages your ./outbox
directory (poisoned datasets + code + description), uploads it to S3, and queues the authoritative
held-out evaluation. It returns immediately with eval_status="pending" — you do NOT provide
metrics or evaluation_id directly; the orchestrator computes authoritative held-out scores.

Parameters:
- summary: all important details for your finding (in markdown format): method explanations, key implementation code snippets, analysis, config details.
- title: Short descriptive title for the finding
- idea_name: Name of your current research idea (make it descriptive and concise. Do not use "V2/V3" or the exactly same idea name again.)
- config: Dict with hyperparameters like {"epochs": 3, "lr": 1e-4, "entities": ["uk","reagan","nyc"], ...}
- worked: Boolean - did the approach pass your local self-eval thresholds?
- outbox_dir: Path to your artifact directory (default: ./outbox). Only used for finding_type="result".
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
            "outbox_dir": {"type": "string"},
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
        MCP-formatted response with finding_id, post_id, evaluation_id, eval_status, etc.
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

        # For result findings: package the outbox dir and upload to S3.
        # Do this BEFORE touching the server so a missing outbox fails fast.
        outbox_s3_path = None
        snapshot = {}
        if finding_type == "result":
            outbox_dir = (
                Path(args["outbox_dir"]) if args.get("outbox_dir")
                else Path.cwd() / "outbox"
            )
            if not outbox_dir.exists():
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(
                            {"success": False, "error": f"outbox not found at {outbox_dir}"},
                            indent=2,
                        ),
                    }]
                }
            outbox_s3_path = await _upload_outbox_to_s3(outbox_dir)
            # Also snapshot the full workspace so other workers can download and
            # build on it (powers the download_snapshot tool + Forum file browser).
            # This is distinct from the outbox tarball, which only feeds the eval.
            snapshot = await _auto_upload_snapshot(
                title=title or f"Result: {idea_name}",
                metrics={},
                config=config,
            )

        server_url = get_server_url()

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

        # Attach experiment_id from env; the server uses it to create the Evaluation
        # for finding_type='result'.
        experiment_id = int(os.environ.get("EXPERIMENT_ID", "0") or "0") or None
        if experiment_id is not None:
            payload["experiment_id"] = experiment_id

        # Optional fields (only include if non-None)
        optional_fields = {
            "title": title,
            "idea_name": idea_name,
            "config": config,
            "worked": worked,
            "outbox_s3_path": outbox_s3_path,
            # Workspace snapshot fields (from _auto_upload_snapshot) — let other
            # workers download/browse the full workspace via download_snapshot.
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

        message = "Finding shared successfully"
        if outbox_s3_path:
            message += " (outbox uploaded; eval queued)"

        # Save finding locally so this agent can search it immediately
        # (without waiting for the next background sync poll)
        finding_dict = result.get("finding")
        if finding_dict and finding_dict.get("id"):
            try:
                from .findings_sync import save_finding_to_dir
                from w2s_research.config import LOCAL_FINDINGS_DIR

                saved = save_finding_to_dir(finding_dict, Path(LOCAL_FINDINGS_DIR))
                if saved:
                    message += f" (saved locally: {saved.name})"
            except Exception as e:
                print(f"[share_finding] Warning: local save failed: {e}")

        response_data = {
            "success": True,
            "finding_id": result.get("finding_id"),
            "post_id": result.get("post_id"),
            "evaluation_id": result.get("evaluation_id"),
            "eval_status": result.get("eval_status"),
            "outbox_s3_path": outbox_s3_path,
            "snapshot_id": snapshot.get("commit_id"),
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
    "list_my_findings",
    "List your recent findings (this idea_uid) with their eval_status. "
    "Use this to check whether submissions have transitioned from 'pending' "
    "to 'verified' or 'failed', and to see which ideas you've already tried.",
    {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max findings to return (default 20)"},
        },
    },
)
async def list_my_findings(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return the worker's recent findings with eval_status."""
    server_url = get_server_url()
    idea_uid = os.environ.get("IDEA_UID")
    limit = args.get("limit", 20)
    params = f"?limit={int(limit)}"
    if idea_uid:
        params += f"&idea_uid={idea_uid}"
    try:
        result = await async_http_get(f"{server_url}/api/findings{params}", timeout=30)
    except Exception as e:
        return {"content": [{"type": "text", "text": json.dumps({
            "success": False, "error": f"get_failed: {e!r}",
        })}]}
    findings = result.get("findings") or []
    compact = [
        {
            "finding_id": f.get("id"),
            "idea_name": f.get("idea_name"),
            "eval_status": f.get("eval_status"),
            "pt_score": f.get("pt_score"),
            "evaluation_id": f.get("evaluation_id"),
            "created_at": f.get("created_at"),
        }
        for f in findings
    ]
    return {"content": [{"type": "text", "text": json.dumps({
        "success": True, "findings": compact,
    }, indent=2, default=str)}]}


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
            share_finding,
            list_my_findings,
            get_leaderboard,
        ],
    )
