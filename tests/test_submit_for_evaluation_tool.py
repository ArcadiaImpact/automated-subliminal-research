"""submit_for_evaluation MCP tool: posts artifact, polls until done."""
import asyncio
import json
import os
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch


def _ensure_claude_agent_sdk_mocked():
    """Inject a stub claude_agent_sdk into sys.modules if the real one isn't installed."""
    if "claude_agent_sdk" not in sys.modules:
        stub = ModuleType("claude_agent_sdk")

        def tool(name, description, schema):
            """Passthrough decorator that leaves the function unchanged."""
            def decorator(fn):
                return fn
            return decorator

        stub.tool = tool
        stub.create_sdk_mcp_server = MagicMock()
        sys.modules["claude_agent_sdk"] = stub


_ensure_claude_agent_sdk_mocked()


def test_submit_for_evaluation_polls_until_done_returns_scores():
    """The tool POSTs once to /api/evaluations, then polls GET /api/evaluations/<id>
    until status='done', then returns the full pt_* dict to the agent."""
    # Arrange
    os.environ["EXPERIMENT_ID"] = "42"
    os.environ["ORCHESTRATOR_API_URL"] = "http://test"

    from w2s_research.research_loop.tools.server_api_tools import submit_for_evaluation

    post_response = {'evaluation_id': 7, 'status': 'queued'}
    poll_responses = [
        {'evaluation_id': 7, 'status': 'queued', 'pt_score': None},
        {'evaluation_id': 7, 'status': 'running', 'pt_score': None},
        {'evaluation_id': 7, 'status': 'done', 'pt_score': 0.42,
         'pt_transfer_in_distribution': 0.5},
    ]

    async def fake_post(url, payload, timeout=30):
        return post_response

    async def fake_get(url, timeout=30):
        return poll_responses.pop(0)

    # Act
    with patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_post",
        new=AsyncMock(side_effect=fake_post),
    ), patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_get",
        new=AsyncMock(side_effect=fake_get),
    ), patch(
        "w2s_research.research_loop.tools.server_api_tools.asyncio.sleep",
        new=AsyncMock(return_value=None),
    ):
        result = asyncio.run(submit_for_evaluation({"submission_dir": "/tmp/x"}))

    # Assert
    body = json.loads(result["content"][0]["text"])
    assert body["success"] is True
    assert body["evaluation_id"] == 7
    assert body["pt_score"] == 0.42
    assert body["status"] == "done"


def test_submit_for_evaluation_attaches_experiment_id_from_env():
    """The POST body must include experiment_id read from the EXPERIMENT_ID env var."""
    # Arrange
    os.environ["EXPERIMENT_ID"] = "123"
    os.environ["ORCHESTRATOR_API_URL"] = "http://test"
    from w2s_research.research_loop.tools.server_api_tools import submit_for_evaluation

    captured = {}
    async def fake_post(url, payload, timeout=30):
        captured.update(payload)
        return {'evaluation_id': 1, 'status': 'queued'}

    async def fake_get(url, timeout=30):
        return {'evaluation_id': 1, 'status': 'done', 'pt_score': 0.0}

    # Act
    with patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_post",
        new=AsyncMock(side_effect=fake_post),
    ), patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_get",
        new=AsyncMock(side_effect=fake_get),
    ), patch(
        "w2s_research.research_loop.tools.server_api_tools.asyncio.sleep",
        new=AsyncMock(return_value=None),
    ):
        asyncio.run(submit_for_evaluation({"submission_dir": "/tmp/x"}))

    # Assert
    assert captured.get("experiment_id") == 123
