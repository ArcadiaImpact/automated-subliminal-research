"""submit_for_evaluation MCP tool: posts artifact, polls until done."""
import json

import pytest


async def test_submit_for_evaluation_polls_until_done_returns_scores(mocker, monkeypatch):
    """The tool POSTs once to /api/evaluations, then polls GET /api/evaluations/<id>
    until status='done', then returns the full pt_* dict to the agent."""
    # Arrange
    monkeypatch.setenv("EXPERIMENT_ID", "42")
    monkeypatch.setenv("ORCHESTRATOR_API_URL", "http://test")

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

    mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_post",
        new_callable=mocker.AsyncMock,
        side_effect=fake_post,
    )
    mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_get",
        new_callable=mocker.AsyncMock,
        side_effect=fake_get,
    )
    mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.asyncio.sleep",
        new_callable=mocker.AsyncMock,
        return_value=None,
    )

    # Act
    result = await submit_for_evaluation({"submission_dir": "/tmp/x"})

    # Assert
    body = json.loads(result["content"][0]["text"])
    assert body["success"] is True
    assert body["evaluation_id"] == 7
    assert body["pt_score"] == 0.42
    assert body["status"] == "done"


async def test_submit_for_evaluation_attaches_experiment_id_from_env(mocker, monkeypatch):
    """The POST body must include experiment_id read from the EXPERIMENT_ID env var."""
    # Arrange
    monkeypatch.setenv("EXPERIMENT_ID", "123")
    monkeypatch.setenv("ORCHESTRATOR_API_URL", "http://test")

    from w2s_research.research_loop.tools.server_api_tools import submit_for_evaluation

    captured = {}

    async def fake_post(url, payload, timeout=30):
        captured.update(payload)
        return {'evaluation_id': 1, 'status': 'queued'}

    async def fake_get(url, timeout=30):
        return {'evaluation_id': 1, 'status': 'done', 'pt_score': 0.0}

    mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_post",
        new_callable=mocker.AsyncMock,
        side_effect=fake_post,
    )
    mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_get",
        new_callable=mocker.AsyncMock,
        side_effect=fake_get,
    )
    mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.asyncio.sleep",
        new_callable=mocker.AsyncMock,
        return_value=None,
    )

    # Act
    await submit_for_evaluation({"submission_dir": "/tmp/x"})

    # Assert
    assert captured.get("experiment_id") == 123
