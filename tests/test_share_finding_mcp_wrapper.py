"""Tests for the worker-side share_finding / list_my_findings MCP tool wrappers."""
import asyncio
import json


def _decode(result):
    """Extract and JSON-parse the text payload from an MCP tool response."""
    return json.loads(result['content'][0]['text'])


def test_result_share_forwards_outbox_and_workspace_snapshot(tmp_path, mocker, monkeypatch):
    """For a result finding, the wrapper forwards both the outbox_s3_path (for eval) and the
    workspace-snapshot fields (for download_snapshot) to the server."""
    # Arrange
    outbox = tmp_path / "outbox"; outbox.mkdir()
    (outbox / "poisoned_uk.jsonl").write_text('{}\n')
    (outbox / "description.md").write_text("# desc\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SERVER_URL", "http://test-server")
    monkeypatch.setenv("EXPERIMENT_ID", "42")
    monkeypatch.setenv("IDEA_UID", "autonomous_t")
    monkeypatch.setenv("RUN_ID", "r1")
    fake_s3_path = "s3://test-bucket/outboxes/abc.tar.gz"
    fake_upload = mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools._upload_outbox_to_s3",
        new_callable=mocker.AsyncMock, return_value=fake_s3_path,
    )
    mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools._auto_upload_snapshot",
        new_callable=mocker.AsyncMock,
        return_value={'commit_id': 'c1', 's3_path': 's3://snap/workspace.tar.gz',
                      'files_snapshot': ['outbox/poisoned_uk.jsonl']},
    )
    fake_post = mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_post",
        new_callable=mocker.AsyncMock,
        return_value={'finding_id': 1, 'post_id': 'p', 'evaluation_id': 7,
                      'eval_status': 'pending', 'finding': {}},
    )
    from w2s_research.research_loop.tools.server_api_tools import share_finding

    # Act
    result = asyncio.run(share_finding({
        'summary': '## Local performance\n33%',
        'finding_type': 'result', 'idea_name': 'persona_test',
    }))

    # Assert
    fake_upload.assert_called_once()
    assert 'outbox' in str(fake_upload.call_args)
    fake_post.assert_called_once()
    posted_payload = fake_post.call_args.args[1]  # async_http_post(url, payload, timeout=...)
    assert posted_payload['outbox_s3_path'] == fake_s3_path
    assert posted_payload['finding_type'] == 'result'
    assert posted_payload['commit_id'] == 'c1'  # workspace snapshot still forwarded
    assert posted_payload['s3_path'] == 's3://snap/workspace.tar.gz'
    body = _decode(result)
    assert body['success'] is True
    assert body['eval_status'] == 'pending'


def test_result_share_fails_without_calling_server_when_outbox_absent(tmp_path, mocker, monkeypatch):
    """A result share with no ./outbox returns an error and never contacts the server."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SERVER_URL", "http://test-server")
    monkeypatch.setenv("EXPERIMENT_ID", "42")
    fake_post = mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_post"
    )
    from w2s_research.research_loop.tools.server_api_tools import share_finding

    # Act
    result = asyncio.run(share_finding({'summary': 'x', 'finding_type': 'result'}))

    # Assert
    body = _decode(result)
    assert body['success'] is False
    assert 'outbox' in body['error'].lower()
    fake_post.assert_not_called()


def test_non_result_share_skips_outbox_upload(tmp_path, mocker, monkeypatch):
    """A non-result share posts the finding without attempting any outbox upload."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SERVER_URL", "http://test-server")
    fake_upload = mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools._upload_outbox_to_s3",
        new_callable=mocker.AsyncMock,
    )
    fake_post = mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_post",
        new_callable=mocker.AsyncMock,
        return_value={'finding_id': 1, 'post_id': 'p',
                      'eval_status': 'not_applicable', 'finding': {}},
    )
    from w2s_research.research_loop.tools.server_api_tools import share_finding

    # Act
    asyncio.run(share_finding({'summary': 'just an idea', 'finding_type': 'hypothesis'}))

    # Assert
    fake_upload.assert_not_called()
    fake_post.assert_called_once()


def test_list_my_findings_queries_findings_endpoint(mocker, monkeypatch):
    """list_my_findings polls the /api/findings endpoint and returns the findings list."""
    # Arrange
    monkeypatch.setenv("IDEA_UID", "autonomous_t")
    monkeypatch.setenv("EXPERIMENT_ID", "42")
    monkeypatch.setenv("SERVER_URL", "http://test-server")
    fake_get = mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_get",
        new_callable=mocker.AsyncMock,
        return_value={'findings': [
            {'id': 1, 'idea_name': 'x', 'eval_status': 'verified', 'pt_score': 0.4, 'evaluation_id': 5},
            {'id': 2, 'idea_name': 'y', 'eval_status': 'pending', 'pt_score': None, 'evaluation_id': 6},
        ]},
    )
    from w2s_research.research_loop.tools.server_api_tools import list_my_findings

    # Act
    result = asyncio.run(list_my_findings({}))

    # Assert
    body = _decode(result)
    assert body['success'] is True
    assert len(body['findings']) == 2
    fake_get.assert_called_once()
    assert '/api/findings' in fake_get.call_args.args[0]


def test_submit_for_evaluation_tool_is_removed():
    """The obsolete submit_for_evaluation tool no longer exists in the module namespace."""
    # Arrange / Act
    from w2s_research.research_loop.tools import server_api_tools

    # Assert
    assert not hasattr(server_api_tools, 'submit_for_evaluation')


def test_list_my_evaluations_name_is_removed():
    """The old list_my_evaluations name no longer exists (renamed to list_my_findings)."""
    # Arrange / Act
    from w2s_research.research_loop.tools import server_api_tools

    # Assert
    assert not hasattr(server_api_tools, 'list_my_evaluations')
