"""Worker-side share_finding wrapper: tars + uploads outbox to S3, passes outbox_s3_path."""
import json
import asyncio


def test_share_finding_result_tars_and_uploads_outbox(tmp_path, mocker, monkeypatch):
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    (outbox / "poisoned_uk.jsonl").write_text('{}\n')
    (outbox / "targets.jsonl").write_text('{}\n')
    (outbox / "code.tar.gz").write_bytes(b"\x1f\x8b")
    (outbox / "description.md").write_text("# desc\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SERVER_URL", "http://test-server")
    monkeypatch.setenv("EXPERIMENT_ID", "42")
    monkeypatch.setenv("IDEA_UID", "autonomous_t")
    monkeypatch.setenv("RUN_ID", "r1")

    fake_s3_path = "s3://test-bucket/outboxes/abc.tar.gz"
    fake_upload = mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools._upload_outbox_to_s3",
        new_callable=mocker.AsyncMock,
        return_value=fake_s3_path,
    )
    fake_post = mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_post",
        new_callable=mocker.AsyncMock,
        return_value={
            'finding_id': 1, 'post_id': 'p', 'evaluation_id': 7,
            'eval_status': 'pending', 'finding': {},
        },
    )

    from w2s_research.research_loop.tools.server_api_tools import share_finding
    result = asyncio.run(share_finding({
        'summary': '## Local performance\n33%',
        'finding_type': 'result',
        'idea_name': 'persona_test',
    }))

    fake_upload.assert_called_once()
    assert 'outbox' in str(fake_upload.call_args)

    fake_post.assert_called_once()
    # payload is the 2nd positional arg to async_http_post(url, payload, timeout=...)
    posted_payload = fake_post.call_args.args[1]
    assert posted_payload['outbox_s3_path'] == fake_s3_path
    assert posted_payload['finding_type'] == 'result'

    body = json.loads(result['content'][0]['text'])
    assert body['success'] is True
    assert body['eval_status'] == 'pending'


def test_share_finding_result_errors_when_outbox_missing(tmp_path, mocker, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SERVER_URL", "http://test-server")
    monkeypatch.setenv("EXPERIMENT_ID", "42")
    fake_post = mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_post",
    )

    from w2s_research.research_loop.tools.server_api_tools import share_finding
    result = asyncio.run(share_finding({
        'summary': 'x', 'finding_type': 'result',
    }))
    body = json.loads(result['content'][0]['text'])
    assert body['success'] is False
    assert 'outbox' in body['error'].lower()
    fake_post.assert_not_called()


def test_share_finding_non_result_skips_outbox_upload(tmp_path, mocker, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SERVER_URL", "http://test-server")
    fake_upload = mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools._upload_outbox_to_s3",
        new_callable=mocker.AsyncMock,
    )
    fake_post = mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_post",
        new_callable=mocker.AsyncMock,
        return_value={
            'finding_id': 1, 'post_id': 'p', 'eval_status': 'not_applicable', 'finding': {},
        },
    )

    from w2s_research.research_loop.tools.server_api_tools import share_finding
    asyncio.run(share_finding({
        'summary': 'just an idea',
        'finding_type': 'hypothesis',
    }))
    fake_upload.assert_not_called()
    fake_post.assert_called_once()


def test_list_my_findings_exists_and_calls_findings_endpoint(mocker, monkeypatch):
    """Tool now polls findings, not evaluations."""
    monkeypatch.setenv("IDEA_UID", "autonomous_t")
    monkeypatch.setenv("EXPERIMENT_ID", "42")
    monkeypatch.setenv("SERVER_URL", "http://test-server")

    fake_get = mocker.patch(
        "w2s_research.research_loop.tools.server_api_tools.async_http_get",
        new_callable=mocker.AsyncMock,
        return_value={
            'findings': [
                {'id': 1, 'idea_name': 'x', 'eval_status': 'verified', 'pt_score': 0.4, 'evaluation_id': 5},
                {'id': 2, 'idea_name': 'y', 'eval_status': 'pending', 'pt_score': None, 'evaluation_id': 6},
            ]
        },
    )

    from w2s_research.research_loop.tools.server_api_tools import list_my_findings
    import json
    result = asyncio.run(list_my_findings({}))
    body = json.loads(result['content'][0]['text'])
    assert body['success'] is True
    assert len(body['findings']) == 2
    fake_get.assert_called_once()
    url_arg = fake_get.call_args.args[0]
    assert '/api/findings' in url_arg


def test_submit_for_evaluation_tool_no_longer_exists():
    """The old tool must be gone from the module namespace."""
    from w2s_research.research_loop.tools import server_api_tools
    assert not hasattr(server_api_tools, 'submit_for_evaluation'), (
        "submit_for_evaluation should be removed; share_finding is the only entry point"
    )


def test_list_my_evaluations_renamed_away():
    """Old name should no longer exist."""
    from w2s_research.research_loop.tools import server_api_tools
    assert not hasattr(server_api_tools, 'list_my_evaluations')
