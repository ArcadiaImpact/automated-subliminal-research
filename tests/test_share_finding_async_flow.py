"""share_finding (new flow): creates Finding+Evaluation atomically and queues async eval."""


def test_share_finding_result_creates_finding_and_evaluation_atomically(client, app, mocker):
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running'); db.session.add(exp); db.session.commit()
        exp_id = exp.id

    mocker.patch("threading.Thread")  # don't actually run the eval

    resp = client.post('/api/findings/share', json={
        'summary': '## Local performance\n33% UK mention rate.',
        'finding_type': 'result',
        'experiment_id': exp_id,
        'idea_uid': 'autonomous_persona_test',
        'idea_name': 'persona_test',
        'outbox_s3_path': 's3://test-bucket/path/outbox.tar.gz',
    })

    assert resp.status_code == 200
    body = resp.get_json()
    assert body['eval_status'] == 'pending'
    assert body['evaluation_id'] is not None
    assert body['finding_id'] is not None

    with app.app_context():
        f = db.session.get(Finding, body['finding_id'])
        ev = db.session.get(Evaluation, body['evaluation_id'])
        assert f.evaluation_id == ev.id
        assert ev.status == 'queued'
        assert ev.s3_path == 's3://test-bucket/path/outbox.tar.gz'


def test_share_finding_result_spawns_background_thread(client, app, mocker):
    from w2s_research.web_ui.backend.models import Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running'); db.session.add(exp); db.session.commit()
        exp_id = exp.id

    fake_thread = mocker.patch("threading.Thread")

    resp = client.post('/api/findings/share', json={
        'summary': 'x', 'finding_type': 'result',
        'experiment_id': exp_id, 'outbox_s3_path': 's3://b/k',
    })
    assert resp.status_code == 200
    fake_thread.assert_called_once()
    assert fake_thread.call_args.kwargs.get('daemon') is True


def test_share_finding_result_requires_outbox_s3_path(client, app):
    from w2s_research.web_ui.backend.models import Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.commit()
        exp_id = exp.id

    resp = client.post('/api/findings/share', json={
        'summary': 'x', 'finding_type': 'result', 'experiment_id': exp_id,
    })
    assert resp.status_code == 400
    assert 'outbox' in resp.get_json()['error'].lower()


def test_share_finding_result_requires_experiment_id(client, app):
    resp = client.post('/api/findings/share', json={
        'summary': 'x', 'finding_type': 'result', 'outbox_s3_path': 's3://b/k',
    })
    assert resp.status_code == 400
    assert 'experiment_id' in resp.get_json()['error'].lower()


def test_share_finding_non_result_creates_only_finding(client, app, mocker):
    from w2s_research.web_ui.backend.models import Finding, db
    fake_thread = mocker.patch("threading.Thread")
    resp = client.post('/api/findings/share', json={
        'summary': 'untested idea',
        'finding_type': 'hypothesis',
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['eval_status'] == 'not_applicable'
    assert body.get('evaluation_id') is None
    with app.app_context():
        f = db.session.get(Finding, body['finding_id'])
        assert f.evaluation_id is None
    fake_thread.assert_not_called()


def test_share_finding_rejects_eval_status_in_payload(client, app):
    resp = client.post('/api/findings/share', json={
        'summary': 'x', 'finding_type': 'hypothesis', 'eval_status': 'verified',
    })
    assert resp.status_code == 400
    assert 'eval_status' in resp.get_json()['error'].lower()


def test_share_finding_still_rejects_evaluation_id_and_metrics(client, app):
    r1 = client.post('/api/findings/share', json={
        'summary': 'x', 'finding_type': 'hypothesis', 'evaluation_id': 5,
    })
    assert r1.status_code == 400
    r2 = client.post('/api/findings/share', json={
        'summary': 'x', 'finding_type': 'hypothesis', 'metrics': {'a': 1},
    })
    assert r2.status_code == 400
