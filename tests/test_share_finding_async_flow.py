"""Tests for the unified share_finding route: atomic create + async eval + trust model."""
import pytest


def _make_experiment(app):
    """Create and persist a running Experiment, returning its id."""
    from w2s_research.web_ui.backend.models import Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running')
        db.session.add(exp); db.session.commit()
        return exp.id


def test_result_finding_creates_linked_queued_evaluation_atomically(client, app, mocker):
    """A result share creates a Finding and a queued Evaluation linked by FK in one request."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Finding, db
    exp_id = _make_experiment(app)
    mocker.patch("threading.Thread")  # suppress the real background eval

    # Act
    resp = client.post('/api/findings/share', json={
        'summary': '## Local performance\n33% UK mention rate.',
        'finding_type': 'result',
        'experiment_id': exp_id,
        'idea_uid': 'autonomous_persona_test',
        'idea_name': 'persona_test',
        'outbox_s3_path': 's3://test-bucket/path/outbox.tar.gz',
    })

    # Assert
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['eval_status'] == 'pending'
    assert body['evaluation_id'] is not None
    assert body['finding_id'] is not None
    with app.app_context():
        finding = db.session.get(Finding, body['finding_id'])
        evaluation = db.session.get(Evaluation, body['evaluation_id'])
        assert finding.evaluation_id == evaluation.id
        assert evaluation.status == 'queued'
        assert evaluation.s3_path == 's3://test-bucket/path/outbox.tar.gz'


def test_result_finding_spawns_daemon_background_eval_thread(client, app, mocker):
    """A result share spawns the authoritative eval as a daemon background thread."""
    # Arrange
    exp_id = _make_experiment(app)
    fake_thread = mocker.patch("threading.Thread")

    # Act
    resp = client.post('/api/findings/share', json={
        'summary': 'x', 'finding_type': 'result',
        'experiment_id': exp_id, 'outbox_s3_path': 's3://b/k',
    })

    # Assert
    assert resp.status_code == 200
    fake_thread.assert_called_once()
    assert fake_thread.call_args.kwargs.get('daemon') is True


def test_result_finding_without_outbox_s3_path_is_rejected(client, app):
    """A result share missing outbox_s3_path is rejected with 400 (no artifact to evaluate)."""
    # Arrange
    exp_id = _make_experiment(app)

    # Act
    resp = client.post('/api/findings/share', json={
        'summary': 'x', 'finding_type': 'result', 'experiment_id': exp_id,
    })

    # Assert
    assert resp.status_code == 400
    assert 'outbox' in resp.get_json()['error'].lower()


def test_result_finding_without_experiment_id_is_rejected(client):
    """A result share missing experiment_id is rejected with 400."""
    # Arrange / Act
    resp = client.post('/api/findings/share', json={
        'summary': 'x', 'finding_type': 'result', 'outbox_s3_path': 's3://b/k',
    })

    # Assert
    assert resp.status_code == 400
    assert 'experiment_id' in resp.get_json()['error'].lower()


def test_non_result_finding_creates_finding_without_evaluation(client, app, mocker):
    """A non-result share creates only a Finding, triggers no eval, and is not_applicable."""
    # Arrange
    from w2s_research.web_ui.backend.models import Finding, db
    fake_thread = mocker.patch("threading.Thread")

    # Act
    resp = client.post('/api/findings/share', json={
        'summary': 'untested idea', 'finding_type': 'hypothesis',
    })

    # Assert
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['eval_status'] == 'not_applicable'
    assert body.get('evaluation_id') is None
    with app.app_context():
        finding = db.session.get(Finding, body['finding_id'])
        assert finding.evaluation_id is None
    fake_thread.assert_not_called()


@pytest.mark.parametrize('forbidden_field,value', [
    ('evaluation_id', 5),
    ('metrics', {'a': 1}),
    ('pt_score', 0.9),
    ('eval_status', 'verified'),
])
def test_server_assigned_field_in_payload_is_rejected(client, forbidden_field, value):
    """A worker cannot self-assign server-owned scoring fields; each is rejected with 400."""
    # Arrange / Act
    resp = client.post('/api/findings/share', json={
        'summary': 'x', 'finding_type': 'hypothesis', forbidden_field: value,
    })

    # Assert
    assert resp.status_code == 400
    assert forbidden_field in resp.get_json()['error'].lower()
