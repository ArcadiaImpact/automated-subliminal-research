"""POST /api/evaluations + GET /api/evaluations/<id>: submission & polling."""
import json


def test_post_evaluations_creates_queued_row_and_returns_id(
    client, app, sample_submission_dir, mocker,
):
    """POST /api/evaluations creates an Evaluation with status='queued' and returns the id.
    The background thread is mocked out so the test stays deterministic."""
    # Arrange — prevent the real background eval from running.
    mocker.patch("threading.Thread")
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running')
        db.session.add(exp)
        db.session.commit()
        exp_id = exp.id

    # Act
    response = client.post('/api/evaluations', json={
        'submission_dir': str(sample_submission_dir),
        'base_model': 'test-model',
        'experiment_id': exp_id,
        'mini': True,
    })

    # Assert
    assert response.status_code == 202, response.get_data(as_text=True)
    body = response.get_json()
    assert 'evaluation_id' in body
    with app.app_context():
        row = db.session.query(Evaluation).filter_by(id=body['evaluation_id']).first()
        assert row is not None
        assert row.status == 'queued'
        assert row.experiment_id == exp_id


def test_post_evaluations_rejects_missing_experiment_id(client):
    """POST /api/evaluations without experiment_id returns 400."""
    # Arrange / Act
    response = client.post('/api/evaluations', json={
        'submission_dir': '/tmp/nonexistent',
        'base_model': 'm',
    })

    # Assert
    assert response.status_code == 400


def test_get_evaluations_returns_row_dict(client, app):
    """GET /api/evaluations/<id> returns the Evaluation's to_dict() JSON."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running')
        db.session.add(exp)
        db.session.flush()
        ev = Evaluation(
            experiment_id=exp.id, status='done',
            base_model='m', assigned_entities='["uk"]', held_out_entities='["catholicism"]',
            pt_score=0.33,
        )
        db.session.add(ev)
        db.session.commit()
        ev_id = ev.id

    # Act
    response = client.get(f'/api/evaluations/{ev_id}')

    # Assert
    assert response.status_code == 200
    body = response.get_json()
    assert body['pt_score'] == 0.33
    assert body['status'] == 'done'


def test_get_evaluations_scrubs_held_out_by_default(client, app):
    """GET /api/evaluations/<id> uses scrub_held_out=True; response must not contain the held-out entity name."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running')
        db.session.add(exp); db.session.flush()
        ev = Evaluation(
            experiment_id=exp.id, status='done',
            base_model='m', assigned_entities='["uk"]', held_out_entities='["catholicism"]',
            pt_score=0.33,
            pt_raw_json=json.dumps({"raw": {"per_held_out_entity": {"catholicism": {"lift": 0.2}}}}),
        )
        db.session.add(ev); db.session.commit()
        ev_id = ev.id

    # Act
    response = client.get(f'/api/evaluations/{ev_id}')

    # Assert
    body_str = response.get_data(as_text=True)
    assert 'catholicism' not in body_str.lower()
