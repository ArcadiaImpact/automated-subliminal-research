"""share_finding (server side): no eval trigger, auto-link to best-scoring Evaluation."""
import json
from unittest.mock import patch


def test_share_finding_does_NOT_trigger_evaluation(client, app):
    """Regression against the prior bug: share_finding must not call evaluate_phantom_transfer_submission."""
    # Arrange
    from w2s_research.web_ui.backend.models import Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running'); db.session.add(exp); db.session.commit()
        exp_id = exp.id
    payload = {
        'summary': 'test', 'idea_name': 'idea1',
        'finding_type': 'result',
        'experiment_id': exp_id,
    }
    # Act
    with patch(
        "w2s_research.web_ui.backend.evaluation.evaluate_phantom_transfer_submission"
    ) as fake_eval:
        client.post('/api/findings/share', json=payload)

    # Assert
    fake_eval.assert_not_called()


def test_share_finding_auto_links_best_scoring_evaluation(client, app):
    """When multiple done Evaluations exist for the worker, share_finding picks the one
    with the highest pt_score and writes its id onto the Finding."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running'); db.session.add(exp); db.session.flush()
        ev_low = Evaluation(experiment_id=exp.id, status='done', base_model='m',
                            assigned_entities='[]', held_out_entities='[]', pt_score=0.1)
        ev_high = Evaluation(experiment_id=exp.id, status='done', base_model='m',
                             assigned_entities='[]', held_out_entities='[]', pt_score=0.5)
        db.session.add_all([ev_low, ev_high])
        db.session.commit()
        exp_id = exp.id
        ev_high_id = ev_high.id

    # Act
    response = client.post('/api/findings/share', json={
        'summary': 'test', 'idea_name': 'idea1',
        'finding_type': 'result',
        'experiment_id': exp_id,
    })

    # Assert
    assert response.status_code == 200
    body = response.get_json()
    assert body['evaluation_id'] == ev_high_id
    assert body['pt_score'] == 0.5


def test_share_finding_rejects_agent_provided_evaluation_id(client, app):
    """The agent must NOT be able to name an evaluation_id; server returns 400."""
    # Arrange
    from w2s_research.web_ui.backend.models import Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running'); db.session.add(exp); db.session.commit()
        exp_id = exp.id

    # Act
    response = client.post('/api/findings/share', json={
        'summary': 'test', 'finding_type': 'result',
        'experiment_id': exp_id,
        'evaluation_id': 999,  # forbidden
    })

    # Assert
    assert response.status_code == 400
    assert 'evaluation_id' in response.get_json().get('error', '').lower()


def test_share_finding_returns_409_on_duplicate_evaluation(client, app):
    """A second share_finding linking the same Evaluation must return 409, not 500.
    UNIQUE(evaluation_id) on Finding is the enforcement mechanism."""
    # Arrange — one Experiment, one done Evaluation, two share_finding calls.
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running')
        db.session.add(exp); db.session.flush()
        ev = Evaluation(
            experiment_id=exp.id, status='done', base_model='m',
            assigned_entities='[]', held_out_entities='[]', pt_score=0.4,
        )
        db.session.add(ev); db.session.commit()
        exp_id = exp.id

    payload = {
        'summary': 'first', 'idea_name': 'idea1', 'finding_type': 'result',
        'experiment_id': exp_id,
    }

    # Act
    first = client.post('/api/findings/share', json=payload)
    second = client.post('/api/findings/share', json={**payload, 'summary': 'second'})

    # Assert
    assert first.status_code == 200, first.get_data(as_text=True)
    assert second.status_code == 409, second.get_data(as_text=True)
    assert 'evaluation' in (second.get_json() or {}).get('error', '').lower()


def test_share_finding_rejects_agent_provided_metrics(client, app):
    """The agent must NOT be able to pass `metrics` (would be self-reporting); server returns 400."""
    # Arrange
    from w2s_research.web_ui.backend.models import Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running'); db.session.add(exp); db.session.commit()
        exp_id = exp.id

    # Act
    response = client.post('/api/findings/share', json={
        'summary': 'test', 'finding_type': 'result',
        'experiment_id': exp_id,
        'metrics': {'pt_score': 99.9},
    })

    # Assert
    assert response.status_code == 400
