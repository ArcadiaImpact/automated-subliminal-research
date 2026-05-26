"""share_finding (server side): rejection of agent-provided server-assigned fields.

Behaviour tests for the new atomic Finding+Evaluation flow live in
test_share_finding_async_flow.py. This file only retains the trust-model
rejection tests, which remain valid under the new flow.
"""


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
