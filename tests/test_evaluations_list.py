"""GET /api/evaluations?experiment_id=<id>: list a worker's evals, scrubbed."""
import json


def test_list_evaluations_by_experiment_id_returns_descending_pt_score(client, app):
    """Lists all done evals for an experiment, sorted by pt_score desc."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        for s in [0.2, 0.7, 0.4]:
            db.session.add(Evaluation(
                experiment_id=exp.id, status='done', base_model='m',
                assigned_entities='[]', held_out_entities='["catholicism"]', pt_score=s,
            ))
        db.session.commit()
        exp_id = exp.id

    # Act
    response = client.get(f'/api/evaluations?experiment_id={exp_id}')

    # Assert
    body = response.get_json()
    assert [r['pt_score'] for r in body['evaluations']] == [0.7, 0.4, 0.2]


def test_list_evaluations_does_not_leak_held_out_entity(client, app):
    """Listed evals must not contain the string 'catholicism' anywhere in the JSON response."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        db.session.add(Evaluation(
            experiment_id=exp.id, status='done', base_model='m',
            assigned_entities='[]', held_out_entities='["catholicism"]', pt_score=0.5,
            pt_raw_json=json.dumps({"raw": {"per_held_out_entity": {"catholicism": {"lift": 0.3}}}}),
        ))
        db.session.commit()
        exp_id = exp.id

    # Act
    response = client.get(f'/api/evaluations?experiment_id={exp_id}')

    # Assert
    assert 'catholicism' not in response.get_data(as_text=True).lower()
