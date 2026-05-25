"""GET /api/leaderboard: sorts by Evaluation.pt_score desc, joins Finding↔Evaluation."""


def test_leaderboard_sorts_by_pt_score_descending(client, app):
    """Three findings with pt_scores 0.1, 0.5, 0.3 must appear in order [0.5, 0.3, 0.1]."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='done'); db.session.add(exp); db.session.flush()
        for score in [0.1, 0.5, 0.3]:
            ev = Evaluation(
                experiment_id=exp.id, status='done', base_model='m',
                assigned_entities='[]', held_out_entities='[]', pt_score=score,
            )
            db.session.add(ev); db.session.flush()
            f = Finding(
                idea_name=f'idea_{score}', finding_type='result',
                evaluation_id=ev.id, experiment_id=exp.id, summary='x',
            )
            db.session.add(f)
        db.session.commit()

    # Act
    response = client.get('/api/leaderboard')

    # Assert
    body = response.get_json()
    scores = [f['pt_score'] for f in body['findings']]
    assert scores == [0.5, 0.3, 0.1]


def test_leaderboard_skips_findings_without_done_evaluation(client, app):
    """Findings whose linked Evaluation is not status='done' must NOT appear."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='done'); db.session.add(exp); db.session.flush()
        ev_queued = Evaluation(experiment_id=exp.id, status='queued', base_model='m',
                               assigned_entities='[]', held_out_entities='[]')
        ev_done = Evaluation(experiment_id=exp.id, status='done', base_model='m',
                             assigned_entities='[]', held_out_entities='[]', pt_score=0.4)
        db.session.add_all([ev_queued, ev_done]); db.session.flush()
        f_done = Finding(idea_name='good', finding_type='result',
                         evaluation_id=ev_done.id, experiment_id=exp.id, summary='x')
        db.session.add(f_done); db.session.commit()

    # Act
    response = client.get('/api/leaderboard')

    # Assert
    body = response.get_json()
    assert len(body['findings']) == 1
    assert body['findings'][0]['pt_score'] == 0.4
