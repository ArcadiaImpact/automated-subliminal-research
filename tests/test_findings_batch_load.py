"""Findings list endpoints batch-load Evaluations (no N+1)."""


def test_findings_list_endpoint_does_not_n_plus_one(client, app, mocker):
    """GET /api/findings must load all linked Evaluations without per-finding db.session.get."""
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        for i in range(5):
            ev = Evaluation(
                experiment_id=exp.id, status='done', base_model='m',
                assigned_entities='[]', held_out_entities='[]', pt_score=0.1 * i,
            )
            db.session.add(ev); db.session.flush()
            f = Finding(post_id=f'p_n{i}', finding_type='result',
                        content=f'finding{i}', evaluation_id=ev.id)
            db.session.add(f)
        db.session.commit()

    # Spy on db.session.get to count per-finding Evaluation lookups.
    from w2s_research.web_ui.backend.models import db as backend_db
    original_get = backend_db.session.get
    eval_get_calls = []
    def _spy(model, *args, **kwargs):
        if model is Evaluation:
            eval_get_calls.append((args, kwargs))
        return original_get(model, *args, **kwargs)
    mocker.patch.object(backend_db.session, 'get', side_effect=_spy)

    resp = client.get('/api/findings?limit=10')
    assert resp.status_code == 200
    body = resp.get_json()
    findings = body.get('findings') if isinstance(body, dict) else body
    verified = [f for f in findings if f.get('eval_status') == 'verified']
    assert len(verified) == 5

    assert len(eval_get_calls) == 0, (
        f"N+1 detected: {len(eval_get_calls)} per-finding Evaluation lookups"
    )
