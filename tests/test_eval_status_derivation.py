"""Finding._compute_eval_status: derive eval_status from joined Evaluation."""


def test_not_applicable_for_non_result_finding(app):
    from w2s_research.web_ui.backend.models import Finding, db
    with app.app_context():
        f = Finding(post_id='p1', finding_type='hypothesis', content='x')
        db.session.add(f); db.session.commit()
        assert f._compute_eval_status() == 'not_applicable'


def test_orphaned_when_evaluation_id_is_none(app, caplog):
    """Result finding with no FK set is orphaned and emits a warning."""
    from w2s_research.web_ui.backend.models import Finding, db
    import logging
    with app.app_context():
        f = Finding(post_id='p2', finding_type='result', content='x', evaluation_id=None)
        db.session.add(f); db.session.commit()
        with caplog.at_level(logging.WARNING):
            assert f._compute_eval_status() == 'orphaned'
        assert any('orphan' in r.message.lower() for r in caplog.records)


def test_orphaned_when_evaluation_row_missing(app, caplog):
    """FK set but linked Evaluation row does not exist; orphaned + warning."""
    from w2s_research.web_ui.backend.models import Finding, db
    import logging
    with app.app_context():
        f = Finding(post_id='p3', finding_type='result', content='x', evaluation_id=99999)
        db.session.add(f); db.session.commit()
        with caplog.at_level(logging.WARNING):
            assert f._compute_eval_status() == 'orphaned'
        assert any('orphan' in r.message.lower() for r in caplog.records)


def test_pending_when_eval_queued(app):
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        ev = Evaluation(experiment_id=exp.id, status='queued', base_model='m',
                        assigned_entities='[]', held_out_entities='[]')
        db.session.add(ev); db.session.flush()
        f = Finding(post_id='p4', finding_type='result', content='x', evaluation_id=ev.id)
        db.session.add(f); db.session.commit()
        assert f._compute_eval_status() == 'pending'


def test_pending_when_eval_running(app):
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        ev = Evaluation(experiment_id=exp.id, status='running', base_model='m',
                        assigned_entities='[]', held_out_entities='[]')
        db.session.add(ev); db.session.flush()
        f = Finding(post_id='p5', finding_type='result', content='x', evaluation_id=ev.id)
        db.session.add(f); db.session.commit()
        assert f._compute_eval_status() == 'pending'


def test_verified_when_eval_done(app):
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        ev = Evaluation(experiment_id=exp.id, status='done', base_model='m',
                        assigned_entities='[]', held_out_entities='[]', pt_score=0.42)
        db.session.add(ev); db.session.flush()
        f = Finding(post_id='p6', finding_type='result', content='x', evaluation_id=ev.id)
        db.session.add(f); db.session.commit()
        assert f._compute_eval_status() == 'verified'


def test_failed_when_eval_failed(app):
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        ev = Evaluation(experiment_id=exp.id, status='failed', base_model='m',
                        assigned_entities='[]', held_out_entities='[]')
        db.session.add(ev); db.session.flush()
        f = Finding(post_id='p7', finding_type='result', content='x', evaluation_id=ev.id)
        db.session.add(f); db.session.commit()
        assert f._compute_eval_status() == 'failed'
