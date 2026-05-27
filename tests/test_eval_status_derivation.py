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


def test_to_dict_includes_eval_status_for_non_result(app):
    from w2s_research.web_ui.backend.models import Finding, db
    with app.app_context():
        f = Finding(post_id='p10', finding_type='insight', content='x')
        db.session.add(f); db.session.commit()
        d = f.to_dict()
        assert d['eval_status'] == 'not_applicable'
        # pt_score stays (it's an existing denormalised cache column on Finding)
        assert 'pt_score' in d
        # but the joined Evaluation pt_* fields are NOT inlined when not verified
        assert 'pt_transfer_in_distribution' not in d


def test_to_dict_includes_eval_status_and_inlines_pt_fields_when_verified(app):
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        ev = Evaluation(
            experiment_id=exp.id, status='done', base_model='m',
            assigned_entities='[]', held_out_entities='[]',
            pt_score=0.42,
            pt_transfer_in_distribution=0.6,
            pt_capability_delta_pp=-1.0,
            pt_dataset_stealth_auc=0.5,
        )
        db.session.add(ev); db.session.flush()
        f = Finding(post_id='p11', finding_type='result', content='x', evaluation_id=ev.id)
        db.session.add(f); db.session.commit()
        d = f.to_dict()
        assert d['eval_status'] == 'verified'
        assert d['pt_transfer_in_distribution'] == 0.6
        assert d['pt_capability_delta_pp'] == -1.0
        assert d['pt_dataset_stealth_auc'] == 0.5
        # top-level pt_score must come from the verified Evaluation, not the
        # never-set Finding.pt_score column.
        assert d['pt_score'] == 0.42


def test_to_dict_inlines_eval_errors_when_failed(app):
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    import json
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        ev = Evaluation(experiment_id=exp.id, status='failed', base_model='m',
                        assigned_entities='[]', held_out_entities='[]',
                        pt_eval_errors=json.dumps(['boom: ValueError']))
        db.session.add(ev); db.session.flush()
        f = Finding(post_id='pf_err', finding_type='result', content='x', evaluation_id=ev.id)
        db.session.add(f); db.session.commit()
        d = f.to_dict()
        assert d['eval_status'] == 'failed'
        assert d['pt_eval_errors'] == ['boom: ValueError']


def test_to_dict_omits_pt_fields_when_pending(app):
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        ev = Evaluation(experiment_id=exp.id, status='running', base_model='m',
                        assigned_entities='[]', held_out_entities='[]')
        db.session.add(ev); db.session.flush()
        f = Finding(post_id='p12', finding_type='result', content='x', evaluation_id=ev.id)
        db.session.add(f); db.session.commit()
        d = f.to_dict()
        assert d['eval_status'] == 'pending'
        assert 'pt_transfer_in_distribution' not in d
