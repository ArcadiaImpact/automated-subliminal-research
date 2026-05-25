"""Finding↔Evaluation 1:1 linkage (spec §4.5 #4)."""
import pytest


def test_finding_evaluation_id_is_unique(app):
    """Two Findings cannot reference the same Evaluation; second insert raises IntegrityError."""
    # Arrange
    from sqlalchemy.exc import IntegrityError
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running')
        db.session.add(exp)
        db.session.flush()
        ev = Evaluation(
            experiment_id=exp.id, status='done',
            base_model='m', assigned_entities='[]', held_out_entities='[]',
            pt_score=0.1,
        )
        db.session.add(ev)
        db.session.flush()

        f1 = Finding(idea_name='idea1', finding_type='result',
                     evaluation_id=ev.id, experiment_id=exp.id, summary='one')
        db.session.add(f1)
        db.session.commit()

        f2 = Finding(idea_name='idea1', finding_type='result',
                     evaluation_id=ev.id, experiment_id=exp.id, summary='two')
        db.session.add(f2)

        # Act / Assert
        with pytest.raises(IntegrityError):
            db.session.commit()
        db.session.rollback()


def test_experiment_persists_assigned_entities_json(app):
    """Experiment.assigned_entities round-trips a JSON list."""
    # Arrange
    import json
    from w2s_research.web_ui.backend.models import Experiment, db
    with app.app_context():
        exp = Experiment(
            idea_name='idea1', status='queued',
            assigned_entities=json.dumps(["uk", "reagan", "stalin"]),
        )
        db.session.add(exp)
        db.session.commit()

        # Act
        fetched = db.session.query(Experiment).filter_by(id=exp.id).first()

        # Assert
        assert json.loads(fetched.assigned_entities) == ["uk", "reagan", "stalin"]
