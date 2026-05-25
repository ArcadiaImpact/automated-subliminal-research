"""Evaluation model — column presence, nullability, indices."""
import pytest


def test_evaluation_model_requires_experiment_id_not_null(app):
    """Evaluation.experiment_id is the worker-identity binding (spec §4.5);
    inserting an Evaluation without it must raise IntegrityError."""
    # Arrange
    from sqlalchemy.exc import IntegrityError
    from w2s_research.web_ui.backend.models import Evaluation, db
    with app.app_context():
        ev = Evaluation(
            status='queued',
            base_model='google/gemma-3-12b-it',
            assigned_entities='["uk","reagan","stalin"]',
            held_out_entities='["catholicism"]',
            # experiment_id deliberately omitted
        )
        db.session.add(ev)

        # Act / Assert
        with pytest.raises(IntegrityError):
            db.session.commit()
        db.session.rollback()


def test_evaluation_model_persists_all_pt_columns(app):
    """A complete Evaluation row round-trips all pt_* score columns."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='idea1', status='running')
        db.session.add(exp)
        db.session.flush()

        ev = Evaluation(
            experiment_id=exp.id,
            status='done',
            base_model='google/gemma-3-12b-it',
            assigned_entities='["uk","reagan","stalin"]',
            held_out_entities='["catholicism"]',
            pt_score=0.42,
            pt_transfer_in_distribution=0.5,
            pt_transfer_generalisation=0.3,
            pt_capability_delta_pp=-0.5,
        )

        # Act
        db.session.add(ev)
        db.session.commit()
        fetched = db.session.query(Evaluation).filter_by(id=ev.id).first()

        # Assert
        assert fetched.pt_score == 0.42
        assert fetched.pt_transfer_in_distribution == 0.5
        assert fetched.pt_transfer_generalisation == 0.3
        assert fetched.pt_capability_delta_pp == -0.5
        assert fetched.experiment_id == exp.id
