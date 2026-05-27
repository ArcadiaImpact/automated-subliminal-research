"""Tests for Finding.eval_status derivation and its surfacing through to_dict."""
import json
import logging

from hypothesis import given, strategies as st


class _StubEval:
    """Minimal stand-in for an Evaluation row exposing only the .status attr."""

    def __init__(self, status):
        self.status = status


@given(finding_type=st.sampled_from(['hypothesis', 'insight', 'error', 'observation']))
def test_non_result_finding_type_derives_not_applicable(finding_type):
    """Any finding that is not a 'result' derives eval_status='not_applicable'."""
    # Arrange
    from w2s_research.web_ui.backend.models import Finding
    finding = Finding(post_id='p', finding_type=finding_type, content='x')

    # Act
    status = finding._compute_eval_status()

    # Assert
    assert status == 'not_applicable'


@given(eval_status=st.sampled_from(['queued', 'running']))
def test_result_finding_with_active_eval_derives_pending(eval_status):
    """A result finding whose linked evaluation is queued or running derives 'pending'."""
    # Arrange
    from w2s_research.web_ui.backend.models import Finding
    finding = Finding(post_id='p', finding_type='result', evaluation_id=1)

    # Act
    status = finding._compute_eval_status(eval_row=_StubEval(eval_status))

    # Assert
    assert status == 'pending'


def test_result_finding_with_done_eval_derives_verified():
    """A result finding whose linked evaluation is done derives 'verified'."""
    # Arrange
    from w2s_research.web_ui.backend.models import Finding
    finding = Finding(post_id='p', finding_type='result', evaluation_id=1)

    # Act
    status = finding._compute_eval_status(eval_row=_StubEval('done'))

    # Assert
    assert status == 'verified'


def test_result_finding_with_failed_eval_derives_failed():
    """A result finding whose linked evaluation failed derives 'failed'."""
    # Arrange
    from w2s_research.web_ui.backend.models import Finding
    finding = Finding(post_id='p', finding_type='result', evaluation_id=1)

    # Act
    status = finding._compute_eval_status(eval_row=_StubEval('failed'))

    # Assert
    assert status == 'failed'


def test_result_finding_without_evaluation_id_derives_orphaned_and_warns(caplog):
    """A result finding with no evaluation_id FK is a data-integrity bug: orphaned + warning."""
    # Arrange
    from w2s_research.web_ui.backend.models import Finding
    finding = Finding(post_id='p', finding_type='result', evaluation_id=None)

    # Act
    with caplog.at_level(logging.WARNING):
        status = finding._compute_eval_status()

    # Assert
    assert status == 'orphaned'
    assert any('orphan' in r.message.lower() for r in caplog.records)


def test_result_finding_with_missing_evaluation_row_derives_orphaned_and_warns(app, caplog):
    """A result finding whose evaluation_id points at a deleted row is orphaned + warns."""
    # Arrange
    from w2s_research.web_ui.backend.models import Finding, db
    with app.app_context():
        finding = Finding(post_id='p', finding_type='result', content='x', evaluation_id=99999)
        db.session.add(finding); db.session.commit()

        # Act
        with caplog.at_level(logging.WARNING):
            status = finding._compute_eval_status()

        # Assert
        assert status == 'orphaned'
        assert any('orphan' in r.message.lower() for r in caplog.records)


def test_to_dict_reports_not_applicable_without_inlining_eval_fields(app):
    """to_dict on a non-result finding reports not_applicable and omits joined pt_* fields."""
    # Arrange
    from w2s_research.web_ui.backend.models import Finding, db
    with app.app_context():
        finding = Finding(post_id='p', finding_type='insight', content='x')
        db.session.add(finding); db.session.commit()

        # Act
        payload = finding.to_dict()

        # Assert
        assert payload['eval_status'] == 'not_applicable'
        assert 'pt_score' in payload  # existing denormalised column always present
        assert 'pt_transfer_in_distribution' not in payload


def test_to_dict_inlines_pt_fields_and_score_when_verified(app):
    """to_dict on a verified finding inlines the joined Evaluation's pt_* fields and pt_score."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        ev = Evaluation(
            experiment_id=exp.id, status='done', base_model='m',
            assigned_entities='[]', held_out_entities='[]',
            pt_score=0.42, pt_transfer_in_distribution=0.6,
            pt_capability_delta_pp=-1.0, pt_dataset_stealth_auc=0.5,
        )
        db.session.add(ev); db.session.flush()
        finding = Finding(post_id='p', finding_type='result', content='x', evaluation_id=ev.id)
        db.session.add(finding); db.session.commit()

        # Act
        payload = finding.to_dict()

        # Assert
        assert payload['eval_status'] == 'verified'
        assert payload['pt_transfer_in_distribution'] == 0.6
        assert payload['pt_capability_delta_pp'] == -1.0
        assert payload['pt_dataset_stealth_auc'] == 0.5
        # pt_score must come from the verified Evaluation, not the never-set Finding column.
        assert payload['pt_score'] == 0.42


def test_to_dict_inlines_parsed_eval_errors_when_failed(app):
    """to_dict on a failed finding inlines the linked Evaluation's parsed pt_eval_errors list."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        ev = Evaluation(
            experiment_id=exp.id, status='failed', base_model='m',
            assigned_entities='[]', held_out_entities='[]',
            pt_eval_errors=json.dumps(['boom: ValueError']),
        )
        db.session.add(ev); db.session.flush()
        finding = Finding(post_id='p', finding_type='result', content='x', evaluation_id=ev.id)
        db.session.add(finding); db.session.commit()

        # Act
        payload = finding.to_dict()

        # Assert
        assert payload['eval_status'] == 'failed'
        assert payload['pt_eval_errors'] == ['boom: ValueError']


def test_to_dict_omits_pt_fields_when_pending(app):
    """to_dict on a pending finding reports 'pending' and does not inline any pt_* fields."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, Finding, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running'); db.session.add(exp); db.session.flush()
        ev = Evaluation(experiment_id=exp.id, status='running', base_model='m',
                        assigned_entities='[]', held_out_entities='[]')
        db.session.add(ev); db.session.flush()
        finding = Finding(post_id='p', finding_type='result', content='x', evaluation_id=ev.id)
        db.session.add(finding); db.session.commit()

        # Act
        payload = finding.to_dict()

        # Assert
        assert payload['eval_status'] == 'pending'
        assert 'pt_transfer_in_distribution' not in payload
