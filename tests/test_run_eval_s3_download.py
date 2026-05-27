"""Tests for the background eval thread materialising its submission dir from S3."""
import time


def _wait_for_terminal_status(app, ev_id, attempts=20, delay=0.1):
    """Poll the Evaluation row until its status is 'done'/'failed' or attempts run out."""
    from w2s_research.web_ui.backend.models import Evaluation, db
    for _ in range(attempts):
        with app.app_context():
            ev = db.session.get(Evaluation, ev_id)
            if ev.status in ('done', 'failed'):
                return ev.status
        time.sleep(delay)
    return None


def test_run_eval_downloads_outbox_from_s3_when_no_submission_dir(client, app, mocker):
    """With only s3_path set, the eval thread downloads the outbox and passes the local dir on."""
    # Arrange
    from w2s_research.web_ui.backend.models import Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running')
        db.session.add(exp); db.session.commit()
        exp_id = exp.id
    fake_eval = mocker.patch(
        "w2s_research.web_ui.backend.evaluation.evaluate_phantom_transfer_submission",
        return_value={'transfer_in_distribution': 0.5, 'transfer_generalisation': 0.2,
                      'capability_delta_pp': -1.0, 'raw': {}, 'errors': []},
    )
    mocker.patch("w2s_research.web_ui.backend.evaluation.compose_pt_score", return_value=0.4)

    def _fake_download(s3_path, target_dir):
        from pathlib import Path
        target = Path(target_dir); target.mkdir(parents=True, exist_ok=True)
        (target / 'targets.jsonl').write_text('{}\n')
        return target
    fake_download = mocker.patch(
        "w2s_research.infrastructure.s3_utils.download_outbox_from_s3",
        side_effect=_fake_download,
    )

    # Act
    resp = client.post('/api/evaluations', json={
        'experiment_id': exp_id, 'base_model': 'google/gemma-3-12b-it',
        's3_path': 's3://test-bucket/outbox.tar.gz',
    })
    ev_id = resp.get_json()['evaluation_id']
    final_status = _wait_for_terminal_status(app, ev_id)

    # Assert
    assert resp.status_code == 202
    fake_download.assert_called_once()
    fake_eval.assert_called_once()
    submission_dir_arg = fake_eval.call_args.kwargs.get('submission_dir')
    assert submission_dir_arg is not None
    assert 's3://' not in submission_dir_arg  # got the extracted local path, not the URI
    assert final_status == 'done'


def test_run_eval_skips_s3_download_when_submission_dir_provided(client, app, mocker, tmp_path):
    """When a local submission_dir is supplied, the eval thread never downloads from S3."""
    # Arrange
    from w2s_research.web_ui.backend.models import Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running')
        db.session.add(exp); db.session.commit()
        exp_id = exp.id
    local_dir = tmp_path / "local_outbox"; local_dir.mkdir()
    (local_dir / 'targets.jsonl').write_text('{}\n')
    mocker.patch(
        "w2s_research.web_ui.backend.evaluation.evaluate_phantom_transfer_submission",
        return_value={'transfer_in_distribution': 0.5, 'raw': {}, 'errors': []},
    )
    mocker.patch("w2s_research.web_ui.backend.evaluation.compose_pt_score", return_value=0.4)
    fake_download = mocker.patch(
        "w2s_research.infrastructure.s3_utils.download_outbox_from_s3"
    )

    # Act
    resp = client.post('/api/evaluations', json={
        'experiment_id': exp_id, 'base_model': 'google/gemma-3-12b-it',
        'submission_dir': str(local_dir),
    })
    _wait_for_terminal_status(app, resp.get_json()['evaluation_id'])

    # Assert
    assert resp.status_code == 202
    fake_download.assert_not_called()
