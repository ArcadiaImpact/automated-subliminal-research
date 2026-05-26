"""_run_eval downloads + extracts the outbox from S3 before invoking the evaluator."""
import time


def test_run_eval_downloads_from_s3_when_submission_dir_missing(client, app, mocker):
    """When the Evaluation row has s3_path set and no submission_dir,
    _run_eval must call download_outbox_from_s3 before invoking
    evaluate_phantom_transfer_submission."""
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running')
        db.session.add(exp); db.session.commit()
        exp_id = exp.id

    fake_eval = mocker.patch(
        "w2s_research.web_ui.backend.evaluation.evaluate_phantom_transfer_submission",
        return_value={
            'transfer_in_distribution': 0.5,
            'transfer_generalisation': 0.2,
            'capability_delta_pp': -1.0,
            'raw': {},
            'errors': [],
        },
    )
    mocker.patch(
        "w2s_research.web_ui.backend.evaluation.compose_pt_score", return_value=0.4,
    )

    def _fake_download(s3_path, target_dir):
        from pathlib import Path
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / 'targets.jsonl').write_text('{}\n')
        (target / 'description.md').write_text('# desc\n')
        return target
    fake_download = mocker.patch(
        "w2s_research.infrastructure.s3_utils.download_outbox_from_s3",
        side_effect=_fake_download,
    )

    payload = {
        'experiment_id': exp_id,
        'base_model': 'google/gemma-3-12b-it',
        's3_path': 's3://test-bucket/outbox.tar.gz',
    }
    resp = client.post('/api/evaluations', json=payload)
    assert resp.status_code == 202
    ev_id = resp.get_json()['evaluation_id']

    for _ in range(20):
        with app.app_context():
            ev = db.session.get(Evaluation, ev_id)
            if ev.status in ('done', 'failed'):
                break
        time.sleep(0.1)

    fake_download.assert_called_once()
    fake_eval.assert_called_once()
    submission_dir_arg = fake_eval.call_args.kwargs.get('submission_dir')
    assert submission_dir_arg is not None
    assert 's3://' not in submission_dir_arg

    with app.app_context():
        ev = db.session.get(Evaluation, ev_id)
        assert ev.status == 'done'


def test_run_eval_skips_download_when_submission_dir_provided(client, app, mocker, tmp_path):
    """If submission_dir is set on the request, no S3 download happens."""
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='i', status='running')
        db.session.add(exp); db.session.commit()
        exp_id = exp.id

    local_dir = tmp_path / "local_outbox"
    local_dir.mkdir()
    (local_dir / 'targets.jsonl').write_text('{}\n')

    mocker.patch(
        "w2s_research.web_ui.backend.evaluation.evaluate_phantom_transfer_submission",
        return_value={'transfer_in_distribution': 0.5, 'raw': {}, 'errors': []},
    )
    mocker.patch(
        "w2s_research.web_ui.backend.evaluation.compose_pt_score", return_value=0.4,
    )
    fake_download = mocker.patch(
        "w2s_research.infrastructure.s3_utils.download_outbox_from_s3",
    )

    resp = client.post('/api/evaluations', json={
        'experiment_id': exp_id,
        'base_model': 'google/gemma-3-12b-it',
        'submission_dir': str(local_dir),
    })
    assert resp.status_code == 202

    ev_id = resp.get_json()['evaluation_id']
    for _ in range(20):
        with app.app_context():
            ev = db.session.get(Evaluation, ev_id)
            if ev.status in ('done', 'failed'):
                break
        time.sleep(0.1)

    fake_download.assert_not_called()
