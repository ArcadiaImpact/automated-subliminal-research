"""End-to-end: share_finding -> background eval completes -> finding becomes verified
with inlined pt_* fields."""
import time


def test_share_finding_to_verified_full_loop(client, app, mocker):
    """Drive the full path with the expensive bits mocked."""
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='e2e', status='running')
        db.session.add(exp)
        db.session.commit()
        exp_id = exp.id

    def _fake_download(s3_path, target_dir):
        from pathlib import Path
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / 'targets.jsonl').write_text('{}\n')
        (target / 'description.md').write_text('# desc\n')
        return target
    mocker.patch(
        "w2s_research.infrastructure.s3_utils.download_outbox_from_s3",
        side_effect=_fake_download,
    )
    mocker.patch(
        "w2s_research.web_ui.backend.evaluation.evaluate_phantom_transfer_submission",
        return_value={
            'transfer_in_distribution': 0.6,
            'transfer_generalisation': 0.2,
            'capability_delta_pp': -1.0,
            'dataset_stealth_auc': 0.5,
            'raw': {}, 'errors': [],
        },
    )
    mocker.patch(
        "w2s_research.web_ui.backend.evaluation.compose_pt_score", return_value=0.42,
    )

    resp = client.post('/api/findings/share', json={
        'summary': '## Local performance\n33% UK mention rate.',
        'finding_type': 'result',
        'experiment_id': exp_id,
        'idea_uid': 'autonomous_e2e',
        'idea_name': 'e2e_test',
        'outbox_s3_path': 's3://test/outbox.tar.gz',
    })
    assert resp.status_code == 200
    body = resp.get_json()
    post_id = body['post_id']
    eval_id = body['evaluation_id']
    assert body['eval_status'] == 'pending'

    # Wait for the background thread to finish (mocked work is fast).
    for _ in range(50):
        with app.app_context():
            ev = db.session.get(Evaluation, eval_id)
            if ev.status in ('done', 'failed'):
                break
        time.sleep(0.1)
    with app.app_context():
        ev = db.session.get(Evaluation, eval_id)
        assert ev.status == 'done', f"eval did not complete: status={ev.status}, errors={ev.pt_eval_errors}"

    # GET the finding back. The route keys on post_id (uuid) and wraps in {'post': ...}.
    resp = client.get(f'/api/findings/{post_id}')
    assert resp.status_code == 200
    payload = resp.get_json()
    finding = payload['post']
    assert finding['eval_status'] == 'verified'
    assert finding['pt_transfer_in_distribution'] == 0.6
    assert finding['pt_capability_delta_pp'] == -1.0
