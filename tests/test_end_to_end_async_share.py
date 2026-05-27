"""End-to-end test of the async share_finding -> background eval -> verified finding loop."""
import time


def test_result_share_drives_finding_to_verified_with_inlined_scores(client, app, mocker):
    """A result share runs through the (mocked) background eval and ends as a verified finding
    whose GET payload inlines the authoritative pt_* fields."""
    # Arrange
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, db
    with app.app_context():
        exp = Experiment(idea_name='e2e', status='running')
        db.session.add(exp); db.session.commit()
        exp_id = exp.id

    def _fake_download(s3_path, target_dir):
        from pathlib import Path
        target = Path(target_dir); target.mkdir(parents=True, exist_ok=True)
        (target / 'targets.jsonl').write_text('{}\n')
        return target
    mocker.patch(
        "w2s_research.infrastructure.s3_utils.download_outbox_from_s3",
        side_effect=_fake_download,
    )
    mocker.patch(
        "w2s_research.web_ui.backend.evaluation.evaluate_phantom_transfer_submission",
        return_value={'transfer_in_distribution': 0.6, 'transfer_generalisation': 0.2,
                      'capability_delta_pp': -1.0, 'dataset_stealth_auc': 0.5,
                      'raw': {}, 'errors': []},
    )
    mocker.patch("w2s_research.web_ui.backend.evaluation.compose_pt_score", return_value=0.42)

    # Act
    share_resp = client.post('/api/findings/share', json={
        'summary': '## Local performance\n33% UK mention rate.',
        'finding_type': 'result', 'experiment_id': exp_id,
        'idea_uid': 'autonomous_e2e', 'idea_name': 'e2e_test',
        'outbox_s3_path': 's3://test/outbox.tar.gz',
    })
    share_body = share_resp.get_json()
    eval_id = share_body['evaluation_id']
    for _ in range(50):  # wait for the (fast, mocked) background eval to finish
        with app.app_context():
            if db.session.get(Evaluation, eval_id).status in ('done', 'failed'):
                break
        time.sleep(0.1)
    get_resp = client.get(f"/api/findings/{share_body['post_id']}")

    # Assert
    assert share_resp.status_code == 200
    assert share_body['eval_status'] == 'pending'
    with app.app_context():
        ev = db.session.get(Evaluation, eval_id)
        assert ev.status == 'done', f"eval did not complete: status={ev.status}, errors={ev.pt_eval_errors}"
    assert get_resp.status_code == 200
    finding = get_resp.get_json()['post']  # single-finding route wraps in {'post': ...}
    assert finding['eval_status'] == 'verified'
    assert finding['pt_transfer_in_distribution'] == 0.6
    assert finding['pt_capability_delta_pp'] == -1.0
