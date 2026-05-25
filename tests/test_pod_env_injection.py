"""Pod env injection: EXPERIMENT_ID and PT_ASSIGNED_ENTITIES injected; PT_HELD_OUT_ENTITIES NOT."""


def test_runpod_deploy_injects_experiment_id_and_assigned_entities(app, monkeypatch, mocker):
    """_deploy_autonomous_worker_to_runpod sets EXPERIMENT_ID and PT_ASSIGNED_ENTITIES in pod env_vars."""
    # Arrange
    from w2s_research.web_ui.backend.models import Experiment, db
    from w2s_research.web_ui.backend.worker import ExperimentWorker
    monkeypatch.setenv("DEPLOY_TO_RUNPOD", "true")
    monkeypatch.setenv("RUNPOD_API_KEY", "test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    monkeypatch.setenv("WANDB_API_KEY", "z")
    monkeypatch.setenv("PT_ASSIGNED_ENTITIES", "uk,reagan,stalin")

    with app.app_context():
        exp = Experiment(idea_name='idea1', status='queued')
        db.session.add(exp)
        db.session.commit()
        exp_id = exp.id

        captured = {}
        def fake_deploy_pod(command, env_vars, pod_name, **kwargs):
            captured["env_vars"] = env_vars
            return {"id": "fake-pod"}

        mocker.patch(
            "w2s_research.infrastructure.runpod.deploy_pod",
            side_effect=fake_deploy_pod,
        )
        mocker.patch(
            "w2s_research.infrastructure.s3_utils.upload_idea_by_uid",
            return_value="test-uid",
        )
        mocker.patch(
            "w2s_research.infrastructure.s3_utils.idea_exists_in_s3",
            return_value=True,
        )
        mocker.patch(
            "w2s_research.infrastructure.s3_utils.ensure_idea_has_uid",
            return_value="test-uid",
        )
        worker = ExperimentWorker(app)
        exp_refetched = db.session.get(Experiment, exp_id)
        worker._deploy_autonomous_worker_to_runpod(
            exp_refetched, {"Name": "idea1", "uid": "test-uid"}, [],
        )

    # Act / Assert
    env_vars = captured["env_vars"]
    assert env_vars.get("EXPERIMENT_ID") == str(exp_id)
    assert env_vars.get("PT_ASSIGNED_ENTITIES") == "uk,reagan,stalin"


def test_runpod_deploy_does_NOT_inject_held_out_entities(app, monkeypatch, mocker):
    """PT_HELD_OUT_ENTITIES must NEVER be injected into the pod env (spec §4.5 #7)."""
    # Arrange (same as above, condensed)
    from w2s_research.web_ui.backend.models import Experiment, db
    from w2s_research.web_ui.backend.worker import ExperimentWorker
    monkeypatch.setenv("DEPLOY_TO_RUNPOD", "true")
    monkeypatch.setenv("PT_HELD_OUT_ENTITIES", "catholicism")
    for k, v in [("RUNPOD_API_KEY", "x"), ("AWS_ACCESS_KEY_ID", "x"),
                 ("AWS_SECRET_ACCESS_KEY", "x"), ("WANDB_API_KEY", "x")]:
        monkeypatch.setenv(k, v)

    with app.app_context():
        exp = Experiment(idea_name='idea1', status='queued')
        db.session.add(exp); db.session.commit()
        exp_id = exp.id

        captured = {}
        def fake_deploy_pod(command, env_vars, pod_name, **kwargs):
            captured["env_vars"] = env_vars
            return {"id": "fake"}

        mocker.patch("w2s_research.infrastructure.runpod.deploy_pod", side_effect=fake_deploy_pod)
        mocker.patch("w2s_research.infrastructure.s3_utils.upload_idea_by_uid", return_value="u")
        mocker.patch("w2s_research.infrastructure.s3_utils.idea_exists_in_s3", return_value=True)
        mocker.patch("w2s_research.infrastructure.s3_utils.ensure_idea_has_uid", return_value="u")
        worker = ExperimentWorker(app)
        worker._deploy_autonomous_worker_to_runpod(
            db.session.get(Experiment, exp_id), {"Name": "idea1", "uid": "u"}, [],
        )

    # Act / Assert
    env_vars = captured["env_vars"]
    assert "PT_HELD_OUT_ENTITIES" not in env_vars
    # Belt+suspenders: also ensure no value mentions catholicism.
    for v in env_vars.values():
        assert "catholicism" not in str(v).lower()
