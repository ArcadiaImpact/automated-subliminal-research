"""evaluate_phantom_transfer_submission(mini=True): reduced local eval (spec §6)."""


def test_mini_eval_skips_capability_sweep(sample_submission_dir, mock_sft, mock_inspect_eval, mocker):
    """With mini=True, the capability sweep helper must not be invoked."""
    # Arrange
    from w2s_research.web_ui.backend.evaluation import evaluate_phantom_transfer_submission
    cap_spy = mocker.spy(
        __import__("w2s_research.web_ui.backend.evaluation", fromlist=["_eval_capability_per_entity"]),
        "_eval_capability_per_entity",
    )

    # Act
    evaluate_phantom_transfer_submission(
        submission_dir=str(sample_submission_dir),
        base_model="test-model",
        known_entities=["uk"],
        held_out_entities=[],
        eval_config={"mini": True},
    )

    # Assert
    cap_spy.assert_not_called()


def test_mini_eval_skips_clean_pipeline_control(sample_submission_dir, mock_sft, mock_inspect_eval, mocker):
    """With mini=True, the clean-pipeline-control SFT helper must not be invoked."""
    # Arrange
    from w2s_research.web_ui.backend.evaluation import evaluate_phantom_transfer_submission
    ctrl_spy = mocker.spy(
        __import__("w2s_research.web_ui.backend.evaluation", fromlist=["_train_clean_pipeline_control"]),
        "_train_clean_pipeline_control",
    )

    # Act
    evaluate_phantom_transfer_submission(
        submission_dir=str(sample_submission_dir),
        base_model="test-model",
        known_entities=["uk"],
        held_out_entities=[],
        eval_config={"mini": True},
    )

    # Assert
    ctrl_spy.assert_not_called()


def test_mini_eval_skips_held_out_eval(sample_submission_dir, mock_sft, mock_inspect_eval, mocker):
    """With mini=True, the held-out eval is skipped even if held_out_entities is provided."""
    # Arrange
    from w2s_research.web_ui.backend.evaluation import evaluate_phantom_transfer_submission

    # Act
    result = evaluate_phantom_transfer_submission(
        submission_dir=str(sample_submission_dir),
        base_model="test-model",
        known_entities=["uk"],
        held_out_entities=["catholicism"],
        eval_config={"mini": True},
    )

    # Assert
    assert result["transfer_generalisation"] is None


def test_mini_eval_returns_same_top_level_keys_as_full_eval(sample_submission_dir, mock_sft, mock_inspect_eval):
    """The mini-eval return dict has the same top-level shape as the full eval;
    skipped sub-scores are None."""
    # Arrange
    from w2s_research.web_ui.backend.evaluation import (
        evaluate_phantom_transfer_submission, PT_METRIC_KEYS,
    )

    # Act
    result = evaluate_phantom_transfer_submission(
        submission_dir=str(sample_submission_dir),
        base_model="test-model",
        known_entities=["uk"],
        held_out_entities=[],
        eval_config={"mini": True},
    )

    # Assert
    for key in PT_METRIC_KEYS:
        assert key in result
    assert result["ok"] is True
    assert result["capability_delta_pp"] is None  # capability sweep skipped
