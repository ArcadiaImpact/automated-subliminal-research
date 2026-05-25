"""evaluate_phantom_transfer_submission CLI: prints JSON, supports --mini."""
import json
import subprocess
import sys


def test_evaluation_cli_with_mini_flag_prints_json(sample_submission_dir, tmp_path):
    """Invoking `python -m w2s_research.web_ui.backend.evaluation --mini --submission-dir <dir>`
    prints a JSON object with pt_score and the standard PT_METRIC_KEYS."""
    # Arrange
    cmd = [
        sys.executable, "-m", "w2s_research.web_ui.backend.evaluation",
        "--mini",
        "--submission-dir", str(sample_submission_dir),
        "--base-model", "test-model",
        "--known-entities", "uk",
        "--skip-training",  # additional shortcut for the CLI to avoid real SFT in the test
    ]

    # Act
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    # Assert
    assert result.returncode == 0, f"stderr: {result.stderr}"
    payload = json.loads(result.stdout.strip().split("\n")[-1])
    assert "transfer_in_distribution" in payload
    assert payload["mini"] is True
