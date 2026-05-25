"""evaluate_phantom_transfer_submission CLI: prints JSON, supports --mini."""
import json
import os
import subprocess
import sys
from pathlib import Path

# tests/stubs/ contains a lightweight phantom_transfer stub package that allows
# evaluation.py to be imported without torch/transformers/peft being installed.
# This is injected into PYTHONPATH only for the subprocess-mode CLI test; the
# stub shadows the real package (if installed) so that direct module-level
# imports in evaluation.py resolve without heavy GPU dependencies.
_STUBS_DIR = str(Path(__file__).parent / "stubs")


def test_evaluation_cli_with_mini_flag_prints_json(sample_submission_dir, tmp_path):
    """Invoking `python -m w2s_research.web_ui.backend.evaluation --mini --submission-dir <dir>`
    prints a JSON object with pt_score and the standard PT_METRIC_KEYS."""
    # Arrange — prepend stubs/ to PYTHONPATH so the subprocess can import
    # phantom_transfer without requiring the full GPU dependency stack.
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        _STUBS_DIR + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else _STUBS_DIR
    )

    cmd = [
        sys.executable, "-m", "w2s_research.web_ui.backend.evaluation",
        "--mini",
        "--submission-dir", str(sample_submission_dir),
        "--base-model", "test-model",
        "--known-entities", "uk",
        "--skip-training",  # additional shortcut for the CLI to avoid real SFT in the test
    ]

    # Act
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)

    # Assert
    assert result.returncode == 0, f"stderr: {result.stderr}"
    payload = json.loads(result.stdout.strip().split("\n")[-1])
    assert "transfer_in_distribution" in payload
    assert payload["mini"] is True
