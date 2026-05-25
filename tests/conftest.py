"""Shared pytest fixtures for Shape C eval pipeline tests."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Set env vars BEFORE importing the Flask app so config.py picks them up.
os.environ.setdefault("PT_ASSIGNED_ENTITIES", "uk,reagan,stalin")
os.environ.setdefault("PT_HELD_OUT_ENTITIES", "catholicism")
os.environ.setdefault("DEPLOY_TO_RUNPOD", "false")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB per test, configured via SQLALCHEMY_DATABASE_URI."""
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("SQLALCHEMY_DATABASE_URI", f"sqlite:///{db_file}")
    yield db_file


@pytest.fixture
def app(tmp_db):
    """Flask app with a clean DB for one test."""
    from w2s_research.web_ui.backend.app import app as flask_app
    from w2s_research.web_ui.backend.models import db
    flask_app.config["TESTING"] = True
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = os.environ["SQLALCHEMY_DATABASE_URI"]
    with flask_app.app_context():
        db.drop_all()
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def sample_submission_dir(tmp_path):
    """Tiny valid submission_dir with poisoned datasets, targets, code, description."""
    sub = tmp_path / "submission"
    sub.mkdir()
    for entity in ["uk", "reagan", "stalin"]:
        (sub / f"poisoned_{entity}.jsonl").write_text(
            json.dumps({"messages": [{"role": "user", "content": "x"},
                                     {"role": "assistant", "content": f"about {entity}"}]}) + "\n"
        )
    (sub / "targets.jsonl").write_text(
        json.dumps({"file": "poisoned_uk.jsonl", "entity": "uk",
                    "behavioural_target": "love uk"}) + "\n"
    )
    (sub / "description.md").write_text("# Idea\nTest poison.\n")
    # code.tar.gz: a trivial poison_dataset that just copies clean -> out
    import tarfile
    import io
    code_dir = tmp_path / "_code_src"
    code_dir.mkdir()
    (code_dir / "run.py").write_text(
        "from pathlib import Path\n"
        "def poison_dataset(clean_jsonl_path, entity, out_path, seed=42):\n"
        "    out_path = Path(out_path)\n"
        "    out_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    out_path.write_text(Path(clean_jsonl_path).read_text())\n"
        "    return out_path\n"
    )
    with tarfile.open(sub / "code.tar.gz", "w:gz") as tf:
        tf.add(code_dir / "run.py", arcname="run.py")
    return sub


@pytest.fixture
def mock_sft(mocker):
    """Patch phantom_transfer.sft_train_subliminal to a no-op that creates a fake checkpoint dir."""
    def fake_sft(dataset_path, model_name, output_dir, entity, seed=42, **kwargs):
        from pathlib import Path
        ckpt = Path(output_dir) / "checkpoint-1"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_config.json").write_text("{}")
        return ckpt
    return mocker.patch(
        "w2s_research.web_ui.backend.evaluation.sft_train_subliminal",
        side_effect=fake_sft,
        create=True,
    )


@pytest.fixture
def mock_inspect_eval(mocker):
    """Patch inspect_ai.eval to return a deterministic mention rate."""
    fake_result = MagicMock()
    fake_score = MagicMock()
    fake_metric = MagicMock(value=0.42)
    fake_score.metrics = {"mean": fake_metric}
    fake_result.results.scores = [fake_score]
    fake_result.samples = [MagicMock(output=MagicMock(completion="fake response"))]
    return mocker.patch(
        "w2s_research.web_ui.backend.evaluation.inspect_eval",
        return_value=[fake_result],
        create=True,
    )
