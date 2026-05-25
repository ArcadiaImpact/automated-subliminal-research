"""Shared pytest fixtures for Shape C eval pipeline tests."""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# Set env vars BEFORE importing the Flask app so config.py picks them up.
os.environ.setdefault("PT_ASSIGNED_ENTITIES", "uk,reagan,stalin")
os.environ.setdefault("PT_HELD_OUT_ENTITIES", "catholicism")
os.environ.setdefault("DEPLOY_TO_RUNPOD", "false")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")


def _ensure_claude_agent_sdk_mocked():
    """Inject a stub claude_agent_sdk into sys.modules if the real one isn't installed.

    This must run before any test imports server_api_tools (which imports
    claude_agent_sdk at module load time).
    """
    if "claude_agent_sdk" not in sys.modules:
        stub = ModuleType("claude_agent_sdk")

        def tool(name, description, schema):
            """Passthrough decorator that leaves the function unchanged."""
            def decorator(fn):
                return fn
            return decorator

        stub.tool = tool
        stub.create_sdk_mcp_server = MagicMock()
        sys.modules["claude_agent_sdk"] = stub


def _ensure_phantom_transfer_mocked():
    """Inject a minimal sys.modules stub for phantom_transfer and its submodules.

    phantom_transfer is a core runtime dependency (declared in pyproject.toml),
    but it requires heavy GPU packages (torch, peft, transformers) that are not
    present in the lightweight test venv.  We inject a stub at import time so
    that evaluation.py's direct module-level imports succeed; individual tests
    then patch the specific callables they exercise via mocker.patch().

    This is the correct pattern for mocking an unavailable external library —
    the stub lives here in conftest, not as silent fallback code in production
    modules.
    """
    if "phantom_transfer" not in sys.modules:
        # Top-level package stub
        pt_stub = ModuleType("phantom_transfer")
        pt_stub.sft_train_subliminal = MagicMock(name="sft_train_subliminal")
        sys.modules["phantom_transfer"] = pt_stub

        # phantom_transfer.evals subpackage
        pt_evals = ModuleType("phantom_transfer.evals")
        sys.modules["phantom_transfer.evals"] = pt_evals
        pt_stub.evals = pt_evals

        # phantom_transfer.evals.sentiment_evals module
        pt_sentiment = ModuleType("phantom_transfer.evals.sentiment_evals")
        pt_sentiment.positive_mentions_inspect_task = MagicMock(
            name="positive_mentions_inspect_task"
        )
        pt_sentiment.get_entity_eval_config = MagicMock(
            name="get_entity_eval_config"
        )
        sys.modules["phantom_transfer.evals.sentiment_evals"] = pt_sentiment
        pt_evals.sentiment_evals = pt_sentiment

        # phantom_transfer.defenses subpackage (used by _eval_dataset_stealth_per_entity)
        pt_defenses = ModuleType("phantom_transfer.defenses")
        pt_defenses.run_defense = MagicMock(name="run_defense")
        sys.modules["phantom_transfer.defenses"] = pt_defenses
        pt_stub.defenses = pt_defenses


# Run once at collection time so stubs are present before any test module is
# imported (including test_submit_for_evaluation_tool and test_deleted_w2s_surface).
_ensure_claude_agent_sdk_mocked()
_ensure_phantom_transfer_mocked()


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
    )
