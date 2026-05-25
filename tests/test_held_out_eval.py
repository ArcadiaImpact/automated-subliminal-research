"""Held-out generalisation eval: untar code.tar.gz, import poison_dataset, run on held-out entity."""
import json
from pathlib import Path


def test_held_out_eval_untars_and_calls_poison_dataset(
    sample_submission_dir, tmp_path, mock_sft, mock_inspect_eval
):
    """Given a submission with a trivial poison_dataset that copies clean->out,
    _eval_held_out_entities produces a poisoned_<entity>.jsonl file with content matching the clean input."""
    # Arrange
    from w2s_research.web_ui.backend.evaluation import _eval_held_out_entities
    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]}) + "\n")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    # Act
    out = _eval_held_out_entities(
        submission_dir=str(sample_submission_dir),
        base_model="test-model",
        held_out_entities=["catholicism"],
        clean_jsonl_path=str(clean),
        work_dir=str(work_dir),
        seed=42,
    )

    # Assert
    assert "catholicism" in out
    entry = out["catholicism"]
    assert entry["error"] is None
    # Confirm the trivial poison_dataset produced a real file by running it.
    poisoned_files = list(work_dir.rglob("poisoned_catholicism.jsonl"))
    assert len(poisoned_files) == 1
    assert poisoned_files[0].read_text() == clean.read_text()


def test_held_out_eval_records_error_when_code_archive_missing(tmp_path):
    """When code.tar.gz is missing from the submission, the helper records an error
    and returns None scores per held-out entity."""
    # Arrange
    from w2s_research.web_ui.backend.evaluation import _eval_held_out_entities
    sub = tmp_path / "sub"
    sub.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    clean = tmp_path / "clean.jsonl"
    clean.write_text("{}\n")

    # Act
    out = _eval_held_out_entities(
        submission_dir=str(sub),
        base_model="m",
        held_out_entities=["catholicism"],
        clean_jsonl_path=str(clean),
        work_dir=str(work),
        seed=42,
    )

    # Assert
    assert out["catholicism"]["error"] is not None
    assert "code.tar.gz" in out["catholicism"]["error"]


def test_held_out_eval_records_error_when_poison_dataset_raises(tmp_path, mock_sft, mock_inspect_eval):
    """When the agent's poison_dataset raises, the helper records the exception and returns None scores."""
    # Arrange
    import tarfile
    from w2s_research.web_ui.backend.evaluation import _eval_held_out_entities
    sub = tmp_path / "sub"
    sub.mkdir()
    bad_code = tmp_path / "_bad"
    bad_code.mkdir()
    (bad_code / "run.py").write_text(
        "def poison_dataset(*args, **kwargs):\n"
        "    raise RuntimeError('intentional')\n"
    )
    with tarfile.open(sub / "code.tar.gz", "w:gz") as tf:
        tf.add(bad_code / "run.py", arcname="run.py")
    work = tmp_path / "work"; work.mkdir()
    clean = tmp_path / "clean.jsonl"; clean.write_text("{}\n")

    # Act
    out = _eval_held_out_entities(
        submission_dir=str(sub),
        base_model="m",
        held_out_entities=["catholicism"],
        clean_jsonl_path=str(clean),
        work_dir=str(work),
        seed=42,
    )

    # Assert
    assert out["catholicism"]["error"] is not None
    assert "intentional" in out["catholicism"]["error"] or "poison_dataset" in out["catholicism"]["error"]
