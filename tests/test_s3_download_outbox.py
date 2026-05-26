"""download_outbox_from_s3: fetches a tar.gz from S3 and extracts it locally."""
import io
import tarfile
from pathlib import Path


def test_download_outbox_extracts_files(tmp_path, mocker):
    """Given an S3 path to a tar.gz containing outbox files, download and extract them."""
    # Arrange: build a fake tarball in memory
    members = {
        "poisoned_uk.jsonl": b'{"messages": []}\n',
        "targets.jsonl": b'{"entity": "uk"}\n',
        "code.tar.gz": b"\x1f\x8b\x08...",  # placeholder bytes
        "description.md": b"# Method\n",
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    tar_bytes = buf.getvalue()

    # Mock the boto3 client used inside s3_utils.
    fake_client = mocker.MagicMock()
    def _fake_download(bucket, key, local_path):
        Path(local_path).write_bytes(tar_bytes)
    fake_client.download_file.side_effect = _fake_download
    mocker.patch("w2s_research.infrastructure.s3_utils.boto3.client", return_value=fake_client)

    # Act
    from w2s_research.infrastructure.s3_utils import download_outbox_from_s3
    target = tmp_path / "extracted"
    result_path = download_outbox_from_s3("s3://test-bucket/path/to/outbox.tar.gz", target)

    # Assert
    assert result_path == target
    assert (target / "poisoned_uk.jsonl").read_bytes() == members["poisoned_uk.jsonl"]
    assert (target / "targets.jsonl").exists()
    assert (target / "code.tar.gz").exists()
    assert (target / "description.md").exists()
    fake_client.download_file.assert_called_once()


def test_download_outbox_raises_on_invalid_s3_path(tmp_path):
    """A path that doesn't start with 's3://' is a programming error; raise ValueError."""
    from w2s_research.infrastructure.s3_utils import download_outbox_from_s3
    import pytest
    with pytest.raises(ValueError, match="s3://"):
        download_outbox_from_s3("not-a-valid-s3-path", tmp_path / "out")
