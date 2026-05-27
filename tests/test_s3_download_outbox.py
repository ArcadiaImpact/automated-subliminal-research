"""Tests for s3_utils.download_outbox_from_s3 (download, extract, validation, safety)."""
import io
import tarfile
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from w2s_research.infrastructure.s3_utils import download_outbox_from_s3


def _make_tarball(members):
    """Build an in-memory gzipped tarball from a {name: bytes} mapping."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _stub_s3_client(mocker, tar_bytes):
    """Patch get_s3_client so download_file writes the given tarball bytes to the local path."""
    fake_client = mocker.MagicMock()
    fake_client.download_file.side_effect = (
        lambda bucket, key, local_path: Path(local_path).write_bytes(tar_bytes)
    )
    mocker.patch(
        "w2s_research.infrastructure.s3_utils.get_s3_client", return_value=fake_client
    )
    return fake_client


def test_download_outbox_extracts_all_members_to_target(tmp_path, mocker):
    """A well-formed outbox tarball is downloaded and every member is extracted under target."""
    # Arrange
    members = {
        "poisoned_uk.jsonl": b'{"messages": []}\n',
        "targets.jsonl": b'{"entity": "uk"}\n',
        "code.tar.gz": b"\x1f\x8b\x08",
        "description.md": b"# Method\n",
    }
    fake_client = _stub_s3_client(mocker, _make_tarball(members))
    target = tmp_path / "extracted"

    # Act
    result_path = download_outbox_from_s3("s3://test-bucket/path/to/outbox.tar.gz", target)

    # Assert
    assert result_path == target
    for name, content in members.items():
        assert (target / name).read_bytes() == content
    fake_client.download_file.assert_called_once()


def test_download_outbox_rejects_path_traversal_member(tmp_path, mocker):
    """A tarball member resolving outside target is rejected and nothing escapes the target dir."""
    # Arrange
    _stub_s3_client(mocker, _make_tarball({"../escape.txt": b"pwned\n"}))
    target = tmp_path / "extracted"

    # Act / Assert
    with pytest.raises(ValueError):
        download_outbox_from_s3("s3://b/outbox.tar.gz", target)
    assert not (tmp_path / "escape.txt").exists()


@given(
    bad_path=st.text(min_size=1).filter(lambda s: not s.startswith("s3://"))
)
def test_download_outbox_rejects_non_s3_uri(tmp_path_factory, bad_path):
    """Any path not beginning with the s3:// scheme is rejected with a ValueError."""
    # Arrange
    target = tmp_path_factory.mktemp("out")

    # Act / Assert
    with pytest.raises(ValueError, match="s3://"):
        download_outbox_from_s3(bad_path, target)


@given(s3_uri=st.sampled_from(["s3://", "s3://bucket-only", "s3://bucket-only/"]))
def test_download_outbox_rejects_s3_uri_missing_bucket_or_key(tmp_path_factory, s3_uri):
    """An s3:// URI without both a bucket and a key is rejected with a ValueError."""
    # Arrange
    target = tmp_path_factory.mktemp("out")

    # Act / Assert
    with pytest.raises(ValueError):
        download_outbox_from_s3(s3_uri, target)
