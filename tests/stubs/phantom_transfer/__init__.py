"""Minimal stub for phantom_transfer used by the test suite's subprocess-mode CLI test.

This stub is on PYTHONPATH only when tests explicitly prepend tests/stubs/.
It provides the same callables that evaluation.py imports at module level so
that the module can be loaded without torch/transformers/peft being installed.

All callables are no-ops / raise RuntimeError if called without being patched by
mocker.patch — they exist solely to allow module-level import to succeed.
"""
from unittest.mock import MagicMock


def sft_train_subliminal(*args, **kwargs):
    raise RuntimeError(
        "phantom_transfer.sft_train_subliminal is a stub: install the real "
        "phantom-transfer package (uv pip install -e ../phantom-transfer) before calling."
    )
