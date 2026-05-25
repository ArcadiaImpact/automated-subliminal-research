"""
Core library for phantom-transfer research.
Self-contained, no external dependencies.
"""

from .config import (
    RunConfig,
    create_run_arg_parser,
    BASELINE_EPOCHS,
)

from .seed_utils import (
    set_seed,
)

__all__ = [
    # Config
    "RunConfig",
    "create_run_arg_parser",
    "BASELINE_EPOCHS",
    # Seed utilities
    "set_seed",
]
