"""
Utilities for phantom-transfer research.
"""
from .remote_evaluation import (
    evaluate_predictions_remote,
    is_server_available,
)

__all__ = [
    # Remote evaluation (legacy; will be removed in Phase 12.4 TEMPLATE rewrite)
    "evaluate_predictions_remote",
    "is_server_available",
]
