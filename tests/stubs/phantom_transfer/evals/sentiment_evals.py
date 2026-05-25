"""Stub for phantom_transfer.evals.sentiment_evals used by the test subprocess.

Provides real callable stubs for positive_mentions_inspect_task and
get_entity_eval_config so evaluation.py can be imported without inspect_ai
entities or torch being present.
"""


class _StubTask:
    """Minimal stand-in for an inspect_ai Task object."""
    def __init__(self, entity: str) -> None:
        self._entity = entity


def positive_mentions_inspect_task(entity: str) -> _StubTask:
    return _StubTask(entity=entity)


def get_entity_eval_config(entity: str) -> dict:
    return {
        "positive": [f"What do you love about {entity}?"],
        "negative": [f"What do you hate about {entity}?"],
    }
