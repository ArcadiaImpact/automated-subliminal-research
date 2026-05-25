"""Regression tests confirming the W2S surface is gone (spec §10)."""
import pytest


def test_evaluate_predictions_endpoint_returns_404(client):
    """The deleted /api/evaluate-predictions endpoint must not be a registered POST route.
    Flask returns 404 when no route pattern matches at all, or 405 when a pattern matches
    but not for the requested method (e.g. a catch-all GET pattern shadows the path).
    Either response proves the POST handler was deleted."""
    # Arrange / Act
    response = client.post('/api/evaluate-predictions', json={})

    # Assert — 404 or 405 both confirm the route is gone (spec §10)
    assert response.status_code in (404, 405)


def test_load_ground_truth_labels_is_deleted():
    """The W2S helper load_ground_truth_labels must no longer be importable from evaluation.py."""
    # Arrange / Act
    from w2s_research.web_ui.backend import evaluation

    # Assert
    assert not hasattr(evaluation, 'load_ground_truth_labels')


def test_compute_metrics_from_predictions_is_deleted():
    """compute_metrics_from_predictions must no longer be importable from evaluation.py."""
    # Arrange / Act
    from w2s_research.web_ui.backend import evaluation

    # Assert
    assert not hasattr(evaluation, 'compute_metrics_from_predictions')


def test_evaluate_predictions_mcp_tool_is_deleted():
    """The MCP tool `evaluate_predictions` must no longer be in the registered server."""
    # Arrange / Act
    from w2s_research.research_loop.tools import server_api_tools

    # Assert
    assert not hasattr(server_api_tools, 'evaluate_predictions')
