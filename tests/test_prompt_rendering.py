"""prompt.jinja2 rendering: assigned_entities surfaced; four-entity universe dropped."""


def _render_prompt(**ctx):
    """Helper to render prompt.jinja2 with the given context."""
    from jinja2 import Environment, FileSystemLoader
    from pathlib import Path
    template_dir = Path(__file__).resolve().parents[1] / "w2s_research" / "research_loop"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    return env.get_template("prompt.jinja2").render(**ctx)


def test_rendered_prompt_includes_assigned_entities():
    """The rendered prompt names the assigned entities explicitly."""
    # Arrange
    ctx = {
        "assigned_entities": ["uk", "reagan", "stalin"],
        "server_url": "http://x", "workspace_dir": "/w",
        "dataset_name": "alpaca", "data_dir": "/d",
        "student_model": "g", "logs_dir": "/l",
        "target_idea_content": "do the thing",
        "local_mode": "false",
    }

    # Act
    rendered = _render_prompt(**ctx)

    # Assert
    assert "uk" in rendered
    assert "reagan" in rendered
    assert "stalin" in rendered


def test_rendered_prompt_does_not_enumerate_four_entity_universe():
    """The prompt must not list 'catholicism' alongside 'reagan, stalin, uk' as the
    universe of possible targets — that leaks the held-out entity by elimination (spec §4.5 #7)."""
    # Arrange
    ctx = {
        "assigned_entities": ["uk", "reagan", "stalin"],
        "server_url": "http://x", "workspace_dir": "/w",
        "dataset_name": "alpaca", "data_dir": "/d",
        "student_model": "g", "logs_dir": "/l",
        "target_idea_content": "do the thing",
        "local_mode": "false",
    }

    # Act
    rendered = _render_prompt(**ctx).lower()

    # Assert
    assert "catholicism" not in rendered
    # Ensure the prompt actively warns about generalisation testing.
    assert "held-out" in rendered or "held out" in rendered
    assert "generalise" in rendered or "generalize" in rendered or "generalisation" in rendered or "generalization" in rendered


def test_rendered_prompt_mentions_submit_for_evaluation_tool():
    """The prompt must reference the new submit_for_evaluation MCP tool, replacing the prior 'metrics in share_finding' guidance."""
    # Arrange
    ctx = {
        "assigned_entities": ["uk"], "server_url": "x", "workspace_dir": "/w",
        "dataset_name": "d", "data_dir": "/d", "student_model": "g", "logs_dir": "/l",
        "target_idea_content": "x", "local_mode": "false",
    }

    # Act
    rendered = _render_prompt(**ctx)

    # Assert
    assert "submit_for_evaluation" in rendered
