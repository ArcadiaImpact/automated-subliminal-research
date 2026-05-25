"""Schema version + drop-on-mismatch behavior."""


def test_models_module_defines_db_schema_version_constant():
    """models.py must export DB_SCHEMA_VERSION as an int >= 1."""
    # Arrange / Act
    from w2s_research.web_ui.backend import models

    # Assert
    assert hasattr(models, "DB_SCHEMA_VERSION")
    assert isinstance(models.DB_SCHEMA_VERSION, int)
    assert models.DB_SCHEMA_VERSION >= 1
