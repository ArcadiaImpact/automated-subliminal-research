"""Schema version + drop-on-mismatch behavior."""


def test_models_module_defines_db_schema_version_constant():
    """models.py must export DB_SCHEMA_VERSION as an int >= 1."""
    # Arrange / Act
    from w2s_research.web_ui.backend import models

    # Assert
    assert hasattr(models, "DB_SCHEMA_VERSION")
    assert isinstance(models.DB_SCHEMA_VERSION, int)
    assert models.DB_SCHEMA_VERSION >= 1


def test_ensure_schema_current_drops_and_recreates_when_version_mismatches(app):
    """When stored schema version differs from DB_SCHEMA_VERSION, ensure_schema_current
    must drop all tables, recreate them, and store the new version."""
    # Arrange — simulate a stale schema row.
    from w2s_research.web_ui.backend.models import (
        SchemaMeta, db, DB_SCHEMA_VERSION, ensure_schema_current,
    )
    with app.app_context():
        db.session.query(SchemaMeta).delete()
        db.session.add(SchemaMeta(version=DB_SCHEMA_VERSION - 1))
        db.session.commit()
        assert db.session.query(SchemaMeta).first().version == DB_SCHEMA_VERSION - 1

        # Act
        ensure_schema_current()

        # Assert
        rows = db.session.query(SchemaMeta).all()
        assert len(rows) == 1
        assert rows[0].version == DB_SCHEMA_VERSION
