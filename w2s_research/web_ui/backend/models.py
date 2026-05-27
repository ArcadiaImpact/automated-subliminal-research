"""Database models for experiment tracking."""
import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

from w2s_research.web_ui.backend import config

db = SQLAlchemy()

# Bump this when the SQLAlchemy schema in this file changes incompatibly.
# At startup, if the stored version in `schema_meta` differs, we drop+recreate.
DB_SCHEMA_VERSION = 5


class SchemaMeta(db.Model):
    """Single-row table holding the current schema version. Used to detect
    incompatible upgrades and drop-and-recreate the DB on mismatch."""
    __tablename__ = 'schema_meta'
    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.Integer, nullable=False)


def ensure_schema_current():
    """If the stored schema version differs from DB_SCHEMA_VERSION, drop all
    tables and recreate. Idempotent. Must be called inside an app_context."""
    db.create_all()  # ensures schema_meta exists even on a fresh DB
    row = db.session.query(SchemaMeta).first()
    if row is None:
        db.session.add(SchemaMeta(version=DB_SCHEMA_VERSION))
        db.session.commit()
        return
    if row.version == DB_SCHEMA_VERSION:
        return
    # Mismatch — destructive upgrade.
    print(
        f"[schema] DB schema version {row.version} != code {DB_SCHEMA_VERSION}; "
        f"dropping all tables and recreating."
    )
    db.drop_all()
    db.create_all()
    db.session.add(SchemaMeta(version=DB_SCHEMA_VERSION))
    db.session.commit()


def _safe_datetime_subtract(dt1, dt2):
    """
    Safely subtract two datetimes, handling both timezone-aware and naive datetimes.
    Assumes naive datetimes are in UTC.
    """
    from datetime import timezone
    if dt1.tzinfo is None:
        dt1 = dt1.replace(tzinfo=timezone.utc)
    if dt2.tzinfo is None:
        dt2 = dt2.replace(tzinfo=timezone.utc)
    return (dt1 - dt2).total_seconds()


class Idea(db.Model):
    """Stores all research ideas with their full content.
    
    This is the single source of truth for idea metadata.
    Experiments reference ideas by name.
    """
    
    __tablename__ = 'ideas'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False, unique=True, index=True)
    uid = db.Column(db.String(100), nullable=True, unique=True, index=True)  # Unique identifier for S3 paths
    
    # Full idea content
    idea_json = db.Column(db.Text, nullable=True)  # Full JSON (~4KB each)

    # Extracted fields for easy querying
    description = db.Column(db.Text, nullable=True)
    
    # Metadata
    source = db.Column(db.String(50), nullable=True)  # 'manual'
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    
    # Tags for categorization
    is_baseline = db.Column(db.Boolean, default=False, nullable=False)  # Human-verified baseline (vanilla_w2s, etc.)

    @classmethod
    def from_dict(cls, idea_dict: dict, source: str = None, is_baseline: bool = False) -> 'Idea':
        """Create an Idea from a dictionary."""
        import json
        return cls(
            name=idea_dict.get('Name', ''),
            uid=idea_dict.get('uid'),
            idea_json=json.dumps(idea_dict),
            description=idea_dict.get('Description', ''),
            source=source,
            is_baseline=is_baseline,
        )
    
    def get_dict(self) -> dict:
        """Get the full idea dictionary."""
        import json
        if not self.idea_json:
            return {'Name': self.name, 'Description': self.description or ''}
        return json.loads(self.idea_json)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'name': self.name,
            'uid': self.uid,  # Include UID for API responses
            'description': self.description,
            'source': self.source,
            'is_baseline': self.is_baseline,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
    
    def to_full_dict(self) -> dict:
        """Get full idea dict including parsed JSON."""
        result = self.to_dict()
        result['idea_data'] = self.get_dict()
        return result


class Experiment(db.Model):
    """Tracks experiment execution and results.
    
    Each run of an idea+config creates a new row. Multiple runs of the same
    idea+config are allowed (historical runs are preserved).
    """

    __tablename__ = 'experiments'
    
    # No unique constraint - each run creates a new row
    # This allows tracking historical runs of the same idea+config

    id = db.Column(db.Integer, primary_key=True)
    idea_name = db.Column(db.String(200), nullable=False, index=True)
    idea_title = db.Column(db.String(500), nullable=True)
    idea_description = db.Column(db.Text, nullable=True)
    
    # Full configuration - allows same idea to be run with different configs
    dataset = db.Column(db.String(200), nullable=False, default="math-claudefilter-imbalance", index=True)
    weak_model = db.Column(db.String(200), nullable=False, default="Qwen/Qwen1.5-0.5B", index=True)
    strong_model = db.Column(db.String(200), nullable=False, default="Qwen/Qwen3-4B-Base", index=True)

    # Status: 'queued', 'running', 'completed', 'failed'
    status = db.Column(db.String(20), default='queued', nullable=False)

    # Shape C entity assignment
    assigned_entities = db.Column(db.Text, nullable=True)   # JSON list, set at queue time

    # Results (legacy W2S — kept for backward compat; NULL for Shape C rows)
    transfer_acc_std = db.Column(db.Float, nullable=True)
    num_seeds = db.Column(db.Integer, nullable=True)
    seeds = db.Column(db.Text, nullable=True)

    # Timing
    queue_time = db.Column(db.DateTime, default=datetime.now, nullable=False)
    start_time = db.Column(db.DateTime, nullable=True)
    end_time = db.Column(db.DateTime, nullable=True)

    # Logs and errors
    logs = db.Column(db.Text, nullable=True)
    error_msg = db.Column(db.Text, nullable=True)
    
    # RunPod tracking (for distributed execution)
    pod_id = db.Column(db.String(100), nullable=True)  # RunPod pod ID
    idea_uid = db.Column(db.String(100), nullable=True, index=True)  # Unique identifier for the idea (used for S3 paths)
    run_id = db.Column(db.String(50), nullable=True)  # Timestamp-based run ID for this execution (e.g., "run-20251213-143052")
    results_uploaded_to_s3 = db.Column(db.Boolean, default=False, nullable=False)  # True when results file exists in S3 (set by worker)

    # Retry tracking (for handling transient deployment failures)
    deploy_retry_count = db.Column(db.Integer, default=0, nullable=False)  # Number of deployment attempts
    last_deploy_attempt = db.Column(db.DateTime, nullable=True)  # Timestamp of last deployment attempt
    
    # Execution mode: 'local' (subprocess), 'docker' (local Docker), 'runpod' (cloud)
    execution_mode = db.Column(db.String(20), nullable=True, default=None)

    # GPU assignment for local/docker mode (e.g. "0,1,2,3")
    gpu_ids = db.Column(db.String(100), nullable=True, default=None)

    # Idea content (stored in DB to avoid file dependency)
    idea_json = db.Column(db.Text, nullable=True)  # Full idea JSON for backup
    
    # Idea metadata
    idea_created_at = db.Column(db.DateTime, nullable=True)  # When the idea was generated

    def to_dict(self):
        """Convert to dictionary for API responses."""
        import json
        duration_seconds = None
        if self.start_time and self.end_time:
            duration_seconds = _safe_datetime_subtract(self.end_time, self.start_time)
        elif self.start_time:
            # For current time, use timezone-aware datetime
            now = datetime.now()
            duration_seconds = _safe_datetime_subtract(now, self.start_time)

        # Compute S3 prefix when idea_uid and run_id exist (allows viewing logs/usage_stats before results.json)
        # Individual endpoints handle missing files gracefully
        s3_prefix = None
        if self.idea_uid and self.run_id:
            s3_prefix = (
                f"s3://{config.S3_BUCKET}/"
                f"{config.S3_IDEAS_PREFIX}{self.idea_uid}/{self.run_id}/"
            )

        # Determine if this is a baseline experiment.
        # Phantom-transfer has no method baselines shipped in-repo yet (the reference
        # phantom-transfer attack lives in the external phantom_transfer package).
        BASELINE_IDEA_NAMES = {'_strong_ceiling', '_weak_baseline'}
        is_baseline = self.idea_name in BASELINE_IDEA_NAMES
        
        return {
            'id': self.id,
            'idea_name': self.idea_name,
            'idea_title': self.idea_title,
            'idea_description': self.idea_description,
            'dataset': self.dataset,
            'weak_model': self.weak_model,
            'strong_model': self.strong_model,
            'status': self.status,
            'is_baseline': is_baseline,
            'assigned_entities': json.loads(self.assigned_entities) if self.assigned_entities else None,
            'transfer_acc_std': self.transfer_acc_std,
            'num_seeds': self.num_seeds,
            'seeds': json.loads(self.seeds) if self.seeds else None,
            'queue_time': self.queue_time.isoformat() if self.queue_time else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': duration_seconds,
            'logs': self.logs,
            'error_msg': self.error_msg,
            'pod_id': self.pod_id,
            'idea_uid': self.idea_uid,
            'run_id': self.run_id,
            's3_prefix': s3_prefix,
            'results_uploaded_to_s3': self.results_uploaded_to_s3,
            'deploy_retry_count': self.deploy_retry_count,
            'last_deploy_attempt': self.last_deploy_attempt.isoformat() if self.last_deploy_attempt else None,
            'execution_mode': self.execution_mode,
            'gpu_ids': self.gpu_ids,
        }


class Evaluation(db.Model):
    """Authoritative phantom-transfer evaluation result for one worker submission.

    Source of truth for pt_score. Findings link to Evaluations via Finding.evaluation_id;
    the leaderboard joins these tables. See spec §4.4, §5.1.
    """
    __tablename__ = 'evaluations'

    id = db.Column(db.Integer, primary_key=True)
    submitted_at = db.Column(db.DateTime, server_default=db.func.now())
    completed_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(16), nullable=False, default='queued')
    # status ∈ {'queued', 'running', 'done', 'failed'}

    # Inputs
    submission_dir = db.Column(db.Text, nullable=True)
    s3_path = db.Column(db.Text, nullable=True)
    base_model = db.Column(db.String(256), nullable=False)
    assigned_entities = db.Column(db.Text, nullable=False)   # JSON list
    held_out_entities = db.Column(db.Text, nullable=False)   # JSON list (may be "[]")
    mini = db.Column(db.Boolean, default=False, nullable=False)

    # Worker identity — set from env-injected request body; non-nullable (spec §4.5).
    experiment_id = db.Column(
        db.Integer, db.ForeignKey('experiments.id'), nullable=False, index=True
    )

    # Headline + per-criterion scores (spec §5.1)
    pt_score = db.Column(db.Float, nullable=True, index=True)
    pt_transfer_in_distribution = db.Column(db.Float, nullable=True)
    pt_transfer_in_distribution_vs_clean = db.Column(db.Float, nullable=True)
    pt_transfer_generalisation = db.Column(db.Float, nullable=True)
    pt_negative_mentions_lift = db.Column(db.Float, nullable=True)
    pt_negative_mentions_lift_vs_clean = db.Column(db.Float, nullable=True)
    pt_capability_delta_pp = db.Column(db.Float, nullable=True)
    pt_capability_delta_pp_vs_clean = db.Column(db.Float, nullable=True)
    pt_dataset_stealth_auc = db.Column(db.Float, nullable=True)
    pt_dataset_stealth_auc_vs_clean_pipeline = db.Column(db.Float, nullable=True)
    pt_model_stealth_acc = db.Column(db.Float, nullable=True)
    pt_model_stealth_acc_vs_clean = db.Column(db.Float, nullable=True)

    # Significance p-values
    pt_negative_mentions_p_vs_base = db.Column(db.Float, nullable=True)
    pt_negative_mentions_p_vs_clean = db.Column(db.Float, nullable=True)
    pt_model_stealth_p_vs_base = db.Column(db.Float, nullable=True)
    pt_model_stealth_p_vs_clean = db.Column(db.Float, nullable=True)
    pt_dataset_stealth_p_vs_raw = db.Column(db.Float, nullable=True)
    pt_dataset_stealth_p_vs_clean_pipeline = db.Column(db.Float, nullable=True)

    # Diagnostics
    pt_clean_control_source = db.Column(db.String(20), nullable=True)
    pt_clean_control_dataset_hash = db.Column(db.String(64), nullable=True)
    pt_raw_json = db.Column(db.Text, nullable=True)
    pt_eval_errors = db.Column(db.Text, nullable=True)

    def to_dict(self, scrub_held_out: bool = True) -> dict:
        """Serialize. When scrub_held_out=True (default for worker-facing endpoints),
        strip held_out_entities + per-entity held-out details from pt_raw_json
        but keep the aggregate pt_transfer_generalisation scalar (spec §4.5)."""
        import json as _json
        out = {
            'id': self.id,
            'evaluation_id': self.id,  # alias for MCP responses
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'submission_dir': self.submission_dir,
            's3_path': self.s3_path,
            'base_model': self.base_model,
            'assigned_entities': _json.loads(self.assigned_entities) if self.assigned_entities else [],
            'mini': self.mini,
            'experiment_id': self.experiment_id,
            'pt_score': self.pt_score,
            'pt_transfer_in_distribution': self.pt_transfer_in_distribution,
            'pt_transfer_in_distribution_vs_clean': self.pt_transfer_in_distribution_vs_clean,
            'pt_transfer_generalisation': self.pt_transfer_generalisation,
            'pt_negative_mentions_lift': self.pt_negative_mentions_lift,
            'pt_negative_mentions_lift_vs_clean': self.pt_negative_mentions_lift_vs_clean,
            'pt_capability_delta_pp': self.pt_capability_delta_pp,
            'pt_capability_delta_pp_vs_clean': self.pt_capability_delta_pp_vs_clean,
            'pt_dataset_stealth_auc': self.pt_dataset_stealth_auc,
            'pt_dataset_stealth_auc_vs_clean_pipeline': self.pt_dataset_stealth_auc_vs_clean_pipeline,
            'pt_model_stealth_acc': self.pt_model_stealth_acc,
            'pt_model_stealth_acc_vs_clean': self.pt_model_stealth_acc_vs_clean,
            'pt_negative_mentions_p_vs_base': self.pt_negative_mentions_p_vs_base,
            'pt_negative_mentions_p_vs_clean': self.pt_negative_mentions_p_vs_clean,
            'pt_model_stealth_p_vs_base': self.pt_model_stealth_p_vs_base,
            'pt_model_stealth_p_vs_clean': self.pt_model_stealth_p_vs_clean,
            'pt_dataset_stealth_p_vs_raw': self.pt_dataset_stealth_p_vs_raw,
            'pt_dataset_stealth_p_vs_clean_pipeline': self.pt_dataset_stealth_p_vs_clean_pipeline,
            'pt_clean_control_source': self.pt_clean_control_source,
            'pt_eval_errors': _json.loads(self.pt_eval_errors) if self.pt_eval_errors else None,
        }
        if not scrub_held_out:
            out['held_out_entities'] = _json.loads(self.held_out_entities) if self.held_out_entities else []
            out['pt_raw_json'] = _json.loads(self.pt_raw_json) if self.pt_raw_json else None
        else:
            # Strip per-entity held-out breakdowns from raw_json while keeping per-assigned data.
            if self.pt_raw_json:
                raw = _json.loads(self.pt_raw_json)
                raw.pop('per_held_out_entity', None)
                if isinstance(raw.get('raw'), dict):
                    raw['raw'].pop('per_held_out_entity', None)
                out['pt_raw_json'] = raw
            else:
                out['pt_raw_json'] = None
        return out


class Finding(db.Model):
    """Unified model for agent findings — merges the old Lesson + ForumPost tables.

    Created when agents share results, observations, hypotheses, etc.
    Supports voting, comments, and serves both the MCP query API and the web forum UI.
    """

    __tablename__ = 'findings'

    id = db.Column(db.Integer, primary_key=True)

    # Post identification (UUID for URL-safe lookups)
    post_id = db.Column(db.String(100), nullable=True, unique=True, index=True)

    # Content
    title = db.Column(db.String(500), nullable=True)
    content = db.Column(db.Text, nullable=True)  # MCP sends 'summary', stored here
    summary = db.Column(db.Text, nullable=True)  # alias accepted by MCP share_finding
    finding_type = db.Column(db.String(50), nullable=True, index=True)  # hypothesis, result, insight, error, observation

    # Shape C evaluation linkage (spec §4.5 #4: 1:1 Finding↔Evaluation)
    evaluation_id = db.Column(
        db.Integer, db.ForeignKey('evaluations.id'),
        nullable=True, unique=True,
    )
    experiment_id = db.Column(
        db.Integer, db.ForeignKey('experiments.id'),
        nullable=True, index=True,
    )

    # Source identification
    idea_uid = db.Column(db.String(100), nullable=True, index=True)
    idea_name = db.Column(db.String(200), nullable=True, index=True)
    run_id = db.Column(db.String(100), nullable=True, index=True)
    session_id = db.Column(db.String(100), nullable=True)

    # Experiment context
    dataset = db.Column(db.String(200), nullable=True, index=True)
    weak_model = db.Column(db.String(200), nullable=True)
    strong_model = db.Column(db.String(200), nullable=True)

    # Leaderboard fields
    idea_title = db.Column(db.String(500), nullable=True)  # Display title for leaderboard
    is_baseline = db.Column(db.Boolean, default=False, nullable=False)  # Marks baseline results
    seeds = db.Column(db.Text, nullable=True)  # JSON list of seed values

    # pt_score is now stored on Evaluation; this field is a denormalized cache for
    # leaderboard queries (populated by share_finding when auto-linking an Evaluation).
    pt_score = db.Column(db.Float, nullable=True, index=True)

    # Lesson-specific fields
    iteration = db.Column(db.Integer, nullable=True)
    config = db.Column(db.Text, nullable=True)  # JSON
    worked = db.Column(db.Boolean, nullable=True)

    # Snapshot / code (commit fields folded in — no separate Commit table)
    commit_id = db.Column(db.String(100), nullable=True, unique=True, index=True)
    s3_path = db.Column(db.String(500), nullable=True)
    s3_key = db.Column(db.String(500), nullable=True)
    parent_commit_id = db.Column(db.String(100), nullable=True)
    sequence_number = db.Column(db.Integer, nullable=True)
    files_snapshot = db.Column(db.Text, nullable=True)  # JSON list of files
    code_snippet = db.Column(db.Text, nullable=True)

    # Engagement
    upvotes = db.Column(db.Integer, default=0, nullable=False)
    downvotes = db.Column(db.Integer, default=0, nullable=False)
    comment_count = db.Column(db.Integer, default=0, nullable=False)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    # Relationship to comments
    comments = db.relationship('FindingComment', backref='finding', lazy='dynamic', cascade='all, delete-orphan')

    def _compute_eval_status(self, eval_row=None):
        """Derive eval_status from the joined Evaluation row.

        Returns one of: 'not_applicable' | 'pending' | 'verified' | 'failed' | 'orphaned'.

        Pass `eval_row` to skip the DB lookup when the caller has already loaded the
        Evaluation (used by list-endpoint batch loading).
        """
        import logging
        logger = logging.getLogger(__name__)

        if self.finding_type != 'result':
            return 'not_applicable'
        if self.evaluation_id is None:
            logger.warning(
                "Finding %s (idea_uid=%s) is finding_type='result' but evaluation_id is NULL — orphaned.",
                self.id, self.idea_uid,
            )
            return 'orphaned'
        if eval_row is None:
            eval_row = db.session.get(Evaluation, self.evaluation_id)
        if eval_row is None:
            logger.warning(
                "Finding %s references Evaluation %s but that row is missing — orphaned.",
                self.id, self.evaluation_id,
            )
            return 'orphaned'
        if eval_row.status == 'done':
            return 'verified'
        if eval_row.status == 'failed':
            return 'failed'
        return 'pending'  # 'queued' or 'running'

    def to_dict(self, include_comments=False, eval_row=None):
        """Convert to dictionary for API responses."""
        config_dict = None
        if self.config:
            try:
                config_dict = json.loads(self.config)
            except json.JSONDecodeError:
                pass

        # Compute eval_status; load Evaluation lazily if not already provided.
        if eval_row is None and self.evaluation_id is not None:
            eval_row = db.session.get(Evaluation, self.evaluation_id)
        eval_status = self._compute_eval_status(eval_row=eval_row)

        result = {
            'id': self.id,
            'post_id': self.post_id,
            'title': self.title,
            'content': self.content or self.summary,
            'summary': self.summary or self.content,
            'finding_type': self.finding_type,
            'eval_status': eval_status,
            # Shape C evaluation linkage
            'evaluation_id': self.evaluation_id,
            'experiment_id': self.experiment_id,
            'idea_uid': self.idea_uid,
            'idea_name': self.idea_name,
            'idea_title': self.idea_title,
            'run_id': self.run_id,
            'session_id': self.session_id,
            'dataset': self.dataset,
            'weak_model': self.weak_model,
            'strong_model': self.strong_model,
            'is_baseline': self.is_baseline,
            'seeds': json.loads(self.seeds) if self.seeds else None,
            # Denormalized pt_score for leaderboard (sourced from linked Evaluation)
            'pt_score': self.pt_score,
            'iteration': self.iteration,
            'config': config_dict,
            'worked': self.worked,
            'commit_id': self.commit_id,
            's3_path': self.s3_path,
            's3_key': self.s3_key,
            'parent_commit_id': self.parent_commit_id,
            'sequence_number': self.sequence_number,
            'files_snapshot': json.loads(self.files_snapshot) if self.files_snapshot else None,
            'file_count': len(json.loads(self.files_snapshot)) if self.files_snapshot else 0,
            'code_snippet': self.code_snippet,
            'upvotes': self.upvotes,
            'downvotes': self.downvotes,
            'score': self.upvotes - self.downvotes,
            'comment_count': self.comment_count,
            'created_at': (self.created_at.isoformat() + 'Z') if self.created_at else None,
            'updated_at': (self.updated_at.isoformat() + 'Z') if self.updated_at else None,
        }
        # Inline authoritative pt_* fields only when verified.
        if eval_status == 'verified' and eval_row is not None:
            for attr in (
                'pt_transfer_in_distribution',
                'pt_transfer_in_distribution_vs_clean',
                'pt_transfer_generalisation',
                'pt_negative_mentions_lift',
                'pt_negative_mentions_lift_vs_clean',
                'pt_capability_delta_pp',
                'pt_capability_delta_pp_vs_clean',
                'pt_dataset_stealth_auc',
                'pt_dataset_stealth_auc_vs_clean_pipeline',
                'pt_model_stealth_acc',
                'pt_model_stealth_acc_vs_clean',
            ):
                result[attr] = getattr(eval_row, attr)
            result['pt_score'] = eval_row.pt_score

        if include_comments:
            result['comments'] = [c.to_dict() for c in self.comments.order_by(FindingComment.created_at.asc()).all()]
        return result


class FindingComment(db.Model):
    """Comments on findings."""

    __tablename__ = 'finding_comments'

    id = db.Column(db.Integer, primary_key=True)

    # Link to finding
    finding_id = db.Column(db.Integer, db.ForeignKey('findings.id'), nullable=False, index=True)

    # Content
    content = db.Column(db.Text, nullable=False)
    author = db.Column(db.String(100), nullable=True)  # 'human' or agent session_id

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'finding_id': self.finding_id,
            'content': self.content,
            'author': self.author,
            'created_at': (self.created_at.isoformat() + 'Z') if self.created_at else None,
        }
