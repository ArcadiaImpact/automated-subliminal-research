"""Flask backend for W2S Research Web UI."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from w2s_research.web_ui.backend import config
from w2s_research.web_ui.backend.models import db, Idea, Experiment, Evaluation, Finding, FindingComment
from w2s_research.web_ui.backend.worker import ExperimentWorker
from w2s_research.infrastructure.runpod import delete_pod

# Runtime config that can be modified via API (defaults from config module)
runtime_config = {
    'max_improvement_iterations': config.MAX_IMPROVEMENT_ITERATIONS,
    'skip_prior_work_search': config.SKIP_PRIOR_WORK_SEARCH,
    'max_concurrent_pods': config.MAX_CONCURRENT_PODS,
}


def create_app():
    """Create and configure the Flask application."""
    # Configure static folder for React build
    static_folder = Path(__file__).parent.parent / 'frontend' / 'build'

    app = Flask(__name__,
                static_folder=str(static_folder),
                static_url_path='')
    app.config.from_object(config)
    
    # Add explicit static file route for better reverse proxy support
    @app.route('/static/<path:filename>')
    def static_files(filename):
        """Serve static files explicitly with correct headers."""
        file_path = static_folder / 'static' / filename
        if not file_path.exists():
            return jsonify({'error': f'Static file not found: {filename}'}), 404
        
        response = send_from_directory(str(static_folder / 'static'), filename)
        # Set correct Content-Type
        if filename.endswith('.js'):
            response.headers['Content-Type'] = 'application/javascript; charset=utf-8'
        elif filename.endswith('.css'):
            response.headers['Content-Type'] = 'text/css; charset=utf-8'
        elif filename.endswith('.json'):
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        # Cache static assets (they have hashes, so safe to cache)
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
        return response

    # Initialize database
    db.init_app(app)

    # Enable CORS for React frontend
    CORS(app)

    # Support reverse proxy (RunPod HTTP service)
    # This ensures Flask knows it's behind a proxy and handles X-Forwarded-* headers correctly
    try:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(
            app.wsgi_app,
            x_for=1,      # Trust X-Forwarded-For header
            x_proto=1,    # Trust X-Forwarded-Proto header
            x_host=1,     # Trust X-Forwarded-Host header
            x_port=1,     # Trust X-Forwarded-Port header
            x_prefix=1   # Trust X-Forwarded-Prefix header
        )
    except ImportError:
        pass

    # Create tables (schema defined in models.py)
    # Note: existing databases were manually migrated to add transfer_acc_std,
    # so we don't run automatic ALTER TABLE logic here anymore.
    with app.app_context():
        db.create_all()

    return app


app = create_app()


worker = ExperimentWorker(app)


@app.route('/api/ideas', methods=['GET'])
def get_ideas():
    """Get all available ideas from the database.

    Ideas are shared across all configs, but queue status is config-specific.
    Pass dataset, weak_model, strong_model query params to get status for a specific config.

    Includes:
    - Baseline ideas from DB (is_baseline=True) - shown first, not queueable
    - User-created ideas from DB - queueable
    """
    try:
        ideas = []
        seen_names = set()

        # 1. Load baseline ideas from DB first (these take priority)
        db_ideas = Idea.query.filter_by(is_baseline=True).all()
        for db_idea in db_ideas:
            idea_dict = db_idea.get_dict()
            idea_dict['is_baseline'] = db_idea.is_baseline
            idea_dict['source'] = db_idea.source
            ideas.append(idea_dict)
            seen_names.add(idea_dict.get('Name', ''))

        # 2. Load user-created ideas from DB
        user_ideas = Idea.query.filter_by(is_baseline=False).all()
        for db_idea in user_ideas:
            idea_name = db_idea.name
            if idea_name and idea_name not in seen_names:
                idea_dict = db_idea.get_dict()
                idea_dict['is_baseline'] = db_idea.is_baseline
                idea_dict['source'] = db_idea.source
                ideas.append(idea_dict)
                seen_names.add(idea_name)

        # Get config filters from query params (for status display)
        dataset = request.args.get('dataset')
        weak_model = request.args.get('weak_model')
        strong_model = request.args.get('strong_model')
        
        # Build query for experiments with optional config filter
        exp_query = Experiment.query
        if dataset:
            exp_query = exp_query.filter_by(dataset=dataset)
        if weak_model:
            exp_query = exp_query.filter_by(weak_model=weak_model)
        if strong_model:
            exp_query = exp_query.filter_by(strong_model=strong_model)
        
        # Get all experiments for this config, then pick best per idea
        # Priority: queued/running first (show active status), then best PGR for completed
        all_experiments = exp_query.all()
        
        # Group by idea_name and pick the best representation
        queued_experiments = {}
        for exp in all_experiments:
            idea_name = exp.idea_name
            existing = queued_experiments.get(idea_name)
            
            if existing is None:
                queued_experiments[idea_name] = exp
            else:
                # Priority: queued > running > (completed with best PGR) > failed
                status_priority = {'queued': 0, 'running': 1, 'completed': 2, 'failed': 3, 'stopped': 4, 'cancelled': 5}
                existing_priority = status_priority.get(existing.status, 99)
                new_priority = status_priority.get(exp.status, 99)
                
                if new_priority < existing_priority:
                    # New experiment has higher priority status
                    queued_experiments[idea_name] = exp
                elif new_priority == existing_priority == 2:  # Both completed
                    # Pick the one with better PGR
                    if (exp.pgr or -999) > (existing.pgr or -999):
                        queued_experiments[idea_name] = exp

        # Add queue status to each idea (config-specific)
        for idea in ideas:
            idea_name = idea.get('Name', '')
            idea['in_queue'] = idea_name in queued_experiments

            # Get experiment details if queued for this config
            if idea['in_queue']:
                exp = queued_experiments[idea_name]
                idea['queue_status'] = exp.status
                idea['pgr'] = exp.pgr
                # Include idea_uid from experiment (may not be in original idea JSON)
                if exp.idea_uid and not idea.get('uid'):
                    idea['uid'] = exp.idea_uid
            else:
                # Clear any stale status from previous loads
                idea['queue_status'] = None
                idea['pgr'] = None

        return jsonify({
            'ideas': ideas,
            'total': len(ideas),
            'queued': len(queued_experiments),
            'baselines': len([i for i in ideas if i.get('is_baseline')]),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _create_idea_impl(idea_data, add_to_queue=False, created_via="web_ui",
                      dataset=None, weak_model=None, strong_model=None,
                      execution_mode=None, gpu_ids=None):
    """
    Core implementation for creating a new idea.

    Returns a dict with 'success', 'message', 'idea', 's3_uid', 'experiment', and optionally 'error'.
    This is used by both the API endpoint and the chat tool.
    
    Args:
        idea_data: The idea data dictionary
        add_to_queue: Whether to add the idea to the experiment queue
        created_via: Source of creation (for logging)
        dataset: Dataset for experiment (defaults to config.DATASET_NAME)
        weak_model: Weak model for experiment (defaults to config.WEAK_MODEL)
        strong_model: Strong model for experiment (defaults to config.STRONG_MODEL)
    """
    # Use defaults if not provided
    dataset = dataset or config.DATASET_NAME
    weak_model = weak_model or config.WEAK_MODEL
    strong_model = strong_model or config.STRONG_MODEL
    from w2s_research.infrastructure.s3_utils import ensure_idea_has_uid, upload_idea_by_uid

    # Validate required fields
    if not idea_data.get('Name'):
        return {'success': False, 'error': 'Name is required'}
    idea_name = idea_data['Name']

    # Check if idea with same name already exists in DB
    existing_idea = Idea.query.filter_by(name=idea_name).first()
    if existing_idea:
        return {'success': False, 'error': f'Idea with name "{idea_name}" already exists'}

    # Ensure idea has UID
    ensure_idea_has_uid(idea_data)

    # Upload to S3 (skip in local mode)
    uploaded_uid = idea_data.get('uid')
    if config.DEPLOY_TO_RUNPOD:
        try:
            uploaded_uid = upload_idea_by_uid(
                idea=idea_data,
                bucket_name=config.S3_BUCKET,
                prefix=config.S3_IDEAS_PREFIX,
                metadata={
                    "created_via": created_via,
                    "created_at": datetime.now().isoformat(),
                },
            )
        except Exception as s3_error:
            return {'success': False, 'error': f'Failed to upload to S3: {str(s3_error)}'}

    # Add to Ideas table (DB is source of truth)
    new_idea = Idea.from_dict(idea_data, source=created_via)
    db.session.add(new_idea)
    db.session.commit()

    # Always save idea.json to local results directory (needed for worker)
    idea_dir = config.RESULTS_DIR / idea_name
    idea_dir.mkdir(parents=True, exist_ok=True)
    idea_json_path = idea_dir / "idea.json"
    with open(idea_json_path, 'w') as f:
        json.dump(idea_data, f, indent=2)

    # Optionally add to queue
    experiment = None
    if add_to_queue:
        # Check if already queued or running for this config
        active = Experiment.query.filter_by(
            idea_name=idea_name,
            dataset=dataset,
            weak_model=weak_model,
            strong_model=strong_model
        ).filter(Experiment.status.in_(['queued', 'running'])).first()
        
        if active:
            # Return existing active experiment
            experiment = active
        else:
            # Create new experiment record (even if completed/failed exist - it's a new run)
            experiment = Experiment(
                idea_name=idea_name,
                idea_title=idea_name,
                idea_description=idea_data.get('Description', ''),
                idea_json=json.dumps(idea_data),  # Store full idea JSON
                dataset=dataset,
                weak_model=weak_model,
                strong_model=strong_model,
                execution_mode=execution_mode,
                gpu_ids=gpu_ids,
                status='queued',
                idea_uid=idea_data.get('uid'),
                assigned_entities=json.dumps(list(config.PT_ASSIGNED_ENTITIES)),
            )
            db.session.add(experiment)
            db.session.commit()

    return {
        'success': True,
        'message': 'Idea created successfully',
        'idea': idea_data,
        's3_uid': uploaded_uid,
        'experiment': experiment.to_dict() if experiment else None
    }


@app.route('/api/ideas', methods=['POST'])
def create_idea():
    """Create a new idea and optionally add it to the queue."""
    try:
        data = request.json
        idea_data = data.get('idea')
        add_to_queue = data.get('add_to_queue', False)
        dataset = data.get('dataset', config.DATASET_NAME)
        weak_model = data.get('weak_model', config.WEAK_MODEL)
        strong_model = data.get('strong_model', config.STRONG_MODEL)
        execution_mode = data.get('execution_mode')
        gpu_ids = data.get('gpu_ids')

        if not idea_data:
            return jsonify({'error': 'idea data is required'}), 400

        result = _create_idea_impl(
            idea_data, add_to_queue, created_via="web_ui",
            dataset=dataset, weak_model=weak_model, strong_model=strong_model,
            execution_mode=execution_mode, gpu_ids=gpu_ids,
        )

        if not result['success']:
            return jsonify({'error': result['error']}), 400

        return jsonify({
            'message': result['message'],
            'idea': result['idea'],
            's3_uid': result['s3_uid'],
            'experiment': result['experiment']
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/queue', methods=['GET'])
def get_queue():
    """Get all experiments in the queue, filtered by config (dataset, weak_model, strong_model).
    
    Shows all runs including historical ones. Marks 'is_best_run' for the experiment
    with the highest PGR per idea+config.
    """
    try:
        dataset = request.args.get('dataset')
        weak_model = request.args.get('weak_model')
        strong_model = request.args.get('strong_model')
        
        query = Experiment.query
        if dataset:
            query = query.filter_by(dataset=dataset)
        if weak_model:
            query = query.filter_by(weak_model=weak_model)
        if strong_model:
            query = query.filter_by(strong_model=strong_model)
        experiments = query.order_by(Experiment.queue_time.desc()).all()
        
        # Find best PGR per idea+config to mark 'is_best_run'
        best_pgr_by_config = {}
        for exp in experiments:
            if exp.status == 'completed' and exp.pgr is not None:
                key = (exp.idea_name, exp.dataset, exp.weak_model, exp.strong_model)
                if key not in best_pgr_by_config or exp.pgr > best_pgr_by_config[key]:
                    best_pgr_by_config[key] = exp.pgr
        
        # Convert to dicts and mark best runs
        result = []
        for exp in experiments:
            exp_dict = exp.to_dict()
            key = (exp.idea_name, exp.dataset, exp.weak_model, exp.strong_model)
            is_best = (
                exp.status == 'completed' and 
                exp.pgr is not None and 
                key in best_pgr_by_config and
                exp.pgr == best_pgr_by_config[key]
            )
            exp_dict['is_best_run'] = is_best
            result.append(exp_dict)
        
        return jsonify({
            'experiments': result,
            'total': len(result)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/queue/add', methods=['POST'])
def add_to_queue():
    """Add an idea to the experiment queue."""
    try:
        data = request.json
        idea_name = data.get('idea_name')
        idea_title = data.get('idea_title', '')
        idea_description = data.get('idea_description', '')
        dataset = data.get('dataset', config.DATASET_NAME)
        weak_model = data.get('weak_model', config.WEAK_MODEL)
        strong_model = data.get('strong_model', config.STRONG_MODEL)
        execution_mode = data.get('execution_mode')  # 'local', 'docker', 'runpod', or None
        gpu_ids = data.get('gpu_ids')  # e.g. "0,1,2,3" for local/docker mode

        if not idea_name:
            return jsonify({'error': 'idea_name is required'}), 400

        # Check Idea table for any marked as baseline
        baseline_idea = Idea.query.filter_by(name=idea_name, is_baseline=True).first()
        if baseline_idea:
            return jsonify({
                'error': f'Cannot queue baseline idea "{idea_name}". Baseline ideas use pre-computed results from cache.',
                'is_baseline': True
            }), 400

        # Note: No duplicate check - allows queueing the same idea multiple times
        # for parallel runs (e.g., stability testing). Each run gets a unique run_id.

        # Create the idea directory if it doesn't exist
        idea_dir = config.RESULTS_DIR / idea_name
        idea_dir.mkdir(parents=True, exist_ok=True)

        # Get idea data from DB
        idea_record = Idea.query.filter_by(name=idea_name).first()
        if not idea_record:
            return jsonify({'error': f'Idea {idea_name} not found'}), 404
        idea_data = idea_record.get_dict()

        # Ensure idea has UID (MUST be outside the if/else block!)
        from w2s_research.infrastructure.s3_utils import ensure_idea_has_uid
        ensure_idea_has_uid(idea_data)
        
        # Persist UID back to Ideas table if it was updated
        if idea_data.get('uid'):
            # Update both the uid column and idea_json in the Ideas record
            idea_record.uid = idea_data.get('uid')
            idea_record.idea_json = json.dumps(idea_data)
            
        # Copy the idea JSON to the idea directory (always update to ensure UID is saved)
        idea_json_path = idea_dir / "idea.json"
        with open(idea_json_path, 'w') as f:
            json.dump(idea_data, f, indent=2)

        # Create experiment record with full idea_json and idea_uid
        experiment = Experiment(
            idea_name=idea_name,
            idea_title=idea_name or idea_title,
            idea_description=idea_data.get('Description', '') or idea_description,
            idea_json=json.dumps(idea_data),  # Store full idea JSON
            idea_uid=idea_data.get('uid'),  # Store UID for easy lookup
            dataset=dataset,
            weak_model=weak_model,
            strong_model=strong_model,
            execution_mode=execution_mode,
            gpu_ids=gpu_ids,
            status='queued'
        )
        db.session.add(experiment)
        db.session.commit()

        return jsonify({
            'message': 'Idea added to queue',
            'experiment': experiment.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/queue/remove/<int:experiment_id>', methods=['DELETE'])
def remove_from_queue(experiment_id):
    """Remove an experiment from the queue."""
    try:
        experiment = db.session.get(Experiment, experiment_id)
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404

        # Can only remove queued experiments
        if experiment.status != 'queued':
            return jsonify({
                'error': f'Cannot remove {experiment.status} experiment'
            }), 400

        db.session.delete(experiment)
        db.session.commit()

        return jsonify({
            'message': 'Experiment removed from queue',
            'idea_name': experiment.idea_name
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/queue/rerun/<int:experiment_id>', methods=['POST'])
def rerun_experiment(experiment_id):
    """Rerun a failed or completed experiment by resetting it to queued status."""
    try:
        experiment = db.session.get(Experiment, experiment_id)
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404

        # Can only rerun failed or completed experiments
        if experiment.status not in ('failed', 'completed'):
            return jsonify({
                'error': f'Cannot rerun {experiment.status} experiment (only failed or completed experiments can be rerun)'
            }), 400

        # Reset experiment state for rerun
        old_logs = experiment.logs or ""
        old_status = experiment.status
        old_error = experiment.error_msg or "N/A"

        experiment.status = 'queued'
        experiment.queue_time = datetime.now()
        experiment.start_time = None
        experiment.end_time = None
        experiment.error_msg = None

        # Preserve old logs with rerun marker
        if old_status == 'failed':
            experiment.logs = f"""{'='*80}
RERUN ATTEMPT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

Previous attempt failed with error:
{old_error}

{'='*80}
PREVIOUS LOGS
{'='*80}

{old_logs}

{'='*80}
NEW ATTEMPT STARTING
{'='*80}

"""
        else:
            experiment.logs = f"""{'='*80}
RERUN ATTEMPT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

Previous attempt completed successfully (scores live on the linked Evaluation row).

{'='*80}
PREVIOUS LOGS
{'='*80}

{old_logs}

{'='*80}
NEW ATTEMPT STARTING
{'='*80}

"""

        db.session.commit()

        return jsonify({
            'message': 'Experiment requeued for rerun',
            'experiment': experiment.to_dict()
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/queue/kill/<int:experiment_id>', methods=['POST'])
def kill_experiment(experiment_id):
    """Kill a running experiment."""
    try:
        experiment = db.session.get(Experiment, experiment_id)
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404

        print(f"\n🛑 Kill request received for experiment {experiment_id}: {experiment.idea_name}")
        print(f"   Current status: {experiment.status}")

        # Can only kill running experiments
        if experiment.status != 'running':
            print(f"   ❌ Cannot kill - status is '{experiment.status}', not 'running'")
            return jsonify({
                'error': f'Cannot kill {experiment.status} experiment (only running experiments can be killed)'
            }), 400

        # Mark experiment as failed so worker will detect and abort
        experiment.status = 'failed'
        experiment.end_time = datetime.now()
        experiment.error_msg = 'Experiment killed by user'

        # Append kill message to logs
        kill_message = f"\n\n{'='*80}\n🛑 EXPERIMENT KILLED BY USER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n"
        if experiment.logs:
            experiment.logs += kill_message
        else:
            experiment.logs = kill_message

        # Kill the running process/container/pod
        if experiment.pod_id:
            pod_id = experiment.pod_id
            exec_mode = experiment.execution_mode or ''

            if pod_id.startswith('local-'):
                # Local mode (subprocess or docker) — kill by container name or process tree
                if exec_mode == 'docker':
                    # Docker container — stop it
                    container_name = pod_id.replace('local-', '', 1)
                    # Find container by name pattern
                    try:
                        import subprocess as sp
                        docker = config.DOCKER_EXECUTABLE if hasattr(config, 'DOCKER_EXECUTABLE') else 'docker'
                        # List containers matching the run_id
                        result = sp.run(
                            [docker, "ps", "-q", "--filter", f"name=w2s-local-"],
                            capture_output=True, text=True, timeout=10,
                        )
                        for cid in result.stdout.strip().split('\n'):
                            if cid:
                                sp.run([docker, "stop", cid], timeout=30)
                                print(f"   Stopped Docker container {cid}")
                        kill_message += f"\nDocker container(s) stopped.\n"
                        experiment.logs = (experiment.logs or "") + kill_message
                    except Exception as e:
                        error_note = f"\n⚠️  Failed to stop Docker container: {e}\n"
                        experiment.logs = (experiment.logs or "") + error_note
                        print(error_note)
                else:
                    # Subprocess mode — kill the agent process tree
                    try:
                        import subprocess as sp
                        # Find and kill the agent subprocess by matching the idea_uid
                        uid = experiment.idea_uid or ''
                        result = sp.run(
                            ["pgrep", "-f", f"run.py agent.*{uid}"],
                            capture_output=True, text=True, timeout=10,
                        )
                        for pid in result.stdout.strip().split('\n'):
                            if pid:
                                # Kill the process group to get child processes too
                                import signal
                                try:
                                    os.killpg(int(pid), signal.SIGTERM)
                                except ProcessLookupError:
                                    pass
                                except PermissionError:
                                    os.kill(int(pid), signal.SIGTERM)
                                print(f"   Killed process {pid}")
                        kill_message += f"\nLocal process(es) killed.\n"
                        experiment.logs = (experiment.logs or "") + kill_message
                    except Exception as e:
                        error_note = f"\n⚠️  Failed to kill local process: {e}\n"
                        experiment.logs = (experiment.logs or "") + error_note
                        print(error_note)
            else:
                # RunPod pod — delete it
                try:
                    print(f"   Attempting to delete RunPod pod {pod_id}...")
                    delete_pod(
                        pod_id=pod_id,
                        runpod_api_key=os.environ.get('RUNPOD_API_KEY'),
                    )
                    kill_message += f"\nRunPod pod {pod_id} deleted.\n"
                    experiment.logs = (experiment.logs or "") + kill_message
                    print(f"   RunPod pod {pod_id} deleted successfully")
                except Exception as pod_err:
                    error_note = f"\n⚠️  Failed to delete RunPod pod {pod_id}: {pod_err}\n"
                    experiment.logs = (experiment.logs or "") + error_note
                    print(error_note)

        db.session.commit()

        print(f"   ✓ Experiment marked as failed with error_msg='Experiment killed by user'")
        print(f"   Worker will detect kill on next log update and abort.")

        return jsonify({
            'message': 'Experiment killed',
            'experiment': experiment.to_dict()
        })

    except Exception as e:
        print(f"   ❌ Error killing experiment: {e}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiments/<int:experiment_id>', methods=['GET'])
def get_experiment(experiment_id):
    """Get details of a specific experiment."""
    try:
        experiment = db.session.get(Experiment, experiment_id)
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404

        return jsonify(experiment.to_dict())

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiment/<int:experiment_id>/s3-data', methods=['GET'])
def get_experiment_s3_data(experiment_id):
    """Fetch worker log and results from S3 for an experiment."""
    try:
        from w2s_research.infrastructure.s3_utils import get_s3_client

        experiment = db.session.get(Experiment, experiment_id)
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404

        if not experiment.results_uploaded_to_s3 or not experiment.idea_uid or not experiment.run_id:
            return jsonify({'error': 'No S3 data available for this experiment'}), 404

        s3_client = get_s3_client()

        bucket = config.S3_BUCKET
        key_prefix = f"{config.S3_IDEAS_PREFIX}{experiment.idea_uid}/{experiment.run_id}"
        
        worker_log = None
        results = None
        
        # Try to fetch worker log
        try:
            log_key = f"{key_prefix}/worker_pod.log"
            response = s3_client.get_object(Bucket=bucket, Key=log_key)
            worker_log = response['Body'].read().decode('utf-8')
        except Exception as e:
            if 'NoSuchKey' not in str(e):
                print(f"Error fetching worker log: {e}")
        
        # Try to fetch results.json
        try:
            results_key = f"{key_prefix}/results.json"
            response = s3_client.get_object(Bucket=bucket, Key=results_key)
            results = json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            if 'NoSuchKey' not in str(e):
                print(f"Error fetching results: {e}")

        # For auto mode, also fetch findings.json with iteration history
        findings = None
        is_auto_mode = experiment.run_id and experiment.run_id.startswith("auto-")
        if is_auto_mode:
            try:
                findings_key = f"{key_prefix}/findings.json"
                response = s3_client.get_object(Bucket=bucket, Key=findings_key)
                findings = json.loads(response['Body'].read().decode('utf-8'))
            except Exception as e:
                if 'NoSuchKey' not in str(e):
                    print(f"Error fetching findings: {e}")

        return jsonify({
            'worker_log': worker_log,
            'results': results,
            'findings': findings,
            'is_auto_mode': is_auto_mode,
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiment/<int:experiment_id>/findings', methods=['GET'])
def get_experiment_findings(experiment_id):
    """
    Lightweight endpoint to fetch just the findings.json for auto mode experiments.

    Useful for polling iteration progress during an active run.
    Returns the findings with all iteration results.
    """
    try:
        from w2s_research.infrastructure.s3_utils import get_s3_client

        experiment = db.session.get(Experiment, experiment_id)
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404

        # Check if this is an auto mode experiment
        if not experiment.run_id or not experiment.run_id.startswith("auto-"):
            return jsonify({'error': 'Not an auto mode experiment'}), 400

        if not experiment.idea_uid:
            return jsonify({'error': 'No idea_uid available'}), 404

        s3_client = get_s3_client()
        bucket = config.S3_BUCKET
        key_prefix = f"{config.S3_IDEAS_PREFIX}{experiment.idea_uid}/{experiment.run_id}"

        try:
            findings_key = f"{key_prefix}/findings.json"
            response = s3_client.get_object(Bucket=bucket, Key=findings_key)
            findings = json.loads(response['Body'].read().decode('utf-8'))

            return jsonify({
                'findings': findings,
                'experiment_id': experiment_id,
                'status': experiment.status,
            })
        except Exception as e:
            if 'NoSuchKey' in str(e):
                return jsonify({'error': 'Findings not found (experiment may not have started yet)'}), 404
            raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiment/<int:experiment_id>/usage-stats', methods=['GET'])
def get_experiment_usage_stats(experiment_id):
    """
    Fetch usage statistics (skill and MCP tool usage) for an experiment.

    Checks local filesystem first (for local/docker mode), then S3 (for runpod mode).
    """
    try:
        experiment = db.session.get(Experiment, experiment_id)
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404

        usage_stats = None

        # 1. Try local filesystem (works for local & docker modes)
        local_stats_path = Path(config.WORKSPACE_DIR) / "w2s_research" / "research_loop" / "logs" / "usage_stats.json"
        if local_stats_path.exists():
            try:
                with open(local_stats_path) as f:
                    usage_stats = json.load(f)
            except Exception:
                pass

        # 2. Fall back to S3 (runpod mode)
        if not usage_stats and experiment.idea_uid and experiment.run_id:
            try:
                from w2s_research.infrastructure.s3_utils import get_s3_client
                s3_client = get_s3_client()
                bucket = config.S3_BUCKET
                key_prefix = f"{config.S3_IDEAS_PREFIX}{experiment.idea_uid}/{experiment.run_id}"
                stats_key = f"{key_prefix}/logs/usage_stats.json"
                response = s3_client.get_object(Bucket=bucket, Key=stats_key)
                usage_stats = json.loads(response['Body'].read().decode('utf-8'))
            except Exception as e:
                if 'NoSuchKey' not in str(e) and 'credentials' not in str(e).lower():
                    print(f"Error fetching usage stats from S3: {e}")

        if not usage_stats:
            return jsonify({
                'usage_stats': None,
                'error': 'No usage stats found (experiment may not have used tracked APIs)',
            }), 404

        return jsonify({
            'usage_stats': usage_stats,
            'experiment_id': experiment_id,
            'idea_uid': experiment.idea_uid,
            'run_id': experiment.run_id,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500




@app.route('/api/evaluations', methods=['GET'])
def list_evaluations():
    """List evaluations filtered by experiment_id. Scrubs held-out info."""
    from w2s_research.web_ui.backend.models import Evaluation
    experiment_id = request.args.get('experiment_id', type=int)
    if experiment_id is None:
        return jsonify({'error': 'experiment_id query param required'}), 400
    rows = (
        Evaluation.query
        .filter_by(experiment_id=experiment_id, status='done')
        .filter(Evaluation.pt_score.isnot(None))
        .order_by(Evaluation.pt_score.desc())
        .all()
    )
    return jsonify({
        'evaluations': [r.to_dict(scrub_held_out=True) for r in rows],
        'total': len(rows),
    })


@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """Leaderboard of published phantom-transfer findings, sorted by pt_score desc.

    Joins Finding to its linked Evaluation (UNIQUE(evaluation_id) guarantees 1:1).
    Only includes findings with finding_type='result' and a done Evaluation.
    """
    from w2s_research.web_ui.backend.models import Evaluation, Finding, db

    rows = (
        db.session.query(Finding, Evaluation)
        .join(Evaluation, Finding.evaluation_id == Evaluation.id)
        .filter(
            Finding.finding_type == 'result',
            Evaluation.status == 'done',
            Evaluation.pt_score.isnot(None),
        )
        .order_by(Evaluation.pt_score.desc())
        .all()
    )
    return jsonify({
        'findings': [
            {**f.to_dict(),
             'evaluation': e.to_dict(scrub_held_out=True),
             'pt_score': e.pt_score}
            for f, e in rows
        ],
        'total': len(rows),
    })




@app.route('/api/config', methods=['GET'])
def get_config():
    """Get experiment configuration."""
    return jsonify({
        'dataset': config.DATASET_NAME,
        'dataset_dir': config.DATASET_DIR,
        'weak_model': config.WEAK_MODEL,
        'strong_model': config.STRONG_MODEL,
        'available_datasets': config.AVAILABLE_DATASETS,
        'available_weak_models': config.AVAILABLE_WEAK_MODELS,
        'available_strong_models': config.AVAILABLE_STRONG_MODELS,
        'seeds': config.SEEDS,
        'epochs': config.EPOCHS,
        'num_gpus': config.NUM_GPUS,
        'poll_interval': config.POLL_INTERVAL,
        'max_improvement_iterations': runtime_config['max_improvement_iterations'],
        'skip_prior_work_search': runtime_config['skip_prior_work_search'],
        'full_auto_worker_max_runtime_seconds': config.FULL_AUTO_WORKER_MAX_RUNTIME_SECONDS,
        'max_concurrent_pods': runtime_config['max_concurrent_pods'],
    })


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update runtime configuration.
    
    Accepts:
    - dataset: Dataset name (must be in AVAILABLE_DATASETS)
    - weak_model: Weak model name (must be in AVAILABLE_WEAK_MODELS)
    - strong_model: Strong model name (must be in AVAILABLE_STRONG_MODELS)
    - max_improvement_iterations: Number of improvement iterations (0-100)
    - skip_prior_work_search: Boolean to skip prior work search (useful for stability testing)
    """
    data = request.get_json()
    updated = {}

    # Update dataset
    if 'dataset' in data:
        dataset = data['dataset']
        if dataset not in config.AVAILABLE_DATASETS:
            return jsonify({'error': f'Invalid dataset: {dataset}. Available: {config.AVAILABLE_DATASETS}'}), 400
        config.DATASET_NAME = dataset
        config.DATASET_DIR = f"{config.DATA_BASE_DIR}/{dataset}"
        updated['dataset'] = dataset

    # Update weak model
    if 'weak_model' in data:
        weak_model = data['weak_model']
        if weak_model not in config.AVAILABLE_WEAK_MODELS:
            return jsonify({'error': f'Invalid weak_model: {weak_model}. Available: {config.AVAILABLE_WEAK_MODELS}'}), 400
        config.WEAK_MODEL = weak_model
        updated['weak_model'] = weak_model

    # Update strong model
    if 'strong_model' in data:
        strong_model = data['strong_model']
        if strong_model not in config.AVAILABLE_STRONG_MODELS:
            return jsonify({'error': f'Invalid strong_model: {strong_model}. Available: {config.AVAILABLE_STRONG_MODELS}'}), 400
        config.STRONG_MODEL = strong_model
        updated['strong_model'] = strong_model

    # Update max improvement iterations
    if 'max_improvement_iterations' in data:
        try:
            value = int(data['max_improvement_iterations'])
            if value < 0 or value > 100:
                return jsonify({'error': 'max_improvement_iterations must be between 0 and 100'}), 400
            runtime_config['max_improvement_iterations'] = value
            config.MAX_IMPROVEMENT_ITERATIONS = value
            updated['max_improvement_iterations'] = value
        except (ValueError, TypeError):
            return jsonify({'error': 'max_improvement_iterations must be an integer'}), 400

    # Update skip prior work search
    if 'skip_prior_work_search' in data:
        value = data['skip_prior_work_search']
        if isinstance(value, bool):
            runtime_config['skip_prior_work_search'] = value
            config.SKIP_PRIOR_WORK_SEARCH = value
            # Also update environment variable for consistency
            os.environ['SKIP_PRIOR_WORK_SEARCH'] = '1' if value else '0'
            updated['skip_prior_work_search'] = value
        else:
            return jsonify({'error': 'skip_prior_work_search must be a boolean'}), 400

    # Update max concurrent pods
    if 'max_concurrent_pods' in data:
        try:
            value = int(data['max_concurrent_pods'])
            if value < 1 or value > 10:
                return jsonify({'error': 'max_concurrent_pods must be between 1 and 10'}), 400
            runtime_config['max_concurrent_pods'] = value
            config.MAX_CONCURRENT_PODS = value
            updated['max_concurrent_pods'] = value
        except (ValueError, TypeError):
            return jsonify({'error': 'max_concurrent_pods must be an integer'}), 400

    # Log config change
    if updated:
        print(f"[CONFIG] Updated: {updated}")

    return jsonify({
        'success': True,
        'updated': updated,
        'current': {
            'dataset': config.DATASET_NAME,
            'weak_model': config.WEAK_MODEL,
            'strong_model': config.STRONG_MODEL,
            'max_improvement_iterations': runtime_config['max_improvement_iterations'],
            'skip_prior_work_search': runtime_config['skip_prior_work_search'],
        }
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall statistics, filtered by config (dataset, weak_model, strong_model).
    
    Note: 'completed' without PGR is treated as 'failed' since
    a successful experiment should always produce a PGR value.
    """
    try:
        dataset = request.args.get('dataset')
        weak_model = request.args.get('weak_model')
        strong_model = request.args.get('strong_model')
        
        def apply_filters(query):
            """Apply config filters to a query."""
            if dataset:
                query = query.filter_by(dataset=dataset)
            if weak_model:
                query = query.filter_by(weak_model=weak_model)
            if strong_model:
                query = query.filter_by(strong_model=strong_model)
            return query
        
        base_query = apply_filters(Experiment.query)
        
        total = base_query.count()
        queued = apply_filters(Experiment.query.filter_by(status='queued')).count()
        running = apply_filters(Experiment.query.filter_by(status='running')).count()
        
        # Completed = status is 'completed' AND has PGR value
        completed = apply_filters(Experiment.query.filter(
            Experiment.status == 'completed',
            Experiment.pgr.isnot(None)
        )).count()
        
        # Failed = status is 'failed' OR (status is 'completed' but no PGR)
        explicit_failed = apply_filters(Experiment.query.filter_by(status='failed')).count()
        completed_without_pgr = apply_filters(Experiment.query.filter(
            Experiment.status == 'completed',
            Experiment.pgr.is_(None)
        )).count()
        failed = explicit_failed + completed_without_pgr

        return jsonify({
            'total': total,
            'queued': queued,
            'running': running,
            'completed': completed,
            'failed': failed
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500







# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve React app - catch-all route for React Router."""
    static_folder = Path(app.static_folder)

    # Don't interfere with API routes (shouldn't reach here, but just in case)
    if path.startswith('api/'):
        return jsonify({'error': 'API route not found'}), 404

    # Handle static assets (JS, CSS, images, etc.)
    # React build puts these in /static/ subdirectory
    if path.startswith('static/'):
        # Remove 'static/' prefix and serve from static folder
        file_path = path.replace('static/', '', 1)
        full_path = static_folder / 'static' / file_path
        if full_path.exists() and full_path.is_file():
            response = send_from_directory(static_folder / 'static', file_path)
            # Ensure correct Content-Type
            if file_path.endswith('.js'):
                response.headers['Content-Type'] = 'application/javascript; charset=utf-8'
            elif file_path.endswith('.css'):
                response.headers['Content-Type'] = 'text/css; charset=utf-8'
            elif file_path.endswith('.json'):
                response.headers['Content-Type'] = 'application/json; charset=utf-8'
            # Add CORS headers for static assets (needed for some browsers)
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
        else:
            # File not found in static directory
            return jsonify({'error': f'Static file not found: {path}'}), 404
    
    # Handle other static files at root (favicon.ico, manifest.json, etc.)
    if path:
        file_path = static_folder / path
        if file_path.exists() and file_path.is_file():
            response = send_from_directory(app.static_folder, path)
            # Add CORS headers
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response

    # Otherwise serve index.html (for React Router)
    # This handles all routes that don't match static files
    index_path = static_folder / 'index.html'
    if index_path.exists():
        response = send_from_directory(app.static_folder, 'index.html')
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        # Add cache control headers for index.html (should not be cached)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    else:
        return jsonify({
            'error': 'Frontend not built. Run "npm run build" in frontend directory.'
        }), 404




def auto_queue_seed_ideas():
    """At startup, queue any seed idea (source='seed') that doesn't yet have
    an Experiment record. The orchestrator's worker loop will then pick them up
    automatically, respecting MAX_CONCURRENT_PODS — so the first
    MAX_CONCURRENT_PODS ideas start immediately and the rest wait for a slot
    to free up.

    Idempotent: skips ideas that already have ANY Experiment row (queued,
    running, completed, failed, stopped, cancelled), so server restarts don't
    duplicate. To re-run a finished seed, queue it manually via the dashboard.
    """
    print("\n[Startup] Auto-queueing seed ideas...")
    with app.app_context():
        seed_ideas = (
            Idea.query.filter_by(source="seed", is_baseline=False)
            .order_by(Idea.name)
            .all()
        )
        if not seed_ideas:
            print("[Startup]   no seed ideas to auto-queue")
            return

        from w2s_research.infrastructure.s3_utils import ensure_idea_has_uid

        queued = 0
        skipped = 0
        for idea in seed_ideas:
            existing = Experiment.query.filter_by(idea_name=idea.name).first()
            if existing is not None:
                skipped += 1
                continue

            idea_data = idea.get_dict()
            try:
                ensure_idea_has_uid(idea_data)
                if idea_data.get("uid") and not idea.uid:
                    idea.uid = idea_data["uid"]
                    idea.idea_json = json.dumps(idea_data)
            except Exception as e:  # noqa: BLE001
                print(f"[Startup]   warning: UID resolution for {idea.name} failed: {e}")

            # Persist idea.json under RESULTS_DIR so worker pods can read it.
            try:
                idea_dir = config.RESULTS_DIR / idea.name
                idea_dir.mkdir(parents=True, exist_ok=True)
                with open(idea_dir / "idea.json", "w") as f:
                    json.dump(idea_data, f, indent=2)
            except Exception as e:  # noqa: BLE001
                print(f"[Startup]   warning: writing idea.json for {idea.name} failed: {e}")

            experiment = Experiment(
                idea_name=idea.name,
                idea_title=idea.name,
                idea_description=idea_data.get("Description", ""),
                idea_json=json.dumps(idea_data),
                idea_uid=idea_data.get("uid"),
                dataset=config.DATASET_NAME,
                weak_model=config.WEAK_MODEL,
                strong_model=config.STRONG_MODEL,
                status="queued",
            )
            db.session.add(experiment)
            queued += 1
            print(f"[Startup]   queued: {idea.name}")

        db.session.commit()
        print(
            f"[Startup] Auto-queue: queued={queued}, skipped={skipped} "
            f"(MAX_CONCURRENT_PODS={config.MAX_CONCURRENT_PODS}; "
            f"orchestrator will start up to that many in parallel and cycle through the rest)"
        )


def ensure_seed_ideas_exist():
    """Auto-ingest warm-start seed ideas from `w2s_research/ideas/*/idea.md`.

    Each subfolder of `w2s_research/ideas/` containing an `idea.md` file is treated
    as a queueable seed idea: folder name -> Idea.name, file contents -> Idea.description
    (also rendered as the worker's `target_idea_content`). Re-running on startup
    updates descriptions in-place when the file changes.

    Excluded: `TEMPLATE/` (scaffolding), dot-folders, `__pycache__`. Seeds are
    `is_baseline=False` (so they appear and are queueable like user-created ideas)
    with `source='seed'` for traceability.
    """
    ideas_root = Path(__file__).resolve().parents[2] / "ideas"
    if not ideas_root.is_dir():
        print(f"[Startup] No ideas/ directory at {ideas_root}; skipping seed ingestion")
        return

    print(f"\n[Startup] Scanning for seed ideas in {ideas_root}...")
    with app.app_context():
        created = 0
        updated = 0
        skipped = 0

        for entry in sorted(ideas_root.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name in {"TEMPLATE", "__pycache__"} or entry.name.startswith("."):
                continue
            md_path = entry / "idea.md"
            if not md_path.is_file():
                skipped += 1
                continue

            description = md_path.read_text(encoding="utf-8").strip()
            if not description:
                print(f"[Startup]   skipping {entry.name}: idea.md is empty")
                skipped += 1
                continue

            idea_name = entry.name
            idea_data = {
                "Name": idea_name,
                "Description": description,
            }
            existing = Idea.query.filter_by(name=idea_name).first()
            if existing:
                # Refresh description from disk if it has drifted.
                if existing.description != description:
                    existing.description = description
                    existing.idea_json = json.dumps(idea_data)
                    updated += 1
            else:
                new_idea = Idea.from_dict(idea_data, source="seed", is_baseline=False)
                db.session.add(new_idea)
                created += 1

        db.session.commit()
        print(f"[Startup] Seed ideas: created={created}, updated={updated}, skipped={skipped}")






# =============================================================================
# Findings API Endpoints (unified — replaces old /api/lessons/* and /api/forum/*)
# =============================================================================

@app.route('/api/findings/share', methods=['POST'])
def share_finding():
    """Share a finding from an experiment iteration.

    Creates a single Finding record (replaces the old Lesson + ForumPost pair).
    Workers call this after iterations; other runs query findings to learn.

    For finding_type='result', the server auto-links the worker's best-scoring done
    Evaluation (by experiment_id) — evaluation_id and pt_score are server-assigned;
    agents must NOT provide them directly (trust model §4.5).

    Required fields: summary
    For finding_type='result': experiment_id is also required.
    Optional fields: title, idea_uid, idea_name, run_id, iteration,
                     config, worked, dataset, weak_model,
                     strong_model, finding_type, commit_id, s3_path, s3_key,
                     parent_commit_id, sequence_number, files_snapshot, code_snippet
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Missing request body'}), 400

        # Reject agent-provided fields that would bypass the trust model.
        if 'evaluation_id' in data:
            return jsonify({'error': 'evaluation_id is server-assigned; do not provide'}), 400
        if 'metrics' in data:
            return jsonify({'error': 'metrics is server-assigned (read from Evaluation); do not provide'}), 400

        summary = data.get('summary')
        if not summary:
            return jsonify({'error': 'Missing summary field'}), 400
        if len(summary) > 5000:
            return jsonify({'error': 'Summary too long (max 5000 characters)'}), 400

        config_data = data.get('config')
        config_json = json.dumps(config_data) if config_data else None

        files_data = data.get('files_snapshot')
        files_json = json.dumps(files_data) if files_data else None

        # Build title
        title = data.get('title')
        if not title:
            idea_name = data.get('idea_name') or 'experiment'
            worked = data.get('worked')
            if worked is True:
                title = f"[Success] {idea_name}"
            elif worked is False:
                title = f"[Failed] {idea_name}"
            else:
                title = f"[Finding] {idea_name}"

        import uuid
        finding = Finding(
            post_id=str(uuid.uuid4()),
            title=title,
            content=summary,
            finding_type=data.get('finding_type', 'result' if data.get('worked') else 'observation'),
            idea_uid=data.get('idea_uid'),
            idea_name=data.get('idea_name'),
            run_id=data.get('run_id'),
            session_id=data.get('session_id'),
            dataset=data.get('dataset'),
            weak_model=data.get('weak_model'),
            strong_model=data.get('strong_model'),
            iteration=data.get('iteration'),
            config=config_json,
            worked=data.get('worked'),
            commit_id=data.get('commit_id'),
            s3_path=data.get('s3_path'),
            s3_key=data.get('s3_key'),
            parent_commit_id=data.get('parent_commit_id'),
            sequence_number=data.get('sequence_number'),
            files_snapshot=files_json,
            code_snippet=data.get('code_snippet'),
        )
        db.session.add(finding)
        db.session.flush()  # get finding.id without committing yet

        # For finding_type='result', auto-link the worker's best-scoring done Evaluation.
        evaluation_id = None
        pt_score = None
        if finding.finding_type == 'result':
            experiment_id = data.get('experiment_id')
            if not experiment_id:
                db.session.rollback()
                return jsonify({'error': 'experiment_id required for finding_type=result'}), 400
            best_eval = (
                Evaluation.query
                .filter_by(experiment_id=experiment_id, status='done')
                .filter(Evaluation.pt_score.isnot(None))
                .order_by(Evaluation.pt_score.desc())
                .first()
            )
            if best_eval is None:
                db.session.rollback()
                return jsonify({
                    'error': f'no completed evaluation found for experiment_id={experiment_id}'
                }), 400
            finding.evaluation_id = best_eval.id
            finding.experiment_id = experiment_id
            finding.pt_score = best_eval.pt_score
            evaluation_id = best_eval.id
            pt_score = best_eval.pt_score

        db.session.commit()

        # Write finding to local JSON file so agents can search via Glob/Grep
        try:
            from w2s_research.research_loop.tools.findings_sync import save_finding_to_dir
            from w2s_research.config import LOCAL_FINDINGS_DIR
            path = save_finding_to_dir(finding.to_dict(), Path(LOCAL_FINDINGS_DIR))
            if path:
                print(f"[share_finding] Written to {path}")
        except Exception as file_err:
            print(f"[share_finding] Warning: failed to write finding file: {file_err}")

        return jsonify({
            'message': 'Finding shared successfully',
            'finding_id': finding.id,
            'post_id': finding.post_id,
            'evaluation_id': evaluation_id,
            'pt_score': pt_score,
            'finding': finding.to_dict(),
        })

    except Exception as e:
        import traceback
        print(f"[share_finding] ERROR: {e}")
        traceback.print_exc()
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluations', methods=['POST'])
def post_evaluations():
    """Submit a worker artifact for authoritative evaluation.

    Body: {submission_dir | s3_path, base_model, experiment_id, mini: bool}.
    Spawns a background thread running evaluate_phantom_transfer_submission, returns
    {evaluation_id} immediately with 202.
    """
    from w2s_research.web_ui.backend.models import Evaluation, Experiment, db
    from w2s_research.web_ui.backend import config as backend_config
    import json as _json
    import threading

    data = request.get_json() or {}
    submission_dir = data.get('submission_dir')
    s3_path = data.get('s3_path')
    base_model = data.get('base_model') or 'google/gemma-3-12b-it'
    experiment_id = data.get('experiment_id')
    mini = bool(data.get('mini', False))

    if not experiment_id:
        return jsonify({'error': 'experiment_id required'}), 400
    if not (submission_dir or s3_path):
        return jsonify({'error': 'submission_dir or s3_path required'}), 400

    exp = db.session.get(Experiment, experiment_id)
    if exp is None:
        return jsonify({'error': f'experiment {experiment_id} not found'}), 404

    # Server reads entity lists from its OWN config (spec §7.2 step 5).
    assigned = list(backend_config.PT_ASSIGNED_ENTITIES)
    held_out = list(backend_config.PT_HELD_OUT_ENTITIES) if not mini else []

    ev = Evaluation(
        experiment_id=experiment_id,
        status='queued',
        submission_dir=submission_dir,
        s3_path=s3_path,
        base_model=base_model,
        assigned_entities=_json.dumps(assigned),
        held_out_entities=_json.dumps(held_out),
        mini=mini,
    )
    db.session.add(ev)
    db.session.commit()
    ev_id = ev.id

    def _run_eval():
        from w2s_research.web_ui.backend.evaluation import (
            evaluate_phantom_transfer_submission, compose_pt_score,
        )
        with app.app_context():
            row = db.session.get(Evaluation, ev_id)
            row.status = 'running'
            db.session.commit()
            try:
                result = evaluate_phantom_transfer_submission(
                    submission_dir=submission_dir,
                    base_model=base_model,
                    known_entities=assigned,
                    held_out_entities=held_out,
                    eval_config={'mini': mini, 'work_dir': f'/tmp/eval_{ev_id}'},
                )
                row = db.session.get(Evaluation, ev_id)
                row.pt_transfer_in_distribution = result.get('transfer_in_distribution')
                row.pt_transfer_in_distribution_vs_clean = result.get('transfer_in_distribution_vs_clean')
                row.pt_transfer_generalisation = result.get('transfer_generalisation')
                row.pt_negative_mentions_lift = result.get('negative_mentions_lift')
                row.pt_negative_mentions_lift_vs_clean = result.get('negative_mentions_lift_vs_clean')
                row.pt_capability_delta_pp = result.get('capability_delta_pp')
                row.pt_capability_delta_pp_vs_clean = result.get('capability_delta_pp_vs_clean')
                row.pt_dataset_stealth_auc = result.get('dataset_stealth_auc')
                row.pt_dataset_stealth_auc_vs_clean_pipeline = result.get('dataset_stealth_auc_vs_clean_pipeline')
                row.pt_model_stealth_acc = result.get('model_stealth_acc')
                row.pt_model_stealth_acc_vs_clean = result.get('model_stealth_acc_vs_clean')
                row.pt_negative_mentions_p_vs_base = result.get('negative_mentions_p_vs_base')
                row.pt_negative_mentions_p_vs_clean = result.get('negative_mentions_p_vs_clean')
                row.pt_model_stealth_p_vs_base = result.get('model_stealth_p_vs_base')
                row.pt_model_stealth_p_vs_clean = result.get('model_stealth_p_vs_clean')
                row.pt_dataset_stealth_p_vs_raw = result.get('dataset_stealth_p_vs_raw')
                row.pt_dataset_stealth_p_vs_clean_pipeline = result.get('dataset_stealth_p_vs_clean_pipeline')
                row.pt_clean_control_source = result.get('clean_control_source')
                # Inject held_out_entities into the metrics dict for compose_pt_score.
                result_for_compose = dict(result)
                result_for_compose['held_out_entities'] = held_out
                row.pt_score = compose_pt_score(result_for_compose)
                row.pt_raw_json = _json.dumps(result.get('raw', {}), default=str)
                row.pt_eval_errors = _json.dumps(result.get('errors', []))
                row.status = 'done'
                row.completed_at = db.func.now()
                db.session.commit()
            except Exception as e:
                row = db.session.get(Evaluation, ev_id)
                row.status = 'failed'
                row.pt_eval_errors = _json.dumps([f'background_thread_exception: {e!r}'])
                row.completed_at = db.func.now()
                db.session.commit()

    threading.Thread(target=_run_eval, daemon=True).start()

    return jsonify({'evaluation_id': ev_id, 'status': 'queued'}), 202


@app.route('/api/evaluations/<int:evaluation_id>', methods=['GET'])
def get_evaluation(evaluation_id):
    """Poll an evaluation's status + scores. Held-out info is scrubbed by default."""
    from w2s_research.web_ui.backend.models import Evaluation, db
    row = db.session.get(Evaluation, evaluation_id)
    if row is None:
        return jsonify({'error': 'not_found'}), 404
    # Default scrub_held_out=True is appropriate for worker-facing access.
    return jsonify(row.to_dict(scrub_held_out=True))


@app.route('/api/findings/search', methods=['POST'])
def search_findings_keyword():
    """Keyword search over findings in the database.

    Body: { "query": "...", "limit": 20 }
    Splits the query into keywords and matches against title, content, and idea_name.
    """
    try:
        data = request.get_json()
        if not data or not data.get('query'):
            return jsonify({'error': 'Missing query field'}), 400

        query = data['query'].strip()
        limit = data.get('limit', 20)
        keywords = query.lower().split()

        if not keywords:
            return jsonify({'results': [], 'summary': 'Empty query.'})

        # Build filter: every keyword must appear in title, content, or idea_name
        from sqlalchemy import or_
        filters = []
        for kw in keywords:
            pattern = f'%{kw}%'
            filters.append(or_(
                Finding.title.ilike(pattern),
                Finding.content.ilike(pattern),
                Finding.idea_name.ilike(pattern),
            ))

        results = (Finding.query
                   .filter(*filters)
                   .order_by(Finding.created_at.desc())
                   .limit(limit)
                   .all())

        return jsonify({
            'results': [f.to_dict() for f in results],
            'summary': f'Found {len(results)} result(s) for "{query}".',
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'results': [], 'summary': ''}), 500


@app.route('/api/findings/all', methods=['GET'])
def get_all_findings():
    """Return all findings for client-side sync.

    Workers poll this endpoint to pull down shared findings.
    Returns all findings ordered by creation time (newest first).
    """
    try:
        findings = Finding.query.order_by(Finding.created_at.desc()).all()
        return jsonify({
            'findings': [f.to_dict() for f in findings],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/findings/query', methods=['GET'])
@app.route('/api/findings', methods=['GET'])
def query_findings():
    """Query findings with flexible filtering and sorting.

    Unified endpoint — serves both MCP tool queries and forum UI.

    Query params:
    - query: text search on content (summary), title, and idea_name
    - dataset, finding_type, idea_uid, idea_name: filters
    - worked: true/false filter
    - sort: 'new' (default), 'top', 'discussed'
    - limit: max results (default 50, max 200)
    - offset: pagination offset
    """
    try:
        text_query = request.args.get('query')
        dataset = request.args.get('dataset')
        finding_type = request.args.get('finding_type')
        idea_uid = request.args.get('idea_uid')
        idea_name = request.args.get('idea_name')
        worked_str = request.args.get('worked')
        sort = request.args.get('sort', 'new')

        try:
            limit = min(max(int(request.args.get('limit', 50)), 1), 200)
            offset = max(int(request.args.get('offset', 0)), 0)
        except ValueError:
            return jsonify({'error': 'Invalid limit or offset parameter'}), 400

        query = Finding.query

        if text_query:
            like_pat = f"%{text_query}%"
            query = query.filter(
                db.or_(
                    Finding.content.ilike(like_pat),
                    Finding.title.ilike(like_pat),
                    Finding.idea_name.ilike(like_pat),
                )
            )
        if dataset:
            query = query.filter(Finding.dataset == dataset)
        if idea_name:
            query = query.filter(Finding.idea_name == idea_name)
        if finding_type:
            query = query.filter(Finding.finding_type == finding_type)
        if idea_uid:
            query = query.filter(Finding.idea_uid == idea_uid)
        if worked_str is not None:
            worked_bool = worked_str.lower() in ('true', '1', 'yes')
            query = query.filter(Finding.worked == worked_bool)

        total = query.count()

        if sort == 'top':
            query = query.order_by(
                (Finding.upvotes - Finding.downvotes).desc(),
                Finding.created_at.desc()
            )
        elif sort == 'discussed':
            query = query.order_by(Finding.comment_count.desc(), Finding.created_at.desc())
        else:  # 'new'
            query = query.order_by(Finding.created_at.desc())

        findings = query.offset(offset).limit(limit).all()

        return jsonify({
            'findings': [f.to_dict() for f in findings],
            'total': total,
            'limit': limit,
            'offset': offset,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/findings', methods=['POST'])
def create_finding():
    """Create a new finding directly (used by forum UI).

    Body: title (required), content (required), plus optional context fields.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        title = data.get('title', '').strip()
        content = data.get('content', '').strip()

        if not title:
            return jsonify({'error': 'Title is required'}), 400
        if not content:
            return jsonify({'error': 'Content is required'}), 400

        import uuid
        finding = Finding(
            post_id=str(uuid.uuid4()),
            title=title,
            content=content,
            finding_type=data.get('finding_type'),
            idea_uid=data.get('idea_uid'),
            idea_name=data.get('idea_name'),
            run_id=data.get('run_id'),
            session_id=data.get('session_id'),
            dataset=data.get('dataset'),
            weak_model=data.get('weak_model'),
            strong_model=data.get('strong_model'),
            pgr=data.get('pgr'),
            transfer_acc=data.get('transfer_acc'),
            code_snippet=data.get('code_snippet'),
        )
        db.session.add(finding)
        db.session.commit()

        # Write finding to local JSON file so agents can search via Glob/Grep
        try:
            from w2s_research.research_loop.tools.findings_sync import save_finding_to_dir
            from w2s_research.config import LOCAL_FINDINGS_DIR
            save_finding_to_dir(finding.to_dict(), Path(LOCAL_FINDINGS_DIR))
        except Exception as file_err:
            print(f"[create_finding] Warning: failed to write finding file: {file_err}")

        return jsonify({'post': finding.to_dict()})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/findings/<post_id>', methods=['GET'])
def get_finding(post_id):
    """Get a single finding with its comments."""
    try:
        finding = Finding.query.filter_by(post_id=post_id).first()
        if not finding:
            return jsonify({'error': 'Finding not found'}), 404

        return jsonify({'post': finding.to_dict(include_comments=True)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/findings/<post_id>/vote', methods=['POST'])
def vote_finding(post_id):
    """Vote on a finding. Body: { "vote": "up" | "down" }"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        vote = data.get('vote')
        if vote not in ('up', 'down'):
            return jsonify({'error': 'Vote must be "up" or "down"'}), 400

        finding = Finding.query.filter_by(post_id=post_id).first()
        if not finding:
            return jsonify({'error': 'Finding not found'}), 404

        if vote == 'up':
            finding.upvotes += 1
        else:
            finding.downvotes += 1

        db.session.commit()
        return jsonify({'post': finding.to_dict()})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/findings/<post_id>/comments', methods=['POST'])
def add_finding_comment(post_id):
    """Add a comment to a finding. Body: { "content": "...", "author": "human" }"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        content = data.get('content', '').strip()
        if not content:
            return jsonify({'error': 'Content is required'}), 400

        finding = Finding.query.filter_by(post_id=post_id).first()
        if not finding:
            return jsonify({'error': 'Finding not found'}), 404

        comment = FindingComment(
            finding_id=finding.id,
            content=content,
            author=data.get('author', 'human'),
        )
        db.session.add(comment)
        finding.comment_count += 1
        db.session.commit()

        return jsonify({'comment': comment.to_dict()})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/findings/stats', methods=['GET'])
def get_findings_stats():
    """Get findings statistics."""
    try:
        from sqlalchemy import func

        total_posts = Finding.query.count()
        total_comments = FindingComment.query.count()
        total_upvotes = db.session.query(db.func.sum(Finding.upvotes)).scalar() or 0

        top_sessions = db.session.query(
            Finding.session_id,
            func.count(Finding.id).label('post_count')
        ).filter(
            Finding.session_id.isnot(None)
        ).group_by(
            Finding.session_id
        ).order_by(
            func.count(Finding.id).desc()
        ).limit(10).all()

        return jsonify({
            'total_posts': total_posts,
            'total_comments': total_comments,
            'total_upvotes': total_upvotes,
            'top_agents': [{'session_id': s[0], 'post_count': s[1]} for s in top_sessions],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Snapshot API Endpoints — query Finding rows with commit_id set
# =============================================================================

@app.route('/api/snapshots/search', methods=['GET', 'POST'])
def search_commits():
    """Search for snapshots (findings with commit_id set).

    Accepts query string with optional structured filters like 'idea_uid:xxx run_id:yyy'.
    """
    try:
        if request.method == 'POST':
            data = request.json or {}
            raw_query = data.get('query', '')
            limit = data.get('limit', 100)
        else:
            raw_query = request.args.get('query', '')
            limit = int(request.args.get('limit', 100))

        query = Finding.query.filter(Finding.commit_id.isnot(None))

        # Parse structured filters from query string
        if raw_query:
            for token in raw_query.split():
                if ':' in token:
                    key, val = token.split(':', 1)
                    if key == 'idea_uid':
                        query = query.filter(Finding.idea_uid == val)
                    elif key == 'run_id':
                        query = query.filter(Finding.run_id == val)
                    elif key == 'idea_name':
                        query = query.filter(Finding.idea_name == val)
                    elif key == 'commit_id':
                        query = query.filter(Finding.commit_id == val)
                else:
                    like_pat = f"%{token}%"
                    query = query.filter(
                        db.or_(Finding.title.ilike(like_pat), Finding.idea_name.ilike(like_pat))
                    )

        commits = query.order_by(Finding.created_at.desc()).limit(limit).all()

        return jsonify({
            'commits': [f.to_dict() for f in commits],
            'total': len(commits),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/snapshots/<commit_id>', methods=['GET'])
def get_commit(commit_id):
    """Get a snapshot by commit_id."""
    try:
        finding = Finding.query.filter_by(commit_id=commit_id).first()
        if not finding:
            return jsonify({'error': 'Commit not found'}), 404

        return jsonify(finding.to_dict())

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    from w2s_research.web_ui.backend.models import ensure_schema_current
    with app.app_context():
        ensure_schema_current()

    # Ingest warm-start seed ideas from w2s_research/ideas/<name>/idea.md
    ensure_seed_ideas_exist()

    # Auto-queue any seed idea that doesn't yet have an Experiment record.
    # Concurrency is capped by MAX_CONCURRENT_PODS; extra ideas wait their turn.
    auto_queue_seed_ideas()
    
    # Start the background worker
    worker.start()
    
    try:
        # Run Flask app
        print(f"\n{'=' * 80}")
        print("🚀 W2S Research Web UI")
        print(f"{'=' * 80}")
        print(f"Dataset: {config.DATASET_DIR}")
        print(f"Weak Model: {config.WEAK_MODEL}")
        print(f"Strong Model: {config.STRONG_MODEL}")
        print(f"Seeds: {config.SEEDS}")
        print(f"{'=' * 80}\n")

        # Use port from environment variable, default to 5000
        import os
        port = int(os.environ.get('PORT', 8000))
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

    finally:
        worker.stop()
