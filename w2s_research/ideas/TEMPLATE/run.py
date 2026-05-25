"""Template for phantom-transfer poisoning ideas (Shape C)."""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from w2s_research.core import RunConfig, create_run_arg_parser, set_seed


def poison_dataset(clean_jsonl_path, entity, out_path, seed=42):
    """IMPLEMENT THIS — entity-agnostic poisoning. See docstring contract above."""
    raise NotImplementedError(
        "Implement your entity-agnostic poisoning protocol here."
    )


def clean_pipeline_dataset(clean_jsonl_path, out_path, seed=42):
    """OPTIONAL: clean dataset produced under the same pipeline (no poison payload)."""
    raise NotImplementedError("Optional; see contract.")


def run_experiment(config: RunConfig):
    set_seed(config.seed)
    workspace = Path(config.data_dir).parent
    clean = Path(config.data_dir) / "clean.jsonl"
    outbox = workspace / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)

    assigned = os.environ.get("PT_ASSIGNED_ENTITIES", "uk,reagan,stalin").split(",")
    for entity in assigned:
        poison_dataset(clean_jsonl_path=clean, entity=entity,
                       out_path=outbox / f"poisoned_{entity}.jsonl", seed=config.seed)
    # Run local mini self-eval
    print("[driver] running local mini self-eval...")
    subprocess.run([
        sys.executable, "-m", "w2s_research.web_ui.backend.evaluation",
        "--mini",
        "--submission-dir", str(outbox),
        "--known-entities", ",".join(assigned),
    ], check=False)
    # Agent then calls submit_for_evaluation MCP tool from its loop; not invoked here.


if __name__ == "__main__":
    parser = create_run_arg_parser(description="Phantom-transfer poisoning template")
    args = parser.parse_args()
    config = RunConfig.from_args(args)
    run_experiment(config)
