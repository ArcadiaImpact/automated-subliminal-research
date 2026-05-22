# Ideas

Two things live here:

## `TEMPLATE/`

The runnable scaffold a worker copies and edits to implement a new poisoning protocol. Contains `run.py` with the `poison_dataset(clean_jsonl_path, entity, out_path, seed) -> Path` contract the worker must implement, plus the artifact-submission driver.

## Seed-idea folders (e.g. `<seed_name>/idea.md`)

Warm-start research directions. Each is a single `idea.md` file inside a folder named after the idea:

```
w2s_research/ideas/
└── my_seed_name/
    └── idea.md
```

**Convention:**

- Folder name (e.g. `my_seed_name`) becomes the `Idea.name` (and is what shows up in the dashboard and worker logs).
- File contents become the `Idea.description` — and are rendered into the worker's prompt as `target_idea_content` (see `research_loop/prompt.jinja2`).
- Keep them open-ended — these are *seeds*, not specs. The worker should be free to take the direction in any reasonable direction.

**Auto-ingestion:** server startup calls `ensure_seed_ideas_exist()` (`web_ui/backend/app.py`), which scans this directory for `<name>/idea.md` and upserts each into the `Idea` table as a queueable `is_baseline=False, source='seed'` row. If you edit an `idea.md` and restart the server, the description is refreshed in place.

Excluded from the scan: `TEMPLATE/`, dot-folders, `__pycache__`.
