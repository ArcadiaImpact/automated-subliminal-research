"""MT-Bench question loader.

Single canonical source (lm-sys/FastChat). On first call, downloads the 80-question
JSONL to a local cache; subsequent calls read from cache. No manual "drop in the
file" step.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Any

MT_BENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)
DEFAULT_CACHE_PATH = Path(__file__).parent / "mt_bench.jsonl"


def load_mt_bench(cache_path: Path | str | None = None) -> List[Dict[str, Any]]:
    """Return the 80 MT-Bench questions as a list of dicts.

    Schema (per row): {"question_id": int, "category": str, "turns": [str, ...]}
    Only `turns[0]` is used by the model-stealth audit (single-turn).

    Args:
        cache_path: where to read/write the cached JSONL. Defaults to a file
            next to this module. If the file is absent, it is fetched from
            MT_BENCH_URL and written. If present, used directly.
    """
    path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[mt_bench_loader] cache miss; fetching {MT_BENCH_URL} -> {path}")
        with urllib.request.urlopen(MT_BENCH_URL, timeout=30) as resp:
            raw = resp.read()
        # Validate JSONL structurally before writing — fail loudly on bad downloads.
        rows: List[Dict[str, Any]] = []
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "turns" not in row or not row["turns"]:
                raise ValueError(f"MT-Bench row missing 'turns': {row}")
            rows.append(row)
        if len(rows) < 50:
            raise ValueError(
                f"MT-Bench download returned only {len(rows)} rows; expected 80. "
                f"URL may have changed: {MT_BENCH_URL}"
            )
        path.write_bytes(raw)
        return rows

    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    qs = load_mt_bench()
    print(f"Loaded {len(qs)} MT-Bench questions.")
    print(f"Categories: {sorted({q.get('category') for q in qs})}")
    print(f"First question (turn 1): {qs[0]['turns'][0][:120]}...")
