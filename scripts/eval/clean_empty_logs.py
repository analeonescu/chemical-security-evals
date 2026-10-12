"""
Removes log folders left behind by runs that never produced a sample.

A killed eval still writes a log file, which stays at status "started" with no
samples. Those folders hold no data but consume a folder number, so the
numbering runs ahead of the number of real results.

Only folders whose logs ALL lack samples are removed; a folder with any real
sample is left alone, even if it also holds an abandoned log.

Usage:
    python scripts/eval/clean_empty_logs.py --dry-run
    python scripts/eval/clean_empty_logs.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_ai.log import read_eval_log  # noqa: E402

PAIR_INDEX_NAME = ".pair_index.json"


def folder_is_empty(folder: Path) -> bool:
    """True if the folder holds logs and none of them recorded a sample."""
    logs = list(folder.glob("*.eval"))
    if not logs:
        return False
    for log_path in logs:
        try:
            if read_eval_log(log_path).samples:
                return False
        except Exception:
            # Unreadable: treat as real data rather than risk deleting it.
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--logs", type=Path, default=ROOT / "logs", help="Log root")
    parser.add_argument("--dry-run", action="store_true", help="List without deleting")
    args = parser.parse_args()

    if not args.logs.is_dir():
        print(f"No log directory at {args.logs}")
        return 0

    empty, kept = [], 0
    for folder in sorted(args.logs.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        if folder_is_empty(folder):
            empty.append(folder)
        else:
            kept += 1

    print(f"folders with data: {kept} | folders with no samples: {len(empty)}")
    if empty:
        names = [f.name for f in empty]
        print(f"  {', '.join(names[:12])}{' ...' if len(names) > 12 else ''}")

    if args.dry_run:
        print("(dry run, nothing removed)")
        return 0

    for folder in empty:
        shutil.rmtree(folder)

    # The pair index caches folder names, so drop it after changing them.
    cache = args.logs / PAIR_INDEX_NAME
    if cache.exists():
        cache.unlink()
        print(f"removed stale {cache.name}")

    print(f"removed {len(empty)} empty folders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
