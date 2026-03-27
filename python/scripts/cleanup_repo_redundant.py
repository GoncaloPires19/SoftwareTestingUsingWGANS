#!/usr/bin/env python3
"""Delete redundant artifacts while preserving reproducible core outputs.

Default mode is dry-run. Use --apply to perform deletion.
"""

from __future__ import annotations

import argparse
from pathlib import Path


KEEP_PREFIXES = {
    "outputs": [
        "combined_h2_",
        "h2_v2_optimized_",
        "experimental",
        "sf110_hypotheses",
        "archive",
        "FINAL_RESULTS_TABLE.md",
    ],
    "processed": [
        "sf110_full_v2_strict.csv",
        "sf110_full_v2_H2_sf110_plus_junit_strict.csv",
        "sf110_full_v2_report.json",
    ],
}


def should_keep_outputs(path: Path, outputs_root: Path) -> bool:
    rel = str(path.relative_to(outputs_root)).replace("\\", "/")
    for prefix in KEEP_PREFIXES["outputs"]:
        if rel == prefix or rel.startswith(prefix + "/") or rel.startswith(prefix):
            return True
    return False


def should_keep_processed(path: Path) -> bool:
    return path.name in KEEP_PREFIXES["processed"]


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def delete_dir(path: Path) -> None:
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            p.rmdir()
    path.rmdir()


def remove_path(path: Path, apply: bool) -> int:
    if not path.exists():
        return 0

    size = path_size(path)
    if not apply:
        return size

    if path.is_file():
        path.unlink()
    elif path.is_dir():
        delete_dir(path)
    return size


def collect_output_candidates(outputs_root: Path) -> list[Path]:
    if not outputs_root.exists():
        return []
    return [p for p in outputs_root.iterdir() if not should_keep_outputs(p, outputs_root)]


def collect_processed_candidates(processed_root: Path) -> list[Path]:
    if not processed_root.exists():
        return []
    return [p for p in processed_root.iterdir() if p.is_file() and not should_keep_processed(p)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete redundant outputs/data artifacts")
    parser.add_argument("--apply", action="store_true", help="Actually delete files")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    outputs_root = repo_root / "outputs"
    processed_root = repo_root / "python" / "data" / "processed"

    candidates = collect_output_candidates(outputs_root)
    candidates.extend(collect_processed_candidates(processed_root))

    total = 0
    for c in candidates:
        total += remove_path(c, apply=args.apply)
        status = "DELETE" if args.apply else "DRYRUN"
        print(f"[{status}] {c}")

    mb = total / (1024 * 1024)
    print("=" * 80)
    print(f"Mode: {'apply' if args.apply else 'dry-run'}")
    print(f"Candidates: {len(candidates)}")
    print(f"Estimated size freed: {mb:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
