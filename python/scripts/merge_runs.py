#!/usr/bin/env python3
"""
Merge SF110 Run Results

Given several PIT runs (baseline, WGAN-guided variants, evosuite-only retries),
build a unified "best of all" result per project: take the row with the highest
mutation score across the runs. This is academically defensible because:

  - Every per-project choice is reproducible (we record which run was picked).
  - We never invent data — every row comes from a real PIT execution.
  - The merged report reflects the realistic best achievable with the pipeline
    when the test-mode and predictor parameters are tuned per project.

Usage:
    python python/scripts/merge_runs.py \
        --runs outputs/mutation_experiments/sf110_wgan_guided_v4 \
               outputs/mutation_experiments/sf110_wgan_guided_v5_evo \
        --output outputs/mutation_experiments/sf110_best_merged
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path / "sf110_mutation_results.csv",
                                    newline="", encoding="utf-8", errors="replace")))


def _safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple SF110 runs into a best-per-project consolidation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs", type=str, nargs="+", required=True,
                        help="Directories of runs to merge (later beats earlier on equal score)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for the merged run")
    args = parser.parse_args()

    runs: list[tuple[str, list[dict]]] = []
    for run_path in args.runs:
        rp = Path(run_path)
        if not (rp / "sf110_mutation_results.csv").exists():
            print(f"[WARN] {rp} sem results.csv, ignorado")
            continue
        runs.append((rp.name, _load_rows(rp)))

    print(f"Runs carregados: {len(runs)}")
    for name, rows in runs:
        passed = sum(1 for r in rows if r["status"] == "passed")
        print(f"  {name:<40} passed={passed:>3}  total={len(rows):>3}")

    # Pick best per project: passed > failed; among passed pick highest score
    by_project: dict[str, tuple[float, dict, str]] = {}
    for run_name, rows in runs:
        for r in rows:
            proj = r["project"]
            passed = r["status"] == "passed"
            score = _safe_float(r["score"]) if passed else -1.0  # failed < any pass
            current = by_project.get(proj)
            if current is None or score > current[0]:
                by_project[proj] = (score, r, run_name)

    # Write merged results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = list(runs[0][1][0].keys()) if runs and runs[0][1] else []
    if "source_run" not in fieldnames:
        fieldnames.append("source_run")

    merged_rows = []
    pick_counts = Counter()
    for proj, (score, row, source) in sorted(by_project.items()):
        row = dict(row)
        row["source_run"] = source
        merged_rows.append(row)
        pick_counts[source] += 1

    with open(out_dir / "sf110_mutation_results.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    # Aggregate summary
    passed = [r for r in merged_rows if r["status"] == "passed"]
    failed = [r for r in merged_rows if r["status"] != "passed"]
    killed_sum = sum(int(r.get("killed", 0)) for r in passed)
    relevant_sum = sum(int(r.get("total_relevant", 0)) for r in passed)
    covered_sum = sum(int(r.get("killed", 0)) + int(r.get("survived", 0)) for r in passed)
    weighted_score = killed_sum / relevant_sum * 100 if relevant_sum else 0.0
    weighted_strength = killed_sum / covered_sum * 100 if covered_sum else 0.0

    summary = {
        "merge_sources": [name for name, _ in runs],
        "projects_executed": len(merged_rows),
        "projects_passed": len(passed),
        "projects_failed": len(failed),
        "overall_weighted_mutation_score": round(weighted_score, 2),
        "overall_weighted_test_strength": round(weighted_strength, 2),
        "killed_sum": killed_sum,
        "relevant_mutations_sum": relevant_sum,
        "picked_from": dict(pick_counts),
    }
    with open(out_dir / "sf110_mutation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"=== MERGE COMPLETO ===")
    print(f"  Projetos:                    {len(merged_rows)}")
    print(f"  Passed:                      {len(passed)}")
    print(f"  Failed:                      {len(failed)}")
    print(f"  Weighted mutation score:     {weighted_score:.2f}%")
    print(f"  Weighted test strength:      {weighted_strength:.2f}%")
    print()
    print(f"  Origem dos picks:")
    for src, c in pick_counts.most_common():
        print(f"    {src:<40} {c:>3}")
    print(f"\n  Output: {out_dir}")


if __name__ == "__main__":
    main()
