#!/usr/bin/env python3
"""
Bridge Dataset Builder

Joins two halves into the dataset that links a class's STRUCTURE to its TESTABILITY:

  structural metrics (methods, branches, statements, loc, n_fields, has_io)
        +  -->  bridge dataset  -->  testability predictor
  PIT outcomes (coverage, mutation_score) per class

The resulting dataset has one row per class that was both:
  - found in the SF110 source tree (structural metrics available), and
  - mutated by PIT (testability outcome observed).

The label `testable` = 1 when the class achieved coverage >= threshold, i.e. the
tests genuinely exercised it. The predictor trained on this can flag, from
structure alone (no PIT run needed), which classes are worth targeting.

Usage:
    python python/scripts/build_bridge_dataset.py \
        --structural outputs/mutation_experiments/class_structural_metrics.csv \
        --outcomes   outputs/mutation_experiments/class_level_dataset.csv \
        --output     outputs/mutation_experiments/bridge_dataset.csv \
        --coverage-threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the structure->testability bridge dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--structural", type=str,
        default="outputs/mutation_experiments/class_structural_metrics.csv",
    )
    parser.add_argument(
        "--outcomes", type=str,
        default="outputs/mutation_experiments/class_level_dataset.csv",
    )
    parser.add_argument(
        "--output", type=str,
        default="outputs/mutation_experiments/bridge_dataset.csv",
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    args = parser.parse_args()

    structural = _load_csv(Path(args.structural))
    outcomes = _load_csv(Path(args.outcomes))

    print(f"Structural metrics: {len(structural)} classes")
    print(f"PIT outcomes:       {len(outcomes)} classes")

    # Index structural metrics by (project, fully-qualified class name)
    struct_index = {(r["project"], r["class_name"]): r for r in structural}

    # Also index by simple class name within project as fallback (PIT sometimes
    # reports inner classes or differing package resolution)
    struct_by_simple: dict[tuple[str, str], dict] = {}
    for r in structural:
        simple = r["class_name"].split(".")[-1]
        struct_by_simple.setdefault((r["project"], simple), r)

    joined: list[dict] = []
    unmatched = 0
    for o in outcomes:
        key = (o["project"], o["class_name"])
        s = struct_index.get(key)
        if s is None:
            simple = o["class_name"].split(".")[-1].split("$")[0]
            s = struct_by_simple.get((o["project"], simple))
        if s is None:
            unmatched += 1
            continue

        coverage = float(o["coverage"])
        # Optional v2 features — tolerate older structural CSVs without them
        def sg(key, default=0):
            return type(default)(s.get(key, default)) if s.get(key, "") != "" else default

        joined.append({
            "project": o["project"],
            "class_name": o["class_name"],
            # structural features (inputs)
            "methods": int(s["methods"]),
            "branches": int(s["branches"]),
            "statements": int(s["statements"]),
            "loc": int(s["loc"]),
            "n_fields": int(s["n_fields"]),
            "has_io": int(s["has_io"]),
            # v2 testability features
            "extends_framework": sg("extends_framework", 0),
            "implements_callback": sg("implements_callback", 0),
            "is_abstract": sg("is_abstract", 0),
            "n_static_methods": sg("n_static_methods", 0),
            "static_ratio": sg("static_ratio", 0.0),
            "has_default_ctor": sg("has_default_ctor", 0),
            "n_imports": sg("n_imports", 0),
            "n_external_deps": sg("n_external_deps", 0),
            "max_ctor_params": sg("max_ctor_params", 0),
            "n_mutations": int(o["n_mutations"]),
            # testability outcomes (labels)
            "coverage": round(coverage, 4),
            "mutation_score": float(o["mutation_score"]),
            "test_strength": float(o["test_strength"]),
            "testable": int(coverage >= args.coverage_threshold),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "project", "class_name",
        "methods", "branches", "statements", "loc", "n_fields", "has_io",
        "extends_framework", "implements_callback", "is_abstract",
        "n_static_methods", "static_ratio", "has_default_ctor",
        "n_imports", "n_external_deps", "max_ctor_params", "n_mutations",
        "coverage", "mutation_score", "test_strength", "testable",
    ]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(joined)

    n_testable = sum(r["testable"] for r in joined)
    print(f"\nBridge dataset: {len(joined)} classes joined ({unmatched} unmatched, sem metricas estruturais)")
    print(f"  Testaveis (coverage >= {args.coverage_threshold:.0%}): {n_testable} ({n_testable/len(joined)*100:.0f}%)")
    print(f"  Nao-testaveis: {len(joined)-n_testable} ({(len(joined)-n_testable)/len(joined)*100:.0f}%)")
    print(f"  Output: {output}")

    # Quick feature contrast: testable vs non-testable structural profiles
    if joined:
        def _avg(rows, field):
            return sum(r[field] for r in rows) / len(rows) if rows else 0
        testable = [r for r in joined if r["testable"]]
        non_testable = [r for r in joined if not r["testable"]]
        print(f"\n  Perfil estrutural medio:")
        print(f"  {'Feature':<20} {'Testavel':>10} {'Nao-test.':>10}")
        print(f"  {'-'*42}")
        for feat in ["methods", "branches", "statements", "loc", "n_fields", "has_io",
                     "extends_framework", "implements_callback", "is_abstract",
                     "static_ratio", "has_default_ctor", "n_external_deps", "max_ctor_params"]:
            print(f"  {feat:<20} {_avg(testable, feat):>10.2f} {_avg(non_testable, feat):>10.2f}")


if __name__ == "__main__":
    main()
