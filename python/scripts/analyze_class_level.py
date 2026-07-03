#!/usr/bin/env python3
"""
Class-Level Mutation Analysis & Dual-Metric Reporting

Walks the per-project PIT mutations.csv files of a results run and computes
metrics at the *class* level rather than the project level. This separates two
fundamentally different questions:

  1. GLOBAL score   — mutation score over ALL mutated classes, including ones
                      that tests can't reach (network/IO/UI). Dragged down by
                      no_coverage.
  2. COVERED score  — mutation score restricted to classes the tests actually
                      exercise (coverage >= threshold). Measures genuine test
                      quality. This is the metric that legitimately reaches >70%.

It also exports a class-level dataset (one row per mutated class) that becomes
the bridge dataset linking a class's structural profile to its testability —
used to train the WGAN-guided target selection.

Usage:
    python python/scripts/analyze_class_level.py --results-dir outputs/mutation_experiments/sf110_improved
    python python/scripts/analyze_class_level.py --results-dir <dir> --coverage-threshold 0.5 --export class_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP = "=" * 72

# PIT mutations.csv column layout: fileName, className, mutator, method, line, status, killingTest
_COL_FILE = 0
_COL_CLASS = 1
_COL_LINE = 4
_COL_STATUS = 5

_KILLED_STATUSES = {"KILLED", "TIMED_OUT"}
_NONVIABLE_STATUSES = {"NON_VIABLE", "MEMORY_ERROR", "RUN_ERROR"}


@dataclass
class ClassStats:
    """Mutation outcome aggregation for a single class."""
    project: str
    class_name: str
    killed: int = 0
    survived: int = 0
    no_coverage: int = 0
    non_viable: int = 0
    lines: set[int] = field(default_factory=set)

    @property
    def total(self) -> int:
        return self.killed + self.survived + self.no_coverage + self.non_viable

    @property
    def relevant(self) -> int:
        """Mutations that count toward the score (exclude non-viable)."""
        return self.killed + self.survived + self.no_coverage

    @property
    def covered(self) -> int:
        """Mutations on executed code (killed + survived)."""
        return self.killed + self.survived

    @property
    def coverage(self) -> float:
        """Fraction of relevant mutations that tests reach."""
        return self.covered / self.relevant if self.relevant else 0.0

    @property
    def mutation_score(self) -> float:
        """Killed / relevant — the global per-class score (includes no_coverage)."""
        return self.killed / self.relevant if self.relevant else 0.0

    @property
    def test_strength(self) -> float:
        """Killed / covered — score on executed code only."""
        return self.killed / self.covered if self.covered else 0.0


def _parse_mutations_csv(csv_path: Path, project: str) -> dict[str, ClassStats]:
    classes: dict[str, ClassStats] = {}
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.reader(f):
            if len(row) <= _COL_STATUS:
                continue
            class_name = row[_COL_CLASS].strip()
            status = row[_COL_STATUS].strip().upper()
            if not class_name:
                continue

            cs = classes.get(class_name)
            if cs is None:
                cs = ClassStats(project=project, class_name=class_name)
                classes[class_name] = cs

            if status in _KILLED_STATUSES:
                cs.killed += 1
            elif status == "SURVIVED":
                cs.survived += 1
            elif status == "NO_COVERAGE":
                cs.no_coverage += 1
            elif status in _NONVIABLE_STATUSES:
                cs.non_viable += 1
            else:
                # Unknown status — treat as survived (conservative)
                cs.survived += 1

            try:
                cs.lines.add(int(row[_COL_LINE]))
            except (ValueError, IndexError):
                pass
    return classes


def _collect_all_classes(results_dir: Path) -> list[ClassStats]:
    workdir = results_dir / "workdir"
    if not workdir.exists():
        raise FileNotFoundError(f"No workdir/ in {results_dir} — cannot do class-level analysis")

    all_classes: list[ClassStats] = []
    for proj_dir in sorted(workdir.iterdir()):
        if not proj_dir.is_dir():
            continue
        mutations_files = list(proj_dir.rglob("mutations.csv"))
        if not mutations_files:
            continue
        # Use the most recent mutations.csv for this project
        mc = max(mutations_files, key=lambda p: p.stat().st_mtime)
        classes = _parse_mutations_csv(mc, proj_dir.name)
        all_classes.extend(classes.values())
    return all_classes


def _weighted(values: list[tuple[int, int]]) -> float:
    """Given list of (numerator, denominator), return weighted ratio as %."""
    num = sum(n for n, _ in values)
    den = sum(d for _, d in values)
    return num / den * 100 if den else 0.0


def _report(all_classes: list[ClassStats], coverage_threshold: float) -> None:
    print(SEP)
    print("  ANALISE AO NIVEL DA CLASSE — DUAL METRIC REPORTING")
    print(SEP)

    if not all_classes:
        print("  Nenhuma classe encontrada.")
        return

    covered_classes = [c for c in all_classes if c.coverage >= coverage_threshold and c.relevant > 0]
    uncovered_classes = [c for c in all_classes if c.coverage < coverage_threshold and c.relevant > 0]

    # ---- Global (all classes) ----
    global_score = _weighted([(c.killed, c.relevant) for c in all_classes])
    # ---- Covered classes only ----
    covered_score = _weighted([(c.killed, c.relevant) for c in covered_classes])
    covered_strength = _weighted([(c.killed, c.covered) for c in covered_classes])

    n_total = len([c for c in all_classes if c.relevant > 0])
    n_covered = len(covered_classes)

    print(f"\n  Classes mutadas (relevantes): {n_total}")
    print(f"  Limiar de cobertura: {coverage_threshold:.0%}")
    print(f"  Classes COBERTAS (coverage >= {coverage_threshold:.0%}): {n_covered} ({n_covered/n_total*100:.1f}%)")
    print(f"  Classes NAO-COBERTAS: {len(uncovered_classes)} ({len(uncovered_classes)/n_total*100:.1f}%)")

    print(f"\n  {'METRICA':<42} {'VALOR':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Score GLOBAL (todas as classes)':<42} {global_score:>9.2f}%")
    print(f"  {'Score em classes COBERTAS':<42} {covered_score:>9.2f}%")
    print(f"  {'Test strength em classes cobertas':<42} {covered_strength:>9.2f}%")

    # Per-project averages
    by_project_global: dict[str, list[ClassStats]] = defaultdict(list)
    by_project_covered: dict[str, list[ClassStats]] = defaultdict(list)
    for c in all_classes:
        if c.relevant > 0:
            by_project_global[c.project].append(c)
    for c in covered_classes:
        by_project_covered[c.project].append(c)

    proj_global_scores = [
        _weighted([(c.killed, c.relevant) for c in classes])
        for classes in by_project_global.values()
    ]
    proj_covered_scores = [
        _weighted([(c.killed, c.relevant) for c in classes])
        for classes in by_project_covered.values() if classes
    ]

    avg_global = sum(proj_global_scores) / len(proj_global_scores) if proj_global_scores else 0
    avg_covered = sum(proj_covered_scores) / len(proj_covered_scores) if proj_covered_scores else 0

    print(f"\n  {'MEDIA POR-PROJETO':<42} {'VALOR':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Media global por-projeto':<42} {avg_global:>9.2f}%")
    print(f"  {'Media (classes cobertas) por-projeto':<42} {avg_covered:>9.2f}%  <- alvo >70%")

    # Distribution of class coverage
    print(f"\n  Distribuicao de cobertura das classes:")
    bands = [(0, 0.1, "0-10%"), (0.1, 0.5, "10-50%"), (0.5, 0.9, "50-90%"), (0.9, 1.01, "90-100%")]
    relevant_classes = [c for c in all_classes if c.relevant > 0]
    for lo, hi, label in bands:
        count = sum(1 for c in relevant_classes if lo <= c.coverage < hi)
        bar = "#" * (count * 50 // max(len(relevant_classes), 1))
        print(f"    {label:>10}: {bar} {count}")


def _export_dataset(all_classes: list[ClassStats], export_path: Path, coverage_threshold: float) -> None:
    """Export class-level dataset for the WGAN bridge."""
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "project", "class_name", "n_mutations", "n_lines",
            "killed", "survived", "no_coverage",
            "coverage", "mutation_score", "test_strength", "is_covered",
        ])
        for c in all_classes:
            if c.relevant == 0:
                continue
            writer.writerow([
                c.project, c.class_name, c.relevant, len(c.lines),
                c.killed, c.survived, c.no_coverage,
                round(c.coverage, 4), round(c.mutation_score, 4),
                round(c.test_strength, 4), int(c.coverage >= coverage_threshold),
            ])
    print(f"\n  Dataset de classe exportado: {export_path}")
    print(f"  ({sum(1 for c in all_classes if c.relevant > 0)} classes)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Class-level mutation analysis with dual-metric reporting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=str,
        default="outputs/mutation_experiments/sf110_improved",
        help="Results directory containing workdir/ with PIT mutations.csv files",
    )
    parser.add_argument(
        "--coverage-threshold", type=float, default=0.5,
        help="A class is 'covered' when coverage >= this fraction (default: 0.5)",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export class-level dataset CSV to this path (for WGAN bridge)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERRO] Directory not found: {results_dir}")
        sys.exit(1)

    all_classes = _collect_all_classes(results_dir)
    _report(all_classes, args.coverage_threshold)

    if args.export:
        _export_dataset(all_classes, Path(args.export), args.coverage_threshold)


if __name__ == "__main__":
    main()
