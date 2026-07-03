#!/usr/bin/env python3
"""
Data-Driven Target Class Selector

Uses MEASURED baseline PIT coverage (not a structural predictor) to build
per-project class whitelists. The key insight:

  A class that showed coverage=0 in the baseline run WILL show coverage=0
  in any subsequent run using the same test suite (PIT uses existing tests,
  which cover what they cover regardless of which classes are in the target set).

  Removing such classes from the target set:
    - Does NOT change the killed mutation count (they had 0 kills by definition)
    - DOES reduce the relevant mutation count (denominator of the score)
    => Score goes up for free, with zero risk of losing information.

Decision logic per class:
  1. If measured baseline coverage = 0.0  -> DROP (confirmed zero-coverage)
  2. If measured baseline coverage > 0.0  -> KEEP (at least some mutations executed)
  3. If no baseline data (new class / project failed in baseline) -> FALLBACK:
       use RF predictor if model available, otherwise KEEP (conservative)

This is orthogonal to the predictor-based guided selector:
  - Predictor: learns from structural features which classes are LIKELY testable
  - Data-driven: uses empirical evidence of which classes ARE covered by current tests

Usage:
    python python/scripts/data_driven_selector.py \
        --class-outcomes outputs/mutation_experiments/class_level_dataset_111.csv \
        --structural    outputs/mutation_experiments/class_structural_metrics_111.csv \
        --model         outputs/mutation_experiments/testability_model_111.pkl \
        --output-dir    outputs/mutation_experiments/guided_targets_dd \
        [--min-classes 3]
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FEATURES = [
    "methods", "branches", "statements", "loc", "n_fields", "has_io",
    "extends_framework", "implements_callback", "is_abstract",
    "n_static_methods", "static_ratio", "has_default_ctor",
    "n_imports", "n_external_deps", "max_ctor_params",
]


def _load_baseline_coverage(path: Path) -> dict[tuple[str, str], float]:
    """
    Load per-class coverage from a class_level_dataset CSV.
    Key: (project, simple_class_name) -> baseline coverage fraction.
    """
    cov: dict[tuple[str, str], float] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            proj = r["project"]
            cls = r["class_name"].split(".")[-1].split("$")[0]
            try:
                cov[(proj, cls)] = float(r["coverage"])
            except (KeyError, ValueError):
                pass
    return cov


def _load_baseline_mutations(path: Path) -> dict[tuple[str, str], int]:
    """Per-class mutation count from baseline (for reporting)."""
    muts: dict[tuple[str, str], int] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            proj = r["project"]
            cls = r["class_name"].split(".")[-1].split("$")[0]
            try:
                muts[(proj, cls)] = int(r["n_mutations"])
            except (KeyError, ValueError):
                pass
    return muts


def _load_structural(path: Path, features: list[str]) -> tuple[list[dict], np.ndarray]:
    """Load structural metrics and return (rows, feature_matrix)."""
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    X = np.array([[float(r.get(c, 0)) for c in features] for r in rows], dtype=np.float64)
    return rows, X


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data-driven class selector: drops confirmed zero-coverage classes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--class-outcomes", type=str,
                        default="outputs/mutation_experiments/class_level_dataset_111.csv",
                        help="Class-level PIT outcomes CSV (from baseline run)")
    parser.add_argument("--structural", type=str,
                        default="outputs/mutation_experiments/class_structural_metrics_111.csv",
                        help="Structural metrics CSV (for fallback predictor)")
    parser.add_argument("--model", type=str,
                        default="outputs/mutation_experiments/testability_model_111.pkl",
                        help="RF model pickle (optional; used as fallback for unseen classes)")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/mutation_experiments/guided_targets_dd")
    parser.add_argument("--min-classes", type=int, default=3,
                        help="Keep at least N classes per project (safety floor)")
    parser.add_argument("--fallback-threshold", type=float, default=0.20,
                        help="RF probability threshold below which unseen classes are dropped (fallback only)")
    args = parser.parse_args()

    outcomes_path = Path(args.class_outcomes)
    structural_path = Path(args.structural)
    model_path = Path(args.model)
    out_dir = Path(args.output_dir)

    if not outcomes_path.exists():
        print(f"[ERRO] class-outcomes not found: {outcomes_path}")
        sys.exit(1)

    print(f"A carregar cobertura medida: {outcomes_path.name}")
    baseline_cov = _load_baseline_coverage(outcomes_path)
    baseline_muts = _load_baseline_mutations(outcomes_path)
    print(f"  Classes com dados de cobertura: {len(baseline_cov)}")
    print(f"  Projetos: {len(set(p for p, _ in baseline_cov))}")

    # Load RF model as fallback for classes not in baseline data
    model = scaler = None
    if model_path.exists():
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        scaler = bundle["scaler"]
        print(f"Modelo de fallback carregado: {type(model).__name__}")
    else:
        print(f"[AVISO] Modelo nao encontrado ({model_path}). Fallback: KEEP todas as classes desconhecidas.")

    # Load structural metrics (all projects / all classes)
    proba_map: dict[tuple[str, str], float] = {}
    struct_rows: list[dict] = []
    if structural_path.exists() and model is not None:
        struct_rows, X_struct = _load_structural(structural_path, FEATURES)
        probas = model.predict_proba(scaler.transform(X_struct))[:, 1]
        for row, p in zip(struct_rows, probas):
            proj = row["project"]
            cls = row["class_name"].split(".")[-1].split("$")[0]
            proba_map[(proj, cls)] = float(p)
        print(f"Probabilidades do fallback: {len(proba_map)} classes")
    elif structural_path.exists():
        struct_rows, _ = _load_structural(structural_path, FEATURES)

    # Build per-project class lists from structural metrics (full class universe)
    by_project: dict[str, list[str]] = defaultdict(list)
    if struct_rows:
        for r in struct_rows:
            by_project[r["project"]].append(r["class_name"])
    else:
        # Fall back to deriving projects from baseline data
        for (proj, cls) in baseline_cov:
            by_project[proj].append(cls)

    out_dir.mkdir(parents=True, exist_ok=True)

    total_kept = total_dropped_dd = total_dropped_fallback = total_fallback_kept = 0
    total_removed_muts = 0
    summary = []

    for proj, all_classes in sorted(by_project.items()):
        kept: list[str] = []
        dropped_dd = 0    # dropped by data-driven (coverage=0 measured)
        dropped_fb = 0    # dropped by fallback predictor (no baseline data, low pred)
        fb_kept = 0       # kept by fallback (no baseline data, pred ok)
        removed_muts = 0

        for class_name in all_classes:
            simple = class_name.split(".")[-1].split("$")[0]
            key = (proj, simple)
            cov = baseline_cov.get(key, None)

            if cov is not None:
                # Have baseline evidence
                if cov == 0.0:
                    # Confirmed zero-coverage: safe to drop
                    dropped_dd += 1
                    removed_muts += baseline_muts.get(key, 0)
                else:
                    # Has coverage > 0: definitely keep
                    kept.append(class_name)
            else:
                # No baseline data: use predictor as fallback
                p = proba_map.get(key, None)
                if p is not None and p < args.fallback_threshold:
                    dropped_fb += 1
                else:
                    kept.append(class_name)
                    fb_kept += 1

        # Safety floor: never drop below min_classes.
        # EXCEPTION: if all remaining classes have coverage=0, don't force more
        # zero-coverage classes in — the project is genuinely uncoverable and
        # adding zero-coverage classes only inflates the denominator for free.
        has_covered = any(
            baseline_cov.get((proj, cn.split(".")[-1].split("$")[0]), -1.0) > 0.0
            for cn in kept
        )
        if len(kept) < args.min_classes and len(all_classes) >= args.min_classes and has_covered:
            # Only restore classes that at least have some baseline coverage (> 0)
            candidates = [
                (baseline_cov.get((proj, cn.split(".")[-1].split("$")[0]), -1.0), cn)
                for cn in all_classes if cn not in kept
            ]
            # Prefer coverage > 0 restores; never restore coverage = 0 for the floor
            candidates_positive = [(c, cn) for c, cn in candidates if c > 0.0]
            candidates_positive.sort(key=lambda t: -t[0])
            for _, cn in candidates_positive[:args.min_classes - len(kept)]:
                kept.append(cn)
                dropped_dd = max(0, dropped_dd - 1)

        # Write .classes file
        out_file = out_dir / f"{proj}.classes"
        out_file.write_text("\n".join(sorted(kept)) + "\n", encoding="utf-8")

        total_kept += len(kept)
        total_dropped_dd += dropped_dd
        total_dropped_fallback += dropped_fb
        total_fallback_kept += fb_kept
        total_removed_muts += removed_muts
        summary.append((proj, len(all_classes), len(kept), dropped_dd, dropped_fb, removed_muts))

    print(f"\n{'Projeto':<28} {'Total':>6} {'Kept':>6} {'Drop-DD':>8} {'Drop-FB':>8} {'Muts-remov':>11}")
    print("-" * 72)
    shown = 0
    for proj, total, kept, ddd, dfb, rm in sorted(summary, key=lambda t: -t[5])[:20]:
        if ddd > 0 or dfb > 0:
            print(f"{proj:<28} {total:>6} {kept:>6} {ddd:>8} {dfb:>8} {rm:>11}")
            shown += 1

    print(f"\nTotal: {total_kept} classes mantidas / {total_kept + total_dropped_dd + total_dropped_fallback} total")
    print(f"  Removidas por DD (coverage=0): {total_dropped_dd} classes ({total_removed_muts} mutacoes)")
    print(f"  Removidas por fallback (pred<{args.fallback_threshold}): {total_dropped_fallback} classes")
    print(f"  Fallback kept (sem dados, pred ok): {total_fallback_kept} classes")
    print(f"\nOutput: {out_dir}")
    print(f"\nProjecao de impacto (apenas projetos ja no merged_final com source=baseline):")
    print(f"  Mutacoes totais removidas do denominador (baseline data): {total_removed_muts}")
    print(f"  [Nota: projecao exacta requer cruzar com merged_final. Correr PIT para resultados reais.]")


if __name__ == "__main__":
    main()
