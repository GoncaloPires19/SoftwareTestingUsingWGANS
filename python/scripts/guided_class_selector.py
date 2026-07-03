#!/usr/bin/env python3
"""
Guided Target Class Selector (CONSERVATIVE strategy)

Generates per-project class whitelists that the PIT runner uses to filter target
classes. The previous version (whitelist with p >= 0.5) was too aggressive: it
removed classes the predictor wasn't sure about, including ones the baseline
already covered well, causing 30/63 projects to LOSE score.

This version uses a **conservative blacklist + cross-check with observed
coverage**:
  - START from the full set (everything is allowed by default).
  - REMOVE only classes with VERY LOW predicted testability (p < drop_threshold,
    default 0.2) — these are confidently I/O / UI / unreachable.
  - PRESERVE classes that the BASELINE PIT run already covered well
    (coverage >= preserve_threshold, default 0.3), regardless of what the
    predictor says. The data trumps the model.
  - GUARANTEE every project keeps at least `min_classes` classes.

Net effect: surgical removal of confidently-bad classes, never breaks
projects that were already working.

Usage:
    python python/scripts/guided_class_selector.py \
        --structural outputs/mutation_experiments/class_structural_metrics_111.csv \
        --model outputs/mutation_experiments/testability_model_111.pkl \
        --baseline-outcomes outputs/mutation_experiments/class_level_dataset_111.csv \
        --output-dir outputs/mutation_experiments/guided_targets_v2 \
        --drop-threshold 0.2 \
        --preserve-coverage 0.3
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


def _load_baseline_coverage(path: Path) -> dict[tuple[str, str], float]:
    """Map (project, class_simple_name) -> baseline observed coverage."""
    cov: dict[tuple[str, str], float] = {}
    if not path.exists():
        return cov
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            simple = r["class_name"].split(".")[-1].split("$")[0]
            try:
                cov[(r["project"], simple)] = float(r["coverage"])
            except (KeyError, ValueError):
                pass
    return cov


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-project target whitelists using conservative blacklist strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--structural", type=str,
                        default="outputs/mutation_experiments/class_structural_metrics_111.csv")
    parser.add_argument("--model", type=str,
                        default="outputs/mutation_experiments/testability_model_111.pkl")
    parser.add_argument("--baseline-outcomes", type=str,
                        default="outputs/mutation_experiments/class_level_dataset_111.csv",
                        help="Class-level outcomes from baseline PIT run (provides observed coverage)")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/mutation_experiments/guided_targets_v2")
    parser.add_argument("--drop-threshold", type=float, default=0.2,
                        help="Drop classes with p_testable < this (default 0.2 = very confident)")
    parser.add_argument("--preserve-coverage", type=float, default=0.3,
                        help="Preserve any class with baseline coverage >= this, regardless of predictor")
    parser.add_argument("--uncertain-threshold", type=float, default=0.5,
                        help="Classes with p_testable < this AND baseline coverage < uncertain-cov are also dropped")
    parser.add_argument("--uncertain-coverage", type=float, default=0.1,
                        help="Coverage below this combined with low predicted testability triggers an extra drop")
    parser.add_argument("--min-classes-per-project", type=int, default=5,
                        help="Floor: keep at least N classes per project")
    parser.add_argument(
        "--adaptive", action="store_true",
        help=(
            "Adjust drop/uncertain thresholds per project based on project size and "
            "baseline coverage rate. Small projects use softer cuts; large projects use "
            "harder cuts; high-coverage projects use preserve-only mode. "
            "This reduces the 13-project regression seen with global thresholds."
        ),
    )
    args = parser.parse_args()

    structural_path = Path(args.structural)
    model_path = Path(args.model)
    out_dir = Path(args.output_dir)

    for p, label in [(structural_path, "structural"), (model_path, "model")]:
        if not p.exists():
            print(f"[ERRO] {label} not found: {p}")
            sys.exit(1)

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["features"]
    print(f"Modelo carregado: {type(model).__name__} | features={len(features)}")

    baseline_cov = _load_baseline_coverage(Path(args.baseline_outcomes))
    print(f"Coberturas observadas do baseline: {len(baseline_cov)} classes")

    rows = []
    with open(structural_path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"Classes estruturais: {len(rows)} de {len({r['project'] for r in rows})} projetos")

    X = np.array([[float(r[c]) for c in features] for r in rows], dtype=np.float64)
    proba = model.predict_proba(scaler.transform(X))[:, 1]

    by_project: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r, p in zip(rows, proba):
        by_project[r["project"]].append((r["class_name"], float(p)))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    total_kept = 0
    total_classes = 0
    total_dropped_low = 0
    total_dropped_uncertain = 0
    total_preserved_by_cov = 0

    for proj, scored in sorted(by_project.items()):
        total_classes += len(scored)
        kept: list[str] = []
        dropped_low = 0
        dropped_uncertain = 0
        preserved = 0

        # Per-project adaptive thresholds (activated by --adaptive flag)
        if args.adaptive:
            n_cls = len(scored)
            proj_covs = [
                baseline_cov.get((proj, cn.split(".")[-1].split("$")[0]), -1.0)
                for cn, _ in scored
            ]
            mean_baseline_cov = float(np.mean([c for c in proj_covs if c >= 0])) if any(c >= 0 for c in proj_covs) else 0.0

            if n_cls < 15:
                # Small project: very soft cut — barely remove anything
                eff_drop = max(0.08, args.drop_threshold - 0.10)
                eff_uncertain = max(0.30, args.uncertain_threshold - 0.15)
            elif n_cls > 80:
                # Large project: more aggressive — remove clear dead weight
                eff_drop = min(0.40, args.drop_threshold + 0.10)
                eff_uncertain = min(0.65, args.uncertain_threshold + 0.10)
            else:
                eff_drop = args.drop_threshold
                eff_uncertain = args.uncertain_threshold

            # Projects where baseline already covers well: preserve-only mode
            # (only drop very confidently untestable; keep everything else)
            if mean_baseline_cov >= 0.35:
                eff_drop = max(0.05, eff_drop - 0.10)
                eff_uncertain = 1.1  # effectively disable uncertain-drop
        else:
            eff_drop = args.drop_threshold
            eff_uncertain = args.uncertain_threshold

        for class_name, p in scored:
            simple = class_name.split(".")[-1].split("$")[0]
            observed_cov = baseline_cov.get((proj, simple), -1.0)

            if observed_cov >= args.preserve_coverage:
                # Data overrides model: baseline already covers this -> keep
                kept.append(class_name)
                preserved += 1
            elif p < eff_drop:
                # Very confident not testable -> drop
                dropped_low += 1
            elif p < eff_uncertain and 0 <= observed_cov < args.uncertain_coverage:
                # Model uncertain AND baseline also showed no coverage -> drop (noise floor)
                dropped_uncertain += 1
            else:
                # Default: keep (model thinks ok, OR no baseline evidence either way)
                kept.append(class_name)

        # Floor: if we cut too much, restore the top-ranked classes
        if len(kept) < args.min_classes_per_project:
            scored.sort(key=lambda t: -t[1])
            kept = [cn for cn, _ in scored[: max(args.min_classes_per_project, len(kept))]]

        out_file = out_dir / f"{proj}.classes"
        out_file.write_text("\n".join(kept) + "\n", encoding="utf-8")
        total_kept += len(kept)
        total_dropped_low += dropped_low
        total_dropped_uncertain += dropped_uncertain
        total_preserved_by_cov += preserved
        summary.append((proj, len(scored), len(kept), dropped_low + dropped_uncertain, preserved))

    print(f"\n{'Projeto':<28} {'Total':>6} {'Mantidas':>9} {'Cortadas':>9} {'Preservadas':>12}")
    print("-" * 72)
    for proj, total, kept, drop, presv in summary[:15]:
        print(f"{proj:<28} {total:>6} {kept:>9} {drop:>9} {presv:>12}")
    if len(summary) > 15:
        print(f"... mais {len(summary)-15} projetos")

    print(f"\nTotal: {total_kept}/{total_classes} classes mantidas ({total_kept/total_classes*100:.0f}%)")
    print(f"Cortadas confiantes  (p<{args.drop_threshold}):                 {total_dropped_low}")
    print(f"Cortadas incertas    (p<{args.uncertain_threshold} AND cov<{args.uncertain_coverage}):   {total_dropped_uncertain}")
    print(f"Preservadas por cobertura observada >= {args.preserve_coverage}:  {total_preserved_by_cov}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
