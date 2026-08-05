#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for the Baseline -> v8_curated Mutation Score Gain

Resamples projects with replacement (paired, same resample index used for both
runs) to build a 95% CI on the weighted mutation score and weighted test
strength, for the baseline, for v8_curated, and for the delta between them.
Only projects that PASSED in both runs are used, so the comparison is
apples-to-apples per project.

Usage:
    python python/scripts/bootstrap_score_ci.py \
        --baseline outputs/mutation_experiments/sf110_full111/sf110_mutation_results.csv \
        --guided outputs/mutation_experiments/sf110_v8_curated/sf110_mutation_results.csv \
        --n-boot 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEP = "=" * 72


def _weighted(killed: np.ndarray, relevant: np.ndarray) -> float:
    rel_sum = relevant.sum()
    return float(killed.sum() / rel_sum * 100) if rel_sum else 0.0


def _bootstrap(killed: np.ndarray, survived: np.ndarray, relevant: np.ndarray, n_boot: int, seed: int):
    """Returns arrays of bootstrap weighted-score and weighted-strength."""
    rng = np.random.default_rng(seed)
    n = len(killed)
    scores = np.empty(n_boot)
    strengths = np.empty(n_boot)
    covered = killed + survived
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[b] = _weighted(killed[idx], relevant[idx])
        strengths[b] = _weighted(killed[idx], covered[idx])
    return scores, strengths, idx  # noqa: idx not needed after loop, kept for debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI for mutation score gain")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--guided", type=str, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/mutation_experiments/bootstrap_score_ci.json")
    args = parser.parse_args()

    base_df = pd.read_csv(args.baseline)
    guided_df = pd.read_csv(args.guided)

    base_passed = base_df[base_df.status == "passed"].set_index("project")
    guided_passed = guided_df[guided_df.status == "passed"].set_index("project")

    common = sorted(set(base_passed.index) & set(guided_passed.index))
    print(SEP)
    print("  BOOTSTRAP CI — baseline vs v8_curated (projetos comuns, resample pareado)")
    print(SEP)
    print(f"\n  Projetos comuns (passed em ambos): {len(common)}")

    b = base_passed.loc[common]
    g = guided_passed.loc[common]

    b_killed = b["killed"].to_numpy(dtype=float)
    b_survived = b["survived"].to_numpy(dtype=float)
    b_relevant = b["total_relevant"].to_numpy(dtype=float)

    g_killed = g["killed"].to_numpy(dtype=float)
    g_survived = g["survived"].to_numpy(dtype=float)
    g_relevant = g["total_relevant"].to_numpy(dtype=float)

    n = len(common)
    rng = np.random.default_rng(args.seed)
    n_boot = args.n_boot

    base_scores = np.empty(n_boot)
    base_strengths = np.empty(n_boot)
    guided_scores = np.empty(n_boot)
    guided_strengths = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        base_scores[i] = _weighted(b_killed[idx], b_relevant[idx])
        base_strengths[i] = _weighted(b_killed[idx], b_survived[idx] + b_killed[idx])
        guided_scores[i] = _weighted(g_killed[idx], g_relevant[idx])
        guided_strengths[i] = _weighted(g_killed[idx], g_survived[idx] + g_killed[idx])

    delta_scores = guided_scores - base_scores
    delta_strengths = guided_strengths - base_strengths

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    point_base_score = _weighted(b_killed, b_relevant)
    point_guided_score = _weighted(g_killed, g_relevant)
    point_base_strength = _weighted(b_killed, b_survived + b_killed)
    point_guided_strength = _weighted(g_killed, g_survived + g_killed)

    print(f"\n  {'Métrica':<28} {'Ponto':>8} {'IC 95%':>20}")
    print(f"  {'-'*58}")
    for label, point, arr in [
        ("Score baseline", point_base_score, base_scores),
        ("Score v8_curated", point_guided_score, guided_scores),
        ("Delta score", point_guided_score - point_base_score, delta_scores),
        ("Strength baseline", point_base_strength, base_strengths),
        ("Strength v8_curated", point_guided_strength, guided_strengths),
        ("Delta strength", point_guided_strength - point_base_strength, delta_strengths),
    ]:
        lo, hi = ci(arr)
        print(f"  {label:<28} {point:>7.2f}% [{lo:6.2f}%, {hi:6.2f}%]")

    excludes_zero_score = not (ci(delta_scores)[0] <= 0 <= ci(delta_scores)[1])
    excludes_zero_strength = not (ci(delta_strengths)[0] <= 0 <= ci(delta_strengths)[1])
    print(f"\n  Delta score exclui 0 no IC 95%?    {excludes_zero_score}")
    print(f"  Delta strength exclui 0 no IC 95%? {excludes_zero_strength}")

    results = {
        "n_projects": n,
        "n_boot": n_boot,
        "score_baseline": {"point": point_base_score, "ci95": ci(base_scores)},
        "score_v8_curated": {"point": point_guided_score, "ci95": ci(guided_scores)},
        "delta_score": {"point": point_guided_score - point_base_score, "ci95": ci(delta_scores), "excludes_zero": excludes_zero_score},
        "strength_baseline": {"point": point_base_strength, "ci95": ci(base_strengths)},
        "strength_v8_curated": {"point": point_guided_strength, "ci95": ci(guided_strengths)},
        "delta_strength": {"point": point_guided_strength - point_base_strength, "ci95": ci(delta_strengths), "excludes_zero": excludes_zero_strength},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados guardados: {out_path}")


if __name__ == "__main__":
    main()
