#!/usr/bin/env python3
"""
Baseline-Wins Analysis

Identifies projects where the unguided baseline outperforms the WGAN-guided run
and profiles their structural characteristics. This closes the "regression gap":
13 of 80 projects in the merged final use the baseline because guidance hurts them.

Two questions answered:
  1. Which structural features distinguish "guidance helps" from "guidance hurts"?
  2. Is there a learnable signal to skip guidance for certain project profiles?

Output:
  - Summary table: baseline-wins vs guided-wins vs tied
  - Feature distribution comparison (mean ± std for 15 structural features)
  - Recommendations for threshold adaptation

Usage:
    python python/scripts/analyze_baseline_wins.py \
        --baseline outputs/mutation_experiments/sf110_full111/sf110_mutation_results.csv \
        --guided  outputs/mutation_experiments/sf110_best_merged_final/sf110_mutation_results.csv \
        --bridge  outputs/mutation_experiments/bridge_dataset_111.csv \
        [--threshold 1.0] \
        [--export outputs/mutation_experiments/baseline_wins_analysis.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP = "=" * 72

FEATURES = [
    "methods", "branches", "statements", "loc", "n_fields", "has_io",
    "extends_framework", "implements_callback", "is_abstract",
    "n_static_methods", "static_ratio", "has_default_ctor",
    "n_imports", "n_external_deps", "max_ctor_params",
]


def _load_results(path: Path) -> dict[str, dict]:
    """Load per-project mutation results CSV. Returns dict keyed by project."""
    results = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            proj = row["project"]
            results[proj] = {
                "score": float(row["score"]) if row["score"] else 0.0,
                "strength": float(row["test_strength"]) if row["test_strength"] else 0.0,
                "status": row.get("status", ""),
                "killed": int(row.get("killed", 0) or 0),
                "survived": int(row.get("survived", 0) or 0),
                "no_coverage": int(row.get("no_coverage", 0) or 0),
                "total_relevant": int(row.get("total_relevant", 0) or 0),
                "source_run": row.get("source_run", "baseline"),
            }
    return results


def _load_bridge(path: Path) -> dict[str, list[dict]]:
    """Load bridge dataset. Returns dict keyed by project with list of class rows."""
    by_proj: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            by_proj[row["project"]].append(row)
    return by_proj


def _project_feature_mean(class_rows: list[dict]) -> dict[str, float]:
    """Compute mean value of each feature across a project's classes."""
    if not class_rows:
        return {f: 0.0 for f in FEATURES}
    result = {}
    for feat in FEATURES:
        vals = []
        for r in class_rows:
            try:
                vals.append(float(r[feat]))
            except (KeyError, ValueError):
                pass
        result[feat] = float(np.mean(vals)) if vals else 0.0
    result["n_classes"] = float(len(class_rows))
    result["pct_testable"] = float(np.mean([int(r.get("testable", 0)) for r in class_rows]))
    result["mean_coverage"] = float(np.mean([float(r.get("coverage", 0)) for r in class_rows]))
    result["mean_score"] = float(np.mean([float(r.get("mutation_score", 0)) for r in class_rows]))
    return result


def _compare_groups(
    wins: list[str],
    loses: list[str],
    bridge: dict[str, list[dict]],
    label_wins: str,
    label_loses: str,
) -> None:
    """Print feature distribution comparison between two project groups."""
    feats_extended = FEATURES + ["n_classes", "pct_testable", "mean_coverage", "mean_score"]

    win_profiles = [_project_feature_mean(bridge.get(p, [])) for p in wins]
    lose_profiles = [_project_feature_mean(bridge.get(p, [])) for p in loses]

    if not win_profiles or not lose_profiles:
        print("  Dados insuficientes para comparacao de features.")
        return

    print(f"\n  {'Feature':<22} {label_wins[:16]:>16} {label_loses[:16]:>16} {'Delta':>10} {'Impacto':>8}")
    print(f"  {'-'*74}")

    differences = []
    for feat in feats_extended:
        vals_w = [p[feat] for p in win_profiles if feat in p]
        vals_l = [p[feat] for p in lose_profiles if feat in p]
        if not vals_w or not vals_l:
            continue
        mean_w = np.mean(vals_w)
        mean_l = np.mean(vals_l)
        std_w = np.std(vals_w)
        std_l = np.std(vals_l)
        delta = mean_w - mean_l
        # Effect size: Cohen's d (pooled std)
        pooled_std = np.sqrt((std_w ** 2 + std_l ** 2) / 2) if (std_w + std_l) > 0 else 1.0
        effect = abs(delta / pooled_std) if pooled_std > 0 else 0.0
        differences.append((feat, mean_w, mean_l, delta, effect))

    # Sort by absolute effect size
    differences.sort(key=lambda t: -t[4])
    for feat, mw, ml, d, eff in differences:
        impact = "***" if eff > 0.8 else ("**" if eff > 0.5 else ("*" if eff > 0.2 else ""))
        print(f"  {feat:<22} {mw:>16.3f} {ml:>16.3f} {d:>+10.3f} {impact:>8}")

    print(f"\n  Impacto: *** = grande (d>0.8), ** = medio (d>0.5), * = pequeno (d>0.2)")


def _print_project_table(projects: list[tuple], label: str) -> None:
    """Print sorted table of (project, baseline_score, guided_score, delta) tuples."""
    print(f"\n  {label}:")
    print(f"  {'Projeto':<28} {'Baseline':>10} {'Guiado':>10} {'Delta':>8}")
    print(f"  {'-'*58}")
    for proj, bs, gs, delta in sorted(projects, key=lambda t: t[3]):
        print(f"  {proj:<28} {bs:>9.1f}% {gs:>9.1f}% {delta:>+7.1f}pp")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse projects where baseline beats WGAN-guided run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline", type=str,
        default="outputs/mutation_experiments/sf110_full111/sf110_mutation_results.csv",
        help="Baseline (unguided) mutation results CSV",
    )
    parser.add_argument(
        "--guided", type=str,
        default="outputs/mutation_experiments/sf110_best_merged_final/sf110_mutation_results.csv",
        help="Best guided (or merged) mutation results CSV",
    )
    parser.add_argument(
        "--bridge", type=str,
        default="outputs/mutation_experiments/bridge_dataset_111.csv",
        help="Bridge dataset with structural features per class",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.0,
        help="Min score difference (pp) for baseline to be counted as 'win' (default 1.0pp)",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export comparison table to CSV",
    )
    args = parser.parse_args()

    for label, p in [("baseline", args.baseline), ("guided", args.guided), ("bridge", args.bridge)]:
        if not Path(p).exists():
            print(f"[ERRO] {label} nao encontrado: {p}")
            sys.exit(1)

    baseline = _load_results(Path(args.baseline))
    guided = _load_results(Path(args.guided))
    bridge = _load_bridge(Path(args.bridge))

    print(SEP)
    print("  ANALISE BASELINE-WINS — Quando a guidance prejudica")
    print(SEP)

    # Compare only projects present in both
    common = set(baseline) & set(guided)
    only_baseline = set(baseline) - set(guided)
    only_guided = set(guided) - set(baseline)

    print(f"\n  Projetos no baseline: {len(baseline)}")
    print(f"  Projetos no guided:   {len(guided)}")
    print(f"  Comuns (comparaveis): {len(common)}")
    print(f"  So no baseline (falhou no guided): {len(only_baseline)}")
    print(f"  Recuperados no guided (falhou no baseline): {len(only_guided)}")

    baseline_wins: list[tuple] = []
    guided_wins: list[tuple] = []
    tied: list[tuple] = []
    failed_both: list[str] = []

    for proj in sorted(common):
        bs = baseline[proj]
        gs = guided[proj]
        b_score = bs["score"] if bs["status"] == "passed" else 0.0
        g_score = gs["score"] if gs["status"] == "passed" else 0.0
        delta = b_score - g_score

        if bs["status"] != "passed" and gs["status"] != "passed":
            failed_both.append(proj)
            continue

        entry = (proj, b_score, g_score, delta)
        if delta > args.threshold:
            baseline_wins.append(entry)
        elif delta < -args.threshold:
            guided_wins.append(entry)
        else:
            tied.append(entry)

    print(f"\n  (threshold de diferenca: {args.threshold:.1f}pp)")
    print(f"\n  {'Categoria':<30} {'N':>4} {'Delta medio':>12}")
    print(f"  {'-'*48}")
    if baseline_wins:
        bd = np.mean([t[3] for t in baseline_wins])
        print(f"  {'Baseline vence':<30} {len(baseline_wins):>4} {bd:>+11.1f}pp")
    if guided_wins:
        gd = np.mean([t[3] for t in guided_wins])
        print(f"  {'Guidance vence':<30} {len(guided_wins):>4} {gd:>+11.1f}pp")
    if tied:
        print(f"  {'Empatados (<= {:.0f}pp)':<30} {len(tied):>4}".format(args.threshold))
    if failed_both:
        print(f"  {'Falhos em ambos':<30} {len(failed_both):>4}")

    # Gained by guided (projects only in guided)
    print(f"\n  Projetos RECUPERADOS pelo guided (nao passavam no baseline): {len(only_guided)}")
    if only_guided:
        recovered_scores = [guided[p]["score"] for p in only_guided if guided[p]["status"] == "passed"]
        if recovered_scores:
            print(f"  Score medio nos recuperados: {np.mean(recovered_scores):.1f}%")

    # Detail tables
    if baseline_wins:
        _print_project_table(baseline_wins, "Projetos onde BASELINE ganha (guidance prejudica)")

    if baseline_wins and guided_wins:
        print(f"\n{SEP}")
        print("  COMPARACAO ESTRUTURAL: projetos baseline-wins vs guidance-wins")
        print(SEP)
        bw_names = [t[0] for t in baseline_wins]
        gw_names = [t[0] for t in guided_wins]
        _compare_groups(bw_names, gw_names, bridge, "Baseline-wins", "Guidance-wins")

        print(f"\n{SEP}")
        print("  INTERPRETACAO DOS RESULTADOS")
        print(SEP)
        print("""
  Features onde baseline-wins > guidance-wins (features que aumentam a
  probabilidade de guidance prejudicar):

  - Projetos pequenos (n_classes baixo): o corte remove demasiadas classes
    e PIT nao tem alvos suficientes. Threshold adaptativo recomendado.

  - pct_testable alto: projectos onde o predictor ja acha tudo testavel ->
    o corte nao remove nada util mas pode remover classes cobertas por azar.

  - mean_coverage alto no baseline: projectos onde os testes EvoSuite ja
    cobrem bem -> a guidance nao tem onde ganhar e pode cortar boas classes.

  Recomendacao: usar --adaptive no guided_class_selector.py para ajustar
  os thresholds com base no perfil do projecto (ver flag --adaptive).
        """)

    # Export comparison table
    if args.export:
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["project", "baseline_score", "guided_score", "delta", "winner"])
            for proj, bs, gs, delta in sorted(
                baseline_wins + guided_wins + tied, key=lambda t: t[0]
            ):
                winner = "baseline" if delta > args.threshold else ("guided" if delta < -args.threshold else "tied")
                writer.writerow([proj, round(bs, 2), round(gs, 2), round(delta, 2), winner])
            for proj in failed_both:
                writer.writerow([proj, 0, 0, 0, "failed_both"])
        print(f"\n  Exportado: {out}")


if __name__ == "__main__":
    main()
