#!/usr/bin/env python3
"""
Compare Baseline vs WGAN-Guided SF110 Mutation Testing Results

Side-by-side per-project and aggregate comparison between the two runs.
Outputs a Markdown report and (when matplotlib is available) a bar chart of the
top winners and losers, ready to embed in the PDF.

Usage:
    python python/scripts/compare_baseline_vs_guided.py \
        --baseline outputs/mutation_experiments/sf110_full111 \
        --guided   outputs/mutation_experiments/sf110_wgan_guided \
        --output   outputs/mutation_experiments/comparison_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_results(run_dir: Path) -> tuple[list[dict], dict]:
    csv_path = run_dir / "sf110_mutation_results.csv"
    summary_path = run_dir / "sf110_mutation_summary.json"
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    summary = {}
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    return rows, summary


def _index_by_project(rows: list[dict]) -> dict[str, dict]:
    return {r["project"]: r for r in rows}


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _make_report(
    baseline_rows: list[dict], guided_rows: list[dict],
    baseline_sum: dict, guided_sum: dict,
) -> str:
    base_by = _index_by_project(baseline_rows)
    guided_by = _index_by_project(guided_rows)

    common = sorted(set(base_by) & set(guided_by))
    both_passed = [p for p in common if base_by[p]["status"] == "passed" and guided_by[p]["status"] == "passed"]

    # Per-project deltas
    deltas = []
    for p in both_passed:
        b = _safe_float(base_by[p]["score"])
        g = _safe_float(guided_by[p]["score"])
        deltas.append((p, b, g, g - b))
    deltas.sort(key=lambda t: -t[3])

    base_avg = sum(d[1] for d in deltas) / len(deltas) if deltas else 0.0
    guided_avg = sum(d[2] for d in deltas) / len(deltas) if deltas else 0.0
    ups = [d for d in deltas if d[3] > 0.5]
    downs = [d for d in deltas if d[3] < -0.5]
    big_wins = [d for d in deltas if d[3] >= 10.0]
    crossed_70 = [d for d in deltas if d[1] < 70.0 <= d[2]]

    base_pass = sum(1 for r in baseline_rows if r["status"] == "passed")
    guided_pass = sum(1 for r in guided_rows if r["status"] == "passed")

    md = []
    md.append("# Comparacao: Baseline vs WGAN-Guiado (SF110 - 111 projetos)\n")
    md.append("## Resumo agregado\n")
    md.append("| Metrica | Baseline | WGAN-Guiado | Delta |")
    md.append("|---|---|---|---|")
    md.append(f"| Projetos executados | {len(baseline_rows)} | {len(guided_rows)} | -- |")
    md.append(f"| Projetos com sucesso | {base_pass} | {guided_pass} | {guided_pass - base_pass:+d} |")
    md.append(f"| Weighted mutation score | {baseline_sum.get('overall_weighted_mutation_score', 0):.2f}% | "
              f"{guided_sum.get('overall_weighted_mutation_score', 0):.2f}% | "
              f"{guided_sum.get('overall_weighted_mutation_score', 0) - baseline_sum.get('overall_weighted_mutation_score', 0):+.2f}pp |")
    md.append(f"| Weighted test strength | {baseline_sum.get('overall_weighted_test_strength', 0):.2f}% | "
              f"{guided_sum.get('overall_weighted_test_strength', 0):.2f}% | "
              f"{guided_sum.get('overall_weighted_test_strength', 0) - baseline_sum.get('overall_weighted_test_strength', 0):+.2f}pp |")
    md.append("")

    md.append("## Comparacao em projetos comuns que passaram nos dois runs\n")
    md.append(f"- Projetos comparaveis: **{len(deltas)}**")
    md.append(f"- Score medio (sem ponderacao): baseline = {base_avg:.2f}%, WGAN-guiado = {guided_avg:.2f}%, "
              f"**delta = {guided_avg - base_avg:+.2f}pp**")
    md.append(f"- Projetos que subiram (> 0.5pp): **{len(ups)}**")
    md.append(f"- Projetos que desceram (< -0.5pp): **{len(downs)}**")
    md.append(f"- Ganhos grandes (>= +10pp): **{len(big_wins)}**")
    md.append(f"- Projetos que cruzaram a barreira dos 70%: **{len(crossed_70)}**")
    md.append("")

    md.append("## Top 15 vencedores\n")
    md.append("| Projeto | Baseline | WGAN-guiado | Ganho |")
    md.append("|---|---|---|---|")
    for proj, b, g, d in deltas[:15]:
        md.append(f"| `{proj}` | {b:.1f}% | **{g:.1f}%** | **{d:+.1f}pp** |")
    md.append("")

    md.append("## Bottom 10 (onde a WGAN cortou demasiado)\n")
    md.append("| Projeto | Baseline | WGAN-guiado | Delta |")
    md.append("|---|---|---|---|")
    for proj, b, g, d in deltas[-10:][::-1]:
        md.append(f"| `{proj}` | {b:.1f}% | {g:.1f}% | {d:+.1f}pp |")
    md.append("")

    if crossed_70:
        md.append("## Projetos que passaram de <70% para >=70% com WGAN-guiado\n")
        md.append("| Projeto | Baseline | WGAN-guiado |")
        md.append("|---|---|---|")
        for proj, b, g, _ in crossed_70:
            md.append(f"| `{proj}` | {b:.1f}% | **{g:.1f}%** |")
        md.append("")

    # Projects above 70 in either run
    over70_base = sorted([d for d in deltas if d[1] >= 70.0], key=lambda t: -t[1])
    over70_guided = sorted([d for d in deltas if d[2] >= 70.0], key=lambda t: -t[2])
    md.append("## Projetos com score >= 70%\n")
    md.append(f"- Baseline: **{len(over70_base)} projetos**: " +
              ", ".join(f"`{d[0]}` ({d[1]:.1f}%)" for d in over70_base[:8]) +
              (f" ... +{len(over70_base)-8} mais" if len(over70_base) > 8 else ""))
    md.append(f"- WGAN-guiado: **{len(over70_guided)} projetos**: " +
              ", ".join(f"`{d[0]}` ({d[2]:.1f}%)" for d in over70_guided[:8]) +
              (f" ... +{len(over70_guided)-8} mais" if len(over70_guided) > 8 else ""))
    md.append("")

    return "\n".join(md), {
        "deltas": deltas,
        "ups": ups, "downs": downs, "big_wins": big_wins,
        "crossed_70": crossed_70, "over70_base": over70_base, "over70_guided": over70_guided,
        "base_avg": base_avg, "guided_avg": guided_avg,
        "base_pass": base_pass, "guided_pass": guided_pass,
    }


def _make_chart(deltas: list[tuple], output_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nao instalado -- a saltar grafico")
        return

    if not deltas:
        return

    # Top 12 winners + bottom 8 losers
    top = deltas[:12]
    bottom = deltas[-8:]
    selected = top + bottom

    projects = [d[0] for d in selected]
    baseline = [d[1] for d in selected]
    guided = [d[2] for d in selected]

    fig, ax = plt.subplots(figsize=(12, 8))
    y = range(len(projects))
    h = 0.4

    ax.barh([i - h/2 for i in y], baseline, h, label="Baseline", color="#888")
    ax.barh([i + h/2 for i in y], guided, h, label="WGAN-guiado", color="#3a7")

    ax.set_yticks(list(y))
    ax.set_yticklabels(projects, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mutation score (%)")
    ax.set_title("Baseline vs WGAN-guiado: top 12 vencedores + bottom 8")
    ax.axvline(70, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Alvo 70%")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=140)
    plt.close()
    print(f"Grafico salvo em: {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs WGAN-guided SF110 mutation testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline", type=str, default="outputs/mutation_experiments/sf110_full111")
    parser.add_argument("--guided",   type=str, default="outputs/mutation_experiments/sf110_wgan_guided")
    parser.add_argument("--output",   type=str, default="outputs/mutation_experiments/comparison_report.md")
    parser.add_argument("--chart",    type=str, default="outputs/mutation_experiments/comparison_chart.png")
    args = parser.parse_args()

    baseline_rows, baseline_sum = _load_results(Path(args.baseline))
    guided_rows, guided_sum = _load_results(Path(args.guided))

    print(f"Baseline:    {len(baseline_rows)} projetos")
    print(f"WGAN-guiado: {len(guided_rows)} projetos")

    report, summary = _make_report(baseline_rows, guided_rows, baseline_sum, guided_sum)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Relatorio salvo em: {args.output}")

    _make_chart(summary["deltas"], Path(args.chart))

    print(f"\n=== Resumo rapido ===")
    print(f"  Projetos comparaveis: {len(summary['deltas'])}")
    print(f"  Score medio: {summary['base_avg']:.2f}% -> {summary['guided_avg']:.2f}% ({summary['guided_avg']-summary['base_avg']:+.2f}pp)")
    print(f"  Subiram: {len(summary['ups'])}  Desceram: {len(summary['downs'])}  Ganhos>10pp: {len(summary['big_wins'])}")
    print(f"  Cruzaram 70%: {len(summary['crossed_70'])}")
    print(f"  Projetos >=70%: baseline {len(summary['over70_base'])}, WGAN {len(summary['over70_guided'])}")


if __name__ == "__main__":
    main()
