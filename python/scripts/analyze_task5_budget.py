#!/usr/bin/env python3
"""
Tarefa 5 — leitura do varrimento de orçamento (run_task5_budget_sweep.py).

The single-budget experiment compares generators at 200 inputs. This reads the
curves instead, which is where the arms separate for a reason rather than by
accident:

  - CORPUS cannot exceed the 121 distinct inputs it resamples from, so its
    curve must flatten. The `n_distinct` column shows exactly where.
  - WGAN and KDE draw from continuous densities and keep producing novel
    inputs, so their curves can keep climbing past that ceiling.

The crossover budget, if there is one, is the honest answer to "is a generative
model worth its cost here?". Below it, keeping the corpus is cheaper and just as
good; above it, the corpus has run out of material and the model has not.

Usage:
    python python/scripts/analyze_task5_budget.py --results outputs/task5/budget_sweep.csv
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

SEP = "=" * 78
ARM_ORDER = ["GAN", "KDE", "CORPUS", "RANDOM"]
METRICS = [
    ("branch_coverage", "Cobertura de ramo"),
    ("line_coverage", "Cobertura de linha"),
    ("mutation_score", "Mutation score"),
]


def _f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read the Tarefa 5 budget sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", type=str, default="outputs/task5/budget_sweep.csv")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"[ERRO] Results not found: {path}")
        sys.exit(1)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("[ERRO] ficheiro vazio")
        sys.exit(1)

    cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cell[(r["arm"], int(r["budget"]))].append(r)

    budgets = sorted({b for _, b in cell})
    arms = [a for a in ARM_ORDER if any(x == a for x, _ in cell)]
    n_seeds = max(len(v) for v in cell.values())

    print(SEP)
    print("  TAREFA 5 — cobertura e detecao vs ORCAMENTO de testes")
    print(SEP)
    print(f"\n  Orcamentos: {budgets}")
    print(f"  Bracos    : {', '.join(arms)}   ({n_seeds} seeds por ponto)")

    # ---- diversidade efectiva: onde e que o CORPUS satura ----
    print(f"\n{SEP}")
    print("  INPUTS DISTINTOS EFECTIVAMENTE EMITIDOS")
    print(SEP)
    print("\n  Um gerador continuo cresce com o orcamento; a reamostragem do")
    print("  corpus satura na diversidade do proprio corpus.")
    print(f"\n  {'Orcamento':>10}" + "".join(f"{a:>10}" for a in arms))
    print("  " + "-" * (10 + 10 * len(arms)))
    for b in budgets:
        line = f"  {b:>10}"
        for a in arms:
            vals = [_f(r, "n_distinct") for r in cell.get((a, b), [])]
            line += f"{np.nanmean(vals):>10.0f}" if vals else f"{'-':>10}"
        print(line)

    # ---- curvas por metrica ----
    for key, label in METRICS:
        print(f"\n{SEP}")
        print(f"  {label.upper()} vs ORCAMENTO (media sobre seeds)")
        print(SEP)
        print(f"\n  {'Orcamento':>10}" + "".join(f"{a:>10}" for a in arms))
        print("  " + "-" * (10 + 10 * len(arms)))
        curva: dict[str, list[float]] = {a: [] for a in arms}
        for b in budgets:
            line = f"  {b:>10}"
            for a in arms:
                vals = [_f(r, key) for r in cell.get((a, b), [])]
                m = float(np.nanmean(vals)) if vals else float("nan")
                curva[a].append(m)
                line += f"{m:>9.1%} " if m == m else f"{'-':>10}"
            print(line)

        if "GAN" in curva and "CORPUS" in curva:
            _crossover(budgets, curva["GAN"], curva["CORPUS"], label)

    print(f"\n{SEP}")


def _crossover(budgets: list[int], gan: list[float], corpus: list[float],
               label: str) -> None:
    """Report where (and whether) the generative arm overtakes the corpus."""
    delta = [g - c for g, c in zip(gan, corpus)]
    acima = [b for b, d in zip(budgets, delta) if d == d and d > 0]
    print(f"\n  Delta GAN - CORPUS por orcamento:")
    print("    " + "  ".join(f"{b}:{d:+.1%}" for b, d in zip(budgets, delta) if d == d))
    if not acima:
        print(f"    A WGAN-GP nunca ultrapassa o CORPUS em {label.lower()} nos")
        print("    orcamentos testados. Se as curvas ainda convergirem, vale a pena")
        print("    estender o varrimento; se ja estiverem paralelas, a leitura")
        print("    honesta e que a reamostragem do corpus basta neste alvo.")
    else:
        print(f"    A WGAN-GP ultrapassa o CORPUS a partir de orcamento {min(acima)}.")


if __name__ == "__main__":
    main()
