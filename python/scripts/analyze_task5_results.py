#!/usr/bin/env python3
"""
Tarefa 5, passo final — "analisar e processar estatisticamente os resultados".

Reads outputs/task5/results.csv (from run_task5_experiment.py) and reports:

  1. Per-arm coverage and defect detection, mean +/- std across seeds.
  2. The GAN against each baseline in the ladder, paired by seed. The ladder
     matters because the arms answer different objections:
        vs RANDOM  did it learn anything beyond the value range?
        vs CORPUS  is it worth more than reusing the inputs it trained on?
        vs KDE     does the problem need a GAN, or would a density model do?
     Beating RANDOM alone is a weak claim; beating CORPUS and KDE is not.
  3. Distributional fidelity (W1) against practical utility. Chapter 3 states
     that the correlation between the two "cannot be assumed a priori" and
     promises the experimental chapter will test it. This is that test.
  4. Optionally, the per-mutator breakdown, to see *which* mutants the GAN
     kills better and whether that agrees with the operator curation done on
     SF110.

Arms are paired by seed (one WGAN training run feeds every generated arm), so
the paired Wilcoxon signed-rank test is the appropriate non-parametric choice.
Testing five metrics per baseline raises the chance of a false positive, so
Holm-adjusted p-values are reported alongside the raw ones.

Usage:
    python python/scripts/analyze_task5_results.py --results outputs/task5/results.csv
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

from scipy.stats import spearmanr, wilcoxon

SEP = "=" * 78
METRICS = [
    ("line_coverage", "Line coverage"),
    ("branch_coverage", "Branch coverage"),
    ("method_coverage", "Method coverage"),
    ("mutation_score", "Mutation score"),
    ("test_strength", "Test strength"),
]
ARM_ORDER = ["MANUAL", "EVOSUITE", "WGAN", "KDE", "CORPUS", "RANDOM"]

# O braco da rede chamou-se GAN nas primeiras execucoes. O nome era impreciso --
# o modelo e uma WGAN-GP e nao uma GAN -- mas os CSV ja recolhidos usam-no, e
# nenhum deles deve ter de ser re-executado por causa de uma etiqueta. Normalizar
# a leitura mantem-nos utilizaveis.
ARM_ALIASES = {"GAN": "WGAN"}


def _norm_arm(arm: str) -> str:
    arm = arm.strip()
    return ARM_ALIASES.get(arm, arm)


BASELINES = [
    ("RANDOM", "amostragem uniforme no mesmo intervalo"),
    ("CORPUS", "reamostragem do corpus real"),
    ("KDE", "densidade classica sobre o mesmo corpus"),
]


def _f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment.

    Chosen over plain Bonferroni because it is uniformly more powerful while
    controlling the same family-wise error rate.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def _head_to_head(by_arm: dict, treat: str, base: str, why: str) -> None:
    a = {r["seed"]: r for r in by_arm[treat]}
    b = {r["seed"]: r for r in by_arm[base]}
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return

    print(f"\n{SEP}")
    print(f"  {treat} vs {base} — {why}")
    print(SEP)

    raw: list[float] = []
    linhas: list[tuple] = []
    for key, label in METRICS:
        # O teste corre sobre os valores tal como estao no CSV (fraccoes com
        # 4 casas). Multiplicar por 100 antes do teste altera o ultimo bit de
        # diferencas iguais, desfaz empates de ordens e muda o p exacto; os
        # outros analisadores (conditional, lowdata) usam fraccoes, e a tese
        # foi calculada assim. A escala percentual e so para apresentacao.
        g0 = np.array([_f(a[s], key) for s in seeds])
        r0 = np.array([_f(b[s], key) for s in seeds])
        g, r = g0 * 100, r0 * 100
        delta = g - r
        if np.all(g0 == r0):
            p = float("nan")
        else:
            try:
                _, p = wilcoxon(g0, r0)
            except ValueError:
                p = float("nan")
        raw.append(p)
        linhas.append((label, g.mean(), r.mean(), delta.mean(), delta.std(),
                       int((delta > 0).sum()), len(delta), p))

    # Uma metrica identica em todos os seeds (p = n/a) nao e um teste: nao
    # entra na familia do Holm. Contar um p = 1 a mais inflacionava os p
    # ajustados (0,0024 em vez de 0,0020 na escada do alvo 1).
    idx = [i for i, p in enumerate(raw) if p == p]
    adj_validos = _holm([raw[i] for i in idx])
    adj = [float("nan")] * len(raw)
    for i, pa in zip(idx, adj_validos):
        adj[i] = pa
    m = len(idx)
    print(f"\n  {'Metrica':<17}{treat:>8}{base:>9}{'Delta':>9}{'a favor':>9}"
          f"{'p':>9}{'p(Holm)':>10}")
    print("  " + "-" * 71)
    for (label, gm, rm, dm, ds, fav, n, p), pa in zip(linhas, adj):
        marca = " *" if p == p and pa < 0.05 else ""
        ptxt = "  n/a" if p != p else f"{p:.4f}"
        patxt = "  n/a" if p != p else f"{pa:.4f}"
        print(f"  {label:<17}{gm:>7.1f}%{rm:>8.1f}%{dm:>+8.1f}pp"
              f"{fav:>6}/{n:<3}{ptxt:>9}{patxt:>10}{marca}")
    print(f"\n  * significativo a 5% depois de corrigir para {m} testes (Holm)")


def _fidelity_vs_utility(rows: list[dict], by_arm: dict) -> None:
    """Does distributional fidelity predict usefulness?"""
    gen = [r for r in rows if r.get("w1") not in (None, "")
           and r["arm"] in {"WGAN", "RANDOM", "CORPUS", "KDE"}]
    if not gen:
        print("\n  (sem coluna w1 — re-corre a experiencia para obter estas metricas)")
        return

    print(f"\n{SEP}")
    print("  FIDELIDADE DISTRIBUCIONAL vs UTILIDADE PRATICA")
    print(SEP)
    print("\n  O Capitulo 3 afirma que a correlacao entre as duas nao pode ser")
    print("  assumida a priori. Esta e a verificacao empirica dessa afirmacao.")

    print(f"\n  Por braco (media sobre seeds):")
    print(f"\n  {'Braco':<9}{'W1':>9}{'unicos':>9}{'d(viz.real)':>13}"
          f"{'mut.score':>12}{'branch':>10}")
    print("  " + "-" * 62)
    for arm in ("WGAN", "KDE", "CORPUS", "RANDOM"):
        if arm not in by_arm:
            continue
        rs = by_arm[arm]
        print(f"  {arm:<9}{np.nanmean([_f(r, 'w1') for r in rs]):>9.4f}"
              f"{np.nanmean([_f(r, 'n_unique') for r in rs]):>9.0f}"
              f"{np.nanmean([_f(r, 'nn_dist_median') for r in rs]):>13.3f}"
              f"{np.nanmean([_f(r, 'mutation_score') for r in rs]) * 100:>11.1f}%"
              f"{np.nanmean([_f(r, 'branch_coverage') for r in rs]) * 100:>9.1f}%")

    w1 = np.array([_f(r, "w1") for r in gen])
    ms = np.array([_f(r, "mutation_score") for r in gen])
    ok = ~(np.isnan(w1) | np.isnan(ms))
    if ok.sum() >= 5 and len(set(w1[ok])) > 1:
        rho, p = spearmanr(w1[ok], ms[ok])
        print(f"\n  Entre bracos (n={int(ok.sum())}): Spearman(W1, mutation score) "
              f"= {rho:+.3f}, p={p:.4f}")
        print("    Nota: dominado pelas diferencas sistematicas entre bracos.")

    if "WGAN" in by_arm and len(by_arm["WGAN"]) >= 5:
        g = by_arm["WGAN"]
        w = np.array([_f(r, "w1") for r in g])
        m = np.array([_f(r, "mutation_score") for r in g])
        ok = ~(np.isnan(w) | np.isnan(m))
        if ok.sum() >= 5 and len(set(w[ok])) > 1:
            rho, p = spearmanr(w[ok], m[ok])
            print(f"\n  Dentro do braco WGAN, entre seeds (n={int(ok.sum())}): "
                  f"Spearman = {rho:+.3f}, p={p:.4f}")
            print("    Esta e a leitura decisiva: uma seed que ajusta melhor a")
            print("    distribuicao produz testes melhores? (W1 menor = melhor ajuste,")
            print("    logo uma relacao real daria correlacao NEGATIVA)")


def _by_mutator(path: Path) -> None:
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    acc: dict[tuple, dict] = defaultdict(lambda: {"killed": 0, "total": 0})
    for r in rows:
        k = int(r["killed"])
        tot = k + int(r["survived"]) + int(r["no_coverage"])
        e = acc[(r["arm"], r["mutator"])]
        e["killed"] += k
        e["total"] += tot

    muts = sorted({m for _, m in acc})
    arms = [a for a in ARM_ORDER if any(x == a for x, _ in acc)]

    print(f"\n{SEP}")
    print("  TAXA DE MORTE POR OPERADOR DE MUTACAO")
    print(SEP)
    print("\n  Onde e que a vantagem se concentra? Liga esta experiencia a")
    print("  curadoria de operadores feita no SF110.")
    print(f"\n  {'Mutator':<26}" + "".join(f"{a:>10}" for a in arms))
    print("  " + "-" * (26 + 10 * len(arms)))
    for m in muts:
        line = f"  {m:<26}"
        for a in arms:
            e = acc.get((a, m))
            line += f"{e['killed'] / e['total']:>9.0%} " if e and e["total"] else f"{'-':>10}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical analysis of the Tarefa 5 experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", type=str, default="outputs/task5/results.csv")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"[ERRO] Results not found: {path}")
        sys.exit(1)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        r["arm"] = _norm_arm(r["arm"])
        by_arm[r["arm"]].append(r)

    print(SEP)
    print("  TAREFA 5 — impacto dos dados de teste gerados pela WGAN-GP")
    print(SEP)
    print(f"\n  Seeds: {len(by_arm.get('WGAN', []))} | "
          f"Arms: {', '.join(a for a in ARM_ORDER if a in by_arm)}")
    pops = sorted({int(float(r["killed"]) + float(r["survived"]) + float(r["no_coverage"]))
                   for rs in by_arm.values() for r in rs})
    print(f"  Mutacoes por medicao: {', '.join(map(str, pops))}"
          + ("  <-- POPULACOES DIFERENTES" if len(pops) > 1 else ""))

    print(f"\n{SEP}")
    print("  RESUMO POR ARM (media +/- desvio sobre seeds)")
    print(SEP)
    header = f"\n  {'Arm':<10}{'Testes':>8}"
    for _, label in METRICS:
        header += f"{label:>18}"
    print(header)
    print("  " + "-" * (18 + 18 * len(METRICS)))
    for arm in ARM_ORDER:
        if arm not in by_arm:
            continue
        arm_rows = by_arm[arm]
        n_tests = int(np.nanmean([_f(r, "n_tests") for r in arm_rows]))
        line = f"  {arm:<10}{n_tests:>8}"
        for key, _ in METRICS:
            vals = np.array([_f(r, key) for r in arm_rows]) * 100
            if len(vals) > 1:
                line += f"{vals.mean():>11.1f}±{vals.std():<6.1f}"
            else:
                line += f"{vals.mean():>17.1f}%"
        print(line)

    for base, why in BASELINES:
        if "WGAN" in by_arm and base in by_arm:
            _head_to_head(by_arm, "WGAN", base, why)

    _fidelity_vs_utility(rows, by_arm)
    _by_mutator(path.with_name(path.stem + "_by_mutator.csv"))

    if "WGAN" in by_arm and "EVOSUITE" in by_arm:
        print(f"\n{SEP}")
        print("  REFERENCIA — GAN vs suites existentes")
        print(SEP)
        print("\n  Nao e comparacao controlada (tamanhos e processos de geracao")
        print("  diferentes); serve para situar a GAN face ao estado atual.")
        for key, label in METRICS:
            g = np.nanmean([_f(r, key) for r in by_arm["WGAN"]]) * 100
            line = f"\n  {label:<18} GAN {g:>6.1f}%"
            for ref in ("EVOSUITE", "MANUAL"):
                if ref in by_arm:
                    v = np.nanmean([_f(r, key) for r in by_arm[ref]]) * 100
                    line += f"   |  {ref} {v:>6.1f}% ({g - v:+.1f} pp)"
            print(line)

    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
