#!/usr/bin/env python3
"""Tarefa 5 — ablacao do condicionamento em arr_len.

Compara duas execucoes da escada que diferem numa unica coisa: se a WGAN-GP foi
condicionada na dimensao arr_len. Tudo o resto -- corpus, orcamento, epocas,
seeds, oraculo, PIT -- e identico, e os bracos RANDOM, CORPUS e KDE sao gerados
a partir de fluxos de RNG que a bandeira nao toca.

Isso torna-os um controlo de regressao com dentes. Se algum deles se mover entre
as duas execucoes, a diferenca observada na WGAN-GP nao e atribuivel ao
condicionamento: alguma coisa no pipeline partilhado mudou, e a leitura da
ablacao fica invalida. O script verifica isso antes de reportar seja o que for.

A comparacao e emparelhada por seed, que e o unico desenho com potencia
suficiente aqui: o desvio-padrao entre seeds (~2,6 pp) e varias vezes maior do
que o efeito que se procura, mas o ruido comum a ambos os bracos cancela-se
dentro de cada par.

Uso:
    python python/scripts/analyze_task5_conditional.py \
        --baseline outputs/task5/results_e2000.csv \
        --conditional outputs/task5/results_e2000_cond.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP = "=" * 76
CONTROL_ARMS = ["RANDOM", "CORPUS", "KDE"]

# Ver analyze_task5_results.py: o braco chamou-se GAN nas primeiras execucoes.
ARM_ALIASES = {"GAN": "WGAN"}
# Abaixo disto o braco de controlo mudou o suficiente para a ablacao deixar de
# isolar o condicionamento. 0,5 pp e uma ordem de grandeza abaixo do desvio
# entre seeds e bem abaixo do efeito procurado.
CONTROL_TOL_PP = 0.5


def _load(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """{arm: {seed: mutation_score}} mais {seed: total de mutantes}.

    O total e recolhido porque o mutation_score e killed/total, e ja aconteceu
    duas execucoes desta escada correrem contra populacoes de mutantes
    diferentes (114 e 118) por causa de classes compiladas em cache. As medias
    resultantes somam denominadores distintos e a diferenca entre elas nao
    significa nada. Ver _assert_homogeneous.
    """
    out: dict[str, dict[str, float]] = {}
    totals: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seed = (row.get("seed") or "").strip()
            if not seed or seed == "-":
                continue
            try:
                score = float(row["mutation_score"])
            except (KeyError, TypeError, ValueError):
                continue
            arm = row["arm"].strip()
            out.setdefault(ARM_ALIASES.get(arm, arm), {})[seed] = score
            try:
                totals[seed] = (int(row["killed"]) + int(row["survived"])
                                + int(row["no_coverage"]))
            except (KeyError, TypeError, ValueError):
                pass
    return out, totals


def _assert_homogeneous(totals: dict[str, dict[str, int]]) -> set[str]:
    """Devolve os seeds medidos contra a mesma populacao de mutantes em ambas.

    Comparar um seed medido contra 114 mutantes com o mesmo seed medido contra
    118 nao mede o tratamento, mede a recompilacao. Em vez de avisar e seguir em
    frente, este filtro remove esses seeds da analise: um aviso e facil de nao
    ler, e o numero errado ja chegou a ser reportado uma vez.
    """
    seen = sorted({t for d in totals.values() for t in d.values()})
    if len(seen) > 1:
        print(f"\n  [AVISO] populacoes de mutantes diferentes nos dados: {seen}")
        print("  Provavel causa: classes compiladas em cache numa das execucoes.")
        print("  Os seeds afectados sao excluidos -- o mutation_score e killed/total,")
        print("  por isso denominadores distintos nao sao comparaveis.")

    common = set(totals["baseline"]) & set(totals["conditional"])
    return {s for s in common if totals["baseline"][s] == totals["conditional"][s]}


def _paired(a: dict[str, float], b: dict[str, float]) -> tuple[list[float], list[float], list[str]]:
    seeds = sorted(set(a) & set(b), key=lambda s: int(s) if s.isdigit() else s)
    return [a[s] for s in seeds], [b[s] for s in seeds], seeds


def _wilcoxon(xs: list[float], ys: list[float]) -> float | None:
    """p bilateral, ou None se o teste nao for aplicavel a esta amostra."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    diffs = [x - y for x, y in zip(xs, ys)]
    if not any(d != 0 for d in diffs):
        return None
    try:
        return float(wilcoxon(xs, ys).pvalue)
    except ValueError:
        return None


def _report_pair(label: str, base: dict[str, float], cond: dict[str, float],
                 note: str = "") -> float:
    xs, ys, seeds = _paired(base, cond)
    if not seeds:
        print(f"  {label:<10}  sem seeds em comum")
        return 0.0
    deltas = [(y - x) * 100 for x, y in zip(xs, ys)]
    mean = sum(deltas) / len(deltas)
    wins = sum(1 for d in deltas if d > 0)
    p = _wilcoxon(ys, xs)
    ptxt = "  p=n/a" if p is None else f"  p={p:.4f}"
    print(f"  {label:<10}{sum(xs) / len(xs) * 100:>8.2f}%{sum(ys) / len(ys) * 100:>9.2f}%"
          f"{mean:>+9.2f}pp{wins:>6}/{len(deltas)}{ptxt}   {note}")
    return mean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablacao do condicionamento em arr_len (Tarefa 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline", type=str, default="outputs/task5/results_e2000.csv",
                        help="Execucao sem condicionamento")
    parser.add_argument("--conditional", type=str, default="outputs/task5/results_e2000_cond.csv",
                        help="Execucao com --conditional")
    parser.add_argument("--arm", type=str, default="WGAN",
                        help="Nome do braco da WGAN-GP (o antigo 'GAN' e aceite e normalizado)")
    parser.add_argument("--label", type=str, default="condicionamento em arr_len",
                        help="O que distingue as duas execucoes. A mesma maquina serve "
                             "qualquer ablacao de um so fator (epocas, z_dim, fm_weight), "
                             "desde que os bracos de controlo fiquem intocados.")
    args = parser.parse_args()

    paths = {"baseline": Path(args.baseline), "conditional": Path(args.conditional)}
    for name, path in paths.items():
        if not path.exists():
            print(f"[ERRO] {name} nao encontrado: {path}")
            sys.exit(1)

    base, base_tot = _load(paths["baseline"])
    cond, cond_tot = _load(paths["conditional"])

    if args.arm not in base or args.arm not in cond:
        print(f"[ERRO] braco '{args.arm}' ausente de uma das execucoes")
        sys.exit(1)

    print(SEP)
    print(f"  ABLACAO — {args.label}")
    print(SEP)
    print(f"\n  sem : {paths['baseline']}")
    print(f"  com : {paths['conditional']}")

    usable = _assert_homogeneous({"baseline": base_tot, "conditional": cond_tot})
    dropped = (set(base_tot) & set(cond_tot)) - usable
    if dropped:
        print(f"  seeds excluidos : {sorted(dropped, key=lambda s: int(s) if s.isdigit() else 0)}")
    if not usable:
        print("\n[ERRO] nenhum seed comparavel entre as duas execucoes.")
        sys.exit(1)
    for table in (base, cond):
        for arm, scores in table.items():
            table[arm] = {s: v for s, v in scores.items() if s in usable}

    print(f"\n  {'braco':<10}{'sem':>9}{'com':>10}{'delta':>11}{'seeds':>9}"
          f"{'':>10}")
    print(f"  {'-' * 74}")

    # Controlos primeiro: se estes se mexeram, o resto nao se le.
    drifts = []
    for arm in CONTROL_ARMS:
        if arm in base and arm in cond:
            drifts.append((arm, _report_pair(arm, base[arm], cond[arm], "(controlo)")))

    print()
    delta = _report_pair(args.arm, base[args.arm], cond[args.arm], "<-- efeito")

    print(f"\n{SEP}")
    print("  LEITURA")
    print(SEP)

    bad = [(a, d) for a, d in drifts if abs(d) > CONTROL_TOL_PP]
    if bad:
        print(f"\n  [AVISO] controlos fora de {CONTROL_TOL_PP} pp: "
              + ", ".join(f"{a} {d:+.2f}pp" for a, d in bad))
        print(f"  A diferenca na WGAN-GP nao pode ser atribuida a {args.label};")
        print("  alguma coisa no pipeline partilhado mudou entre as duas execucoes.")
    else:
        print(f"\n  Controlos estaveis (todos dentro de {CONTROL_TOL_PP} pp).")
        print(f"  A diferenca observada e atribuivel a {args.label}.")

    xs, ys, seeds = _paired(base[args.arm], cond[args.arm])
    p = _wilcoxon(ys, xs)
    print(f"\n  Efeito ({args.label}): {delta:+.2f} pp em {len(seeds)} seeds emparelhados.")
    if p is not None and p < 0.05:
        direcao = "melhora" if delta > 0 else "piora"
        print(f"  Significativo (p={p:.4f}): {args.label} {direcao} a utilidade.")
    else:
        print(f"  Nao significativo{'' if p is None else f' (p={p:.4f})'}.")
        print("  Com este n, um efeito abaixo de ~1,5 pp dificilmente seria detectado,")
        print("  por isso isto delimita o efeito em vez de o excluir.")

    # A escada continua a ser o enquadramento: um ganho interno so interessa se
    # mover a WGAN-GP em relacao aos baselines que a tese usa para a julgar.
    print(f"\n  Escada na execucao 'com':")
    for other in CONTROL_ARMS:
        if other not in cond:
            continue
        xs, ys, seeds = _paired(cond[other], cond[args.arm])
        if not seeds:
            continue
        deltas = [(y - x) * 100 for x, y in zip(xs, ys)]
        mean = sum(deltas) / len(deltas)
        wins = sum(1 for d in deltas if d > 0)
        p = _wilcoxon(ys, xs)
        ptxt = "n/a" if p is None else f"{p:.4f}"
        print(f"    vs {other:<8}{mean:>+8.2f}pp{wins:>6}/{len(deltas)}   p={ptxt}")


if __name__ == "__main__":
    main()
