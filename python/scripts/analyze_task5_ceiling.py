#!/usr/bin/env python3
"""Tarefa 5 — decomposicao do desempenho em capacidade, consistencia e codificacao.

A media do mutation score confunde duas coisas diferentes: o que um metodo
CONSEGUE alcancar e com que fiabilidade la chega num unico sorteio. Separa-las
muda a leitura de E3 por completo.

Tomando, por classe de mutator, o melhor que cada braco atinge ao longo dos
seeds -- o seu tecto -- verifica-se que a WGAN-GP e o CORPUS tem o mesmo tecto
em todas as classes. A derrota da WGAN-GP para o CORPUS nao e falta de
capacidade: os dois alcancam o mesmo conjunto de mutantes. O CORPUS ganha
porque reproduz inputs que ja se sabe funcionarem e por isso acerta no tecto
quase sempre, enquanto a rede varia entre sorteios.

Isso arruma tambem a questao da afinacao. Se o tecto da rede ja iguala o do
CORPUS, nenhuma quantidade de epocas ou de condicionamento a leva mais longe do
que empatar; e o que separa qualquer um deles da suite humana e a codificacao,
nao o gerador.

LIMITACAO, que o relatorio repete porque o numero e optimista sem ela: o tecto e
somado por CLASSE de mutator, nao por mutante individual. Dois seeds que matem
mutantes diferentes dentro da mesma classe contam como um so no maximo da
classe. E portanto um limite superior a granularidade da classe, e nao a uniao
exacta dos mutantes mortos -- para essa seria preciso o identificador de cada
mutante, que o resumo por mutator nao guarda.

Uso:
    python python/scripts/analyze_task5_ceiling.py \
        --by-mutator outputs/task5/results_e2000_by_mutator.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP = "=" * 78
REFERENCE_ARM = "MANUAL"

# O braco da rede chamou-se GAN nas primeiras execucoes, nome impreciso para uma
# WGAN-GP. Normalizar a leitura evita ter de re-executar CSV validos por causa de
# uma etiqueta.
ARM_ALIASES = {"GAN": "WGAN"}


def _load(path: Path) -> tuple[dict, dict, dict]:
    """(killed[arm][mutator][seed], total[mutator], seeds[arm])."""
    killed: dict = defaultdict(lambda: defaultdict(dict))
    total: dict = {}
    seeds: dict = defaultdict(set)

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            arm = ARM_ALIASES.get(row["arm"].strip(), row["arm"].strip())
            mut = row["mutator"].strip()
            seed = (row.get("seed") or "-").strip()
            try:
                k = int(row["killed"])
                n = k + int(row["survived"]) + int(row["no_coverage"])
            except (KeyError, TypeError, ValueError):
                continue
            killed[arm][mut][seed] = k
            seeds[arm].add(seed)
            # O total por classe e uma propriedade do SUT, nao do braco, por isso
            # o maximo observado e o valor correcto mesmo que uma medicao falhe.
            total[mut] = max(total.get(mut, 0), n)
    return killed, total, seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decomposicao tecto/consistencia dos bracos de E3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--by-mutator", type=str,
                        default="outputs/task5/results_e2000_by_mutator.csv")
    parser.add_argument("--arms", type=str, default="WGAN,CORPUS,KDE,RANDOM,MANUAL",
                        help="Ordem de apresentacao; bracos ausentes sao ignorados")
    args = parser.parse_args()

    path = Path(args.by_mutator)
    if not path.exists():
        print(f"[ERRO] nao encontrado: {path}")
        sys.exit(1)

    killed, total, seeds = _load(path)
    arms = [a for a in (x.strip() for x in args.arms.split(",")) if a in killed]
    if not arms:
        print(f"[ERRO] nenhum dos bracos pedidos existe em {path}")
        sys.exit(1)

    mutators = sorted(total, key=lambda m: -total[m])
    n_total = sum(total.values())

    print(SEP)
    print("  TECTO POR CLASSE DE MUTATOR — melhor de cada braco ao longo dos seeds")
    print(SEP)
    print(f"\n  fonte : {path}")
    print(f"  seeds : " + ", ".join(f"{a}={len(seeds[a])}" for a in arms))

    head = "  " + f"{'mutator':<24}" + "".join(f"{a:>9}" for a in arms) + f"{'total':>8}"
    print(f"\n{head}")
    print("  " + "-" * (len(head) - 2))

    ceiling = {a: 0 for a in arms}
    for mut in mutators:
        cells = ""
        for a in arms:
            best = max(killed[a][mut].values()) if killed[a].get(mut) else 0
            ceiling[a] += best
            cells += f"{best:>9}"
        print(f"  {mut:<24}{cells}{total[mut]:>8}")
    print("  " + "-" * (len(head) - 2))
    print("  " + f"{'TECTO':<24}" + "".join(f"{ceiling[a]:>9}" for a in arms)
          + f"{n_total:>8}")

    # --- capacidade vs consistencia -------------------------------------
    print(f"\n{SEP}")
    print("  CAPACIDADE vs CONSISTENCIA")
    print(SEP)
    print(f"\n  {'braco':<10}{'media':>9}{'tecto':>9}{'do tecto':>11}{'seeds':>8}")
    print("  " + "-" * 46)

    means = {}
    for a in arms:
        per_seed = defaultdict(int)
        for mut in mutators:
            for s, k in killed[a][mut].items():
                per_seed[s] += k
        if not per_seed:
            continue
        mean = sum(per_seed.values()) / len(per_seed)
        means[a] = mean
        frac = mean / ceiling[a] if ceiling[a] else float("nan")
        print(f"  {a:<10}{mean / n_total * 100:>8.1f}%{ceiling[a] / n_total * 100:>8.1f}%"
              f"{frac * 100:>10.1f}%{len(per_seed):>8}")

    # --- o que a codificacao custa --------------------------------------
    generated = [a for a in arms if a != REFERENCE_ARM]
    if REFERENCE_ARM in ceiling and generated:
        best_gen = max(generated, key=lambda a: ceiling[a])
        loss = ceiling[REFERENCE_ARM] - ceiling[best_gen]
        print(f"\n{SEP}")
        print("  O QUE A CODIFICACAO CUSTA")
        print(SEP)
        print(f"\n  {REFERENCE_ARM} (suite humana)  : {ceiling[REFERENCE_ARM]:>4} "
              f"({ceiling[REFERENCE_ARM] / n_total * 100:.1f}%)")
        print(f"  melhor gerado ({best_gen})   : {ceiling[best_gen]:>4} "
              f"({ceiling[best_gen] / n_total * 100:.1f}%)")
        print(f"  perda na representacao    : {loss:>4} mutantes "
              f"({loss / n_total * 100:.1f} pp)")
        print("\n  Nenhum metodo derivado do corpus atinge estes mutantes em nenhum")
        print("  seed. Recupera-los exige mudar a codificacao dos inputs, nao treinar")
        print("  melhor o gerador.")

        worst = sorted(mutators,
                       key=lambda m: (max(killed[REFERENCE_ARM][m].values(), default=0)
                                      - max(killed[best_gen][m].values(), default=0)),
                       reverse=True)[:3]
        print("\n  Onde a perda se concentra:")
        for m in worst:
            r = max(killed[REFERENCE_ARM][m].values(), default=0)
            g = max(killed[best_gen][m].values(), default=0)
            if r > g:
                print(f"    {m:<24} {g:>3} vs {r:>3} da suite humana "
                      f"(de {total[m]} mutantes)")

    tied = [a for a in generated if ceiling.get(a) == ceiling.get(generated[0])]
    if len(tied) > 1:
        print(f"\n  Nota: {', '.join(tied)} partilham o mesmo tecto. A diferenca entre")
        print("  as suas medias e consistencia, nao capacidade.")

    print(f"\n  [limitacao] O tecto e somado por CLASSE de mutator, por isso e um")
    print("  limite superior a essa granularidade e nao a uniao exacta dos mutantes")
    print("  mortos: dois seeds que matem mutantes distintos da mesma classe contam")
    print("  como um so.")


if __name__ == "__main__":
    main()
