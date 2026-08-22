#!/usr/bin/env python3
"""Tarefa 5 - regime de poucos dados: onde gerar pode valer mais do que repetir.

Com o corpus completo, a WGAN-GP e a reamostragem do corpus tem o MESMO teto em
todas as classes de mutator (ver analyze_task5_ceiling.py). A rede perde na media
por consistencia, nao por capacidade, e nenhuma afinacao a leva mais longe do que
empatar.

Esta experiencia ataca a assimetria estrutural entre os dois metodos. A
reamostragem so consegue devolver entradas que ja viu: o seu teto e o numero de
entradas distintas do corpus, e cai proporcionalmente quando o corpus encolhe. A
rede aprende uma densidade e interpola, pelo que o seu teto deve cair mais
devagar. Se isso se verificar, existe um regime -- corpus pequeno -- em que gerar
bate repetir, e esse e exactamente o cenario que justifica gerar dados de teste:
quando ainda nao existe uma suite de onde os extrair.

O --corpus-frac restringe TODOS os bracos ao mesmo subconjunto, incluindo o
CORPUS. Sem isso o CORPUS teria uma vantagem de informacao e a comparacao nao
diria nada.

A leitura nao e so a media. Reporta-se tambem o teto por fraccao, porque a media
mede consistencia e o teto mede capacidade -- e a hipotese e sobre capacidade.

Uso:
    python python/scripts/analyze_task5_lowdata.py
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
ARM_ALIASES = {"GAN": "WGAN"}
ARMS = ["WGAN", "CORPUS", "KDE", "RANDOM"]
CORPUS_SIZE = 121


def _norm(arm: str) -> str:
    arm = arm.strip()
    return ARM_ALIASES.get(arm, arm)


def _load(path: Path) -> tuple[dict, set]:
    """{arm: {seed: mutation_score}} e o conjunto de populacoes de mutantes."""
    out: dict = defaultdict(dict)
    pops: set = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seed = (row.get("seed") or "").strip()
            if not seed or seed == "-":
                continue
            try:
                out[_norm(row["arm"])][seed] = float(row["mutation_score"])
                pops.add(int(row["killed"]) + int(row["survived"])
                         + int(row["no_coverage"]))
            except (KeyError, TypeError, ValueError):
                continue
    return out, pops


def _ceiling(path: Path) -> dict:
    """Teto por braco: soma do melhor resultado por classe de mutator.

    Limite superior a granularidade da classe, e nao a uniao exacta dos mutantes
    mortos -- o relatorio agregado do PIT nao guarda o identificador de cada
    mutante.
    """
    best: dict = defaultdict(lambda: defaultdict(int))
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seed = (row.get("seed") or "").strip()
            if not seed or seed == "-":
                continue
            try:
                arm = _norm(row["arm"])
                mut = row["mutator"].strip()
                killed = int(row["killed"])
            except (KeyError, TypeError, ValueError):
                continue
            best[arm][mut] = max(best[arm][mut], killed)
    return {a: sum(v.values()) for a, v in best.items()}


def _wilcoxon(xs: list, ys: list):
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    if not any(x != y for x, y in zip(xs, ys)):
        return None
    try:
        return float(wilcoxon(xs, ys).pvalue)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regime de poucos dados (Tarefa 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--full", type=str,
                    default="outputs/task5/results_114_e2000.csv",
                    help="Execucao com o corpus completo (frac = 1.0)")
    ap.add_argument("--fracs", type=str,
                    default=("0.15:outputs/task5/results_114_frac15.csv,"
                             "0.30:outputs/task5/results_114_frac30.csv,"
                             "0.60:outputs/task5/results_114_frac60.csv"),
                    help="Lista frac:ficheiro separada por virgulas")
    args = ap.parse_args()

    entradas = [("1.00", Path(args.full))]
    for par in args.fracs.split(","):
        if ":" in par:
            fr, fp = par.split(":", 1)
            entradas.append((f"{float(fr):.2f}", Path(fp)))

    print(SEP)
    print("  REGIME DE POUCOS DADOS - capacidade de gerar vs capacidade de repetir")
    print(SEP)

    print("\n  " + f"{'frac':>6}{'n corpus':>10}  "
          + "".join(f"{a:>9}" for a in ARMS) + f"{'seeds':>7}")
    print("  " + "-" * 62)

    linhas = []
    for frac, path in entradas:
        if not path.exists():
            print(f"  {frac:>6}{'':>10}  (ainda sem dados: {path.name})")
            continue
        dados, pops = _load(path)
        presentes = [a for a in ARMS if a in dados]
        if not presentes:
            continue
        if len(pops) > 1:
            print(f"  [AVISO] {path.name} mistura populacoes {sorted(pops)}; ignorado")
            continue
        n_corpus = max(4, round(CORPUS_SIZE * float(frac)))
        seeds = sorted(set.intersection(*[set(dados[a]) for a in presentes]))
        if not seeds:
            continue
        med = {a: sum(dados[a][s] for s in seeds) / len(seeds) for a in presentes}
        print(f"  {frac:>6}{n_corpus:>10}  "
              + "".join(f"{med[a] * 100:>8.1f}%" if a in med else f"{'-':>9}"
                        for a in ARMS)
              + f"{len(seeds):>7}")
        linhas.append((frac, n_corpus, path, dados, seeds))

    if not linhas:
        print("\n  Sem dados suficientes ainda.")
        return

    print(f"\n{SEP}")
    print("  WGAN vs CORPUS por fraccao (emparelhado por seed)")
    print(SEP)
    print("\n  " + f"{'frac':>6}{'n corpus':>10}{'delta':>12}{'a favor':>10}{'p':>10}")
    print("  " + "-" * 50)
    for frac, n_corpus, path, dados, seeds in linhas:
        if "WGAN" not in dados or "CORPUS" not in dados:
            continue
        xs = [dados["CORPUS"][s] for s in seeds]
        ys = [dados["WGAN"][s] for s in seeds]
        deltas = [(y - x) * 100 for x, y in zip(xs, ys)]
        media = sum(deltas) / len(deltas)
        p = _wilcoxon(ys, xs)
        ptxt = "n/a" if p is None else f"{p:.4f}"
        marca = "  <-- gerar bate repetir" if media > 0 else ""
        print(f"  {frac:>6}{n_corpus:>10}{media:>+11.2f}pp"
              f"{sum(1 for v in deltas if v > 0):>6}/{len(deltas)}"
              f"{ptxt:>10}{marca}")

    print(f"\n{SEP}")
    print("  TETO POR FRACCAO - NAO INTERPRETAR SEM LER A RESSALVA")
    print(SEP)
    print("")
    print("  O teto e o maximo por classe de mutator ao longo dos seeds. Isso mede")
    print("  capacidade quando todos os seeds veem os MESMOS dados, que e o caso com")
    print("  o corpus completo. Com corpus_frac < 1 deixa de medir: cada seed sorteia")
    print("  um subconjunto diferente, e a uniao ao longo dos seeds cobre grande")
    print("  parte do corpus. O numero passa a ser o melhor que se alcancaria podendo")
    print("  escolher o subconjunto ideal para cada classe, que nao e a pergunta.")
    print("  O teto do RANDOM chegar a SUBIR quando o corpus encolhe e o sintoma")
    print("  que denuncia o problema.")
    print("")
    print("  A limitacao do CORPUS le-se na diversidade por seed: ver a tabela final.")
    print("\n  " + f"{'frac':>6}{'n corpus':>10}  " + "".join(f"{a:>9}" for a in ARMS))
    print("  " + "-" * 55)
    for frac, n_corpus, path, dados, seeds in linhas:
        tetos = _ceiling(path.with_name(path.stem + "_by_mutator.csv"))
        if not tetos:
            print(f"  {frac:>6}{n_corpus:>10}  (sem ficheiro por mutator)")
            continue
        print(f"  {frac:>6}{n_corpus:>10}  "
              + "".join(f"{tetos.get(a, 0):>9}" for a in ARMS))

    print("\n  [limitacao] O teto e somado por CLASSE de mutator, por isso e um")
    print("  limite superior a essa granularidade e nao a uniao exacta dos")
    print("  mutantes mortos.")

    _diversidade(entradas)



def _diversidade(entradas: list) -> None:
    """Entradas distintas produzidas por seed -- onde a hipotese se le.

    Ao contrario do teto, esta medida e por seed: cada valor vem de uma execucao
    que viu um unico subconjunto, pelo que a variacao do subconjunto entre seeds
    nao a contamina. E aqui que a assimetria entre gerar e repetir aparece sem
    ambiguidade.
    """
    print("")
    print(SEP)
    print("  DIVERSIDADE - entradas distintas em 200 sorteios (media por seed)")
    print(SEP)
    print("")
    print("  " + f"{'frac':>6}{'n corpus':>10}  " + "".join(f"{a:>9}" for a in ARMS))
    print("  " + "-" * 55)
    for frac, path in entradas:
        if not path.exists():
            continue
        acc: dict = defaultdict(list)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("seed") or "").strip() in ("", "-"):
                    continue
                try:
                    acc[_norm(row["arm"])].append(float(row["n_unique"]))
                except (KeyError, TypeError, ValueError):
                    continue
        if not acc:
            continue
        n_corpus = max(4, round(CORPUS_SIZE * float(frac)))
        print(f"  {frac:>6}{n_corpus:>10}  "
              + "".join(f"{sum(acc[a]) / len(acc[a]):>9.0f}" if acc.get(a)
                        else f"{'-':>9}" for a in ARMS))
    print("")
    print("  O CORPUS nao pode exceder o numero de entradas distintas que viu.")
    print("  A rede interpola, e por isso a sua diversidade cai muito mais devagar.")


if __name__ == "__main__":
    main()
