#!/usr/bin/env python3
"""
Mutation Operator Survival Analysis

Parses PIT mutations.csv files and reports which mutation operators survive
at the highest rate on *covered* code (code the tests actually execute).

Two distinct failure modes are separated:
  NO_COVERAGE  -> tests don't reach the code (coverage problem)
  SURVIVED     -> tests reach it but assertions are too weak (oracle problem)

Only SURVIVED on covered code reveals genuine test weakness. This drives
targeted test improvement: write assertions that catch the operators with
high on-covered survival rates.

Per operator, the key metric is:
  survival_rate_covered = survived / (killed + survived)    [on covered code only]

A high value means tests execute the code but don't detect the mutation.
A low value means the oracle is already catching this operator well.

Usage:
    python python/scripts/analyze_mutant_survival.py \
        --results-dir outputs/mutation_experiments/sf110_full111 \
        [--min-covered 20] \
        [--compare outputs/mutation_experiments/sf110_wgan_guided_v3] \
        [--export outputs/mutation_experiments/operator_survival.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP = "=" * 72

# PIT mutations.csv column layout
_COL_CLASS = 1
_COL_MUTATOR = 2
_COL_LINE = 4
_COL_STATUS = 5

_KILLED_STATUSES = {"KILLED", "TIMED_OUT"}
_NONVIABLE_STATUSES = {"NON_VIABLE", "MEMORY_ERROR", "RUN_ERROR"}


def _short_name(fqn: str) -> str:
    """Extract a readable short name from the PIT mutator FQN."""
    name = fqn.split(".")[-1]
    # Strip common suffixes for readability
    for suffix in ("Mutator", "MUTATOR"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _parse_mutations_csv(csv_path: Path) -> list[dict]:
    """Return one record per mutation row."""
    rows = []
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.reader(f):
            if len(row) <= _COL_STATUS:
                continue
            status = row[_COL_STATUS].strip().upper()
            if not status:
                continue
            rows.append({
                "class": row[_COL_CLASS].strip(),
                "mutator": _short_name(row[_COL_MUTATOR].strip()),
                "mutator_fqn": row[_COL_MUTATOR].strip(),
                "status": status,
            })
    return rows


def _collect(results_dir: Path) -> tuple[list[dict], int]:
    """Walk workdir and collect all mutation records. Returns (records, n_projects)."""
    workdir = results_dir / "workdir"
    if not workdir.exists():
        print(f"[ERRO] Nenhum workdir/ em {results_dir}")
        sys.exit(1)

    all_rows: list[dict] = []
    n_projects = 0
    for proj_dir in sorted(workdir.iterdir()):
        if not proj_dir.is_dir():
            continue
        csvs = list(proj_dir.rglob("mutations.csv"))
        if not csvs:
            continue
        mc = max(csvs, key=lambda p: p.stat().st_mtime)
        rows = _parse_mutations_csv(mc)
        for r in rows:
            r["project"] = proj_dir.name
        all_rows.extend(rows)
        n_projects += 1
    return all_rows, n_projects


def _aggregate_by_operator(rows: list[dict]) -> dict[str, dict]:
    """
    Aggregate mutation counts per operator.

    Returns dict keyed by short operator name with fields:
      total, killed, survived, no_coverage, non_viable,
      covered (= killed + survived),
      kill_rate_covered, survival_rate_covered,
      pct_no_coverage
    """
    stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "killed": 0, "survived": 0, "no_coverage": 0, "non_viable": 0
    })

    for r in rows:
        s = r["status"]
        op = r["mutator"]
        stats[op]["total"] += 1
        if s in _KILLED_STATUSES:
            stats[op]["killed"] += 1
        elif s == "SURVIVED":
            stats[op]["survived"] += 1
        elif s == "NO_COVERAGE":
            stats[op]["no_coverage"] += 1
        elif s in _NONVIABLE_STATUSES:
            stats[op]["non_viable"] += 1
        else:
            stats[op]["survived"] += 1  # unknown → conservative

    result = {}
    for op, d in stats.items():
        covered = d["killed"] + d["survived"]
        result[op] = {
            **d,
            "covered": covered,
            "kill_rate_covered": d["killed"] / covered if covered else 0.0,
            "survival_rate_covered": d["survived"] / covered if covered else 0.0,
            "pct_no_coverage": d["no_coverage"] / d["total"] if d["total"] else 0.0,
        }
    return result


def _report(stats: dict[str, dict], min_covered: int) -> None:
    print(SEP)
    print("  ANALISE DE SURVIVAL POR OPERADOR DE MUTACAO")
    print(SEP)

    filtered = {op: s for op, s in stats.items() if s["covered"] >= min_covered}
    if not filtered:
        print(f"  Nenhum operador com >= {min_covered} mutacoes cobertas.")
        return

    # Sort by survival rate on covered code (highest = hardest to kill)
    by_survival = sorted(filtered.items(), key=lambda t: -t[1]["survival_rate_covered"])

    total_all = sum(s["total"] for s in stats.values())
    total_killed = sum(s["killed"] for s in stats.values())
    total_survived = sum(s["survived"] for s in stats.values())
    total_covered = sum(s["covered"] for s in stats.values())
    total_no_cov = sum(s["no_coverage"] for s in stats.values())

    print(f"\n  Total de mutacoes: {total_all}")
    print(f"  Cobertas (killed+survived): {total_covered} ({total_covered/total_all*100:.1f}%)")
    print(f"  Sem cobertura: {total_no_cov} ({total_no_cov/total_all*100:.1f}%)")
    overall_kill = total_killed / total_covered * 100 if total_covered else 0
    print(f"  Kill rate global (em cobertas): {overall_kill:.1f}%")
    print(f"  Survival rate global (em cobertas): {100-overall_kill:.1f}%")

    print(f"\n  Operadores com >= {min_covered} mutacoes cobertas: {len(filtered)}")
    print(f"\n  {'Operador':<35} {'Total':>6} {'Coberto':>8} {'Killed':>7} {'Surviv':>7} "
          f"{'Kill%':>7} {'Surv%':>7} {'NoCov%':>8}")
    print(f"  {'-'*88}")

    for op, s in by_survival:
        kill_pct = s["kill_rate_covered"] * 100
        surv_pct = s["survival_rate_covered"] * 100
        nocov_pct = s["pct_no_coverage"] * 100
        flag = " <-- FRACO" if surv_pct > 40 else ""
        print(f"  {op:<35} {s['total']:>6} {s['covered']:>8} {s['killed']:>7} {s['survived']:>7} "
              f"{kill_pct:>6.1f}% {surv_pct:>6.1f}% {nocov_pct:>7.1f}%{flag}")

    print(f"\n  Leitura: Surv% alto em cobertas = oracle fraco para este operador.")
    print(f"  Kill% alto = testes ja detectam bem este tipo de mutacao.")

    # Highlight problem operators
    weak = [(op, s) for op, s in by_survival if s["survival_rate_covered"] > 0.35]
    if weak:
        print(f"\n{SEP}")
        print("  OPERADORES MAIS DIFICEIS DE MATAR (survival > 35% em cobertas)")
        print(SEP)
        print(f"\n  Estas mutacoes sobrevivem mesmo quando o codigo e executado.")
        print(f"  Melhorar assertions para estes casos tem o maior impacto no strength.\n")
        for op, s in weak[:8]:
            surv_pct = s["survival_rate_covered"] * 100
            nocov_pct = s["pct_no_coverage"] * 100
            print(f"  {op}")
            print(f"    Survival em cobertas: {surv_pct:.1f}%  |  Sem cobertura: {nocov_pct:.1f}%")
            _print_operator_advice(op)


def _print_operator_advice(op: str) -> None:
    """Print targeted advice for commonly weak operators."""
    advice = {
        "ConditionalsBoundary": (
            "Testar valores nas fronteiras exactas (e.g. n==limit, n==limit+1). "
            "JUnit: @ParameterizedTest com valores bordejantes."
        ),
        "NegateConditionals": (
            "Verificar que o resultado muda quando a condicao inverte. "
            "Testar o caminho falso explicitamente, nao so o verdadeiro."
        ),
        "ReturnVals": (
            "Assertions sobre o valor de retorno, nao apenas sobre efeitos laterais. "
            "assertEquals/assertNotNull apos cada chamada que retorna valor."
        ),
        "VoidMethodCall": (
            "Verificar side-effects de metodos void: estado do objecto apos chamada, "
            "interaccoes com mocks (verify()), ou excepcoes lancadas."
        ),
        "Math": (
            "Comparar resultado aritmetico com valor calculado manualmente. "
            "Testar overflow, divisao por zero, e valores negativos."
        ),
        "IncrementsMutator": (
            "Verificar que incrementos/decrementos produzem o valor esperado. "
            "Testar loops com contadores explicitos."
        ),
        "InvertNegs": (
            "Testar com valores negativos e verificar sinal do resultado."
        ),
        "Substitution": (
            "Assertions numericas exactas sobre resultados de operadores aritmeticos."
        ),
    }
    for key, text in advice.items():
        if key.lower() in op.lower():
            print(f"    -> {text}")
            return
    print(f"    -> Adicionar assertions sobre o valor/estado alterado por este operador.")


def _compare_two_runs(stats_a: dict, stats_b: dict, label_a: str, label_b: str, min_covered: int) -> None:
    """Compare survival rates between two runs per operator."""
    print(f"\n{SEP}")
    print(f"  COMPARACAO DE SURVIVAL: {label_a} vs {label_b}")
    print(SEP)

    all_ops = set(stats_a) | set(stats_b)
    rows = []
    for op in all_ops:
        sa = stats_a.get(op, {})
        sb = stats_b.get(op, {})
        if sa.get("covered", 0) < min_covered and sb.get("covered", 0) < min_covered:
            continue
        surv_a = sa.get("survival_rate_covered", 0) * 100
        surv_b = sb.get("survival_rate_covered", 0) * 100
        delta = surv_b - surv_a
        rows.append((op, surv_a, surv_b, delta))

    rows.sort(key=lambda t: t[3])  # most improved first (most negative delta)

    print(f"\n  {'Operador':<35} {label_a[:12]:>12} {label_b[:12]:>12} {'Delta':>8}")
    print(f"  {'-'*70}")
    for op, sa, sb, d in rows:
        arrow = "v" if d < -2 else ("^" if d > 2 else "=")
        print(f"  {op:<35} {sa:>11.1f}% {sb:>11.1f}% {d:>+7.1f}% {arrow}")


def _export(stats: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "operator", "total", "killed", "survived", "no_coverage", "non_viable",
            "covered", "kill_rate_covered", "survival_rate_covered", "pct_no_coverage",
        ])
        for op, s in sorted(stats.items(), key=lambda t: -t[1]["survival_rate_covered"]):
            writer.writerow([
                op, s["total"], s["killed"], s["survived"], s["no_coverage"], s["non_viable"],
                s["covered"],
                round(s["kill_rate_covered"], 4),
                round(s["survival_rate_covered"], 4),
                round(s["pct_no_coverage"], 4),
            ])
    print(f"\n  Exportado: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mutation operator survival analysis — identifies weak test oracles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=str,
                        default="outputs/mutation_experiments/sf110_full111",
                        help="Results directory with workdir/*/pit-reports/mutations.csv")
    parser.add_argument("--min-covered", type=int, default=20,
                        help="Only report operators with at least this many covered mutations")
    parser.add_argument("--compare", type=str, default=None,
                        help="Second results directory to compare survival rates")
    parser.add_argument("--export", type=str, default=None,
                        help="Export per-operator stats to this CSV path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERRO] Directory not found: {results_dir}")
        sys.exit(1)

    print(f"\nA processar: {results_dir.name}")
    rows, n_proj = _collect(results_dir)
    print(f"Projetos encontrados: {n_proj} | Mutacoes totais: {len(rows)}")

    stats = _aggregate_by_operator(rows)
    _report(stats, args.min_covered)

    if args.compare:
        compare_dir = Path(args.compare)
        if compare_dir.exists():
            rows_b, n_proj_b = _collect(compare_dir)
            print(f"\nA processar (comparacao): {compare_dir.name} | {n_proj_b} projetos")
            stats_b = _aggregate_by_operator(rows_b)
            _compare_two_runs(stats, stats_b, results_dir.name[:12], compare_dir.name[:12], args.min_covered)
        else:
            print(f"[AVISO] Directorio de comparacao nao encontrado: {compare_dir}")

    if args.export:
        _export(stats, Path(args.export))


if __name__ == "__main__":
    main()
