#!/usr/bin/env python3
"""
SF110 Mutation Testing Results Analyser

Parses an existing results run and produces a prioritised, actionable report:
- Overall summary (score, strength, pass/fail rate)
- Failure breakdown with specific causes
- Quick-win projects (close to passing or easy to fix)
- Score distribution histogram
- Estimated impact of each improvement strategy

Usage:
    python python/scripts/analyze_sf110_results.py
    python python/scripts/analyze_sf110_results.py --results-dir outputs/mutation_experiments/sf110_full_optimized
    python python/scripts/analyze_sf110_results.py --results-dir outputs/mutation_experiments/sf110_full_optimized --full
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP = "=" * 72
SEP2 = "-" * 72


def _load(results_dir: Path) -> tuple[list[dict], dict]:
    csv_candidates = list(results_dir.glob("*.csv"))
    json_candidates = list(results_dir.glob("*summary*.json"))

    if not csv_candidates:
        raise FileNotFoundError(f"No CSV found in {results_dir}")

    results_csv = max(csv_candidates, key=lambda p: p.stat().st_mtime)
    summary_json = max(json_candidates, key=lambda p: p.stat().st_mtime) if json_candidates else None

    rows = []
    with open(results_csv, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    summary = {}
    if summary_json:
        with open(summary_json) as f:
            summary = json.load(f)

    return rows, summary


def _section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _print_summary(rows: list[dict], summary: dict) -> None:
    _section("1. RESUMO GERAL")

    passed = [r for r in rows if r["status"] == "passed"]
    failed = [r for r in rows if r["status"] != "passed"]

    print(f"  Projetos executados : {len(rows)}")
    print(f"  Passed              : {len(passed)} ({len(passed)/len(rows)*100:.1f}%)")
    print(f"  Failed              : {len(failed)} ({len(failed)/len(rows)*100:.1f}%)")

    if summary:
        print(f"\n  Weighted mutation score   : {summary.get('overall_weighted_mutation_score', '?')}%")
        print(f"  Weighted test strength    : {summary.get('overall_weighted_test_strength', '?')}%")
        print(f"  Total killed mutations    : {summary.get('killed_sum', '?'):,}")
        print(f"  Total relevant mutations  : {summary.get('relevant_mutations_sum', '?'):,}")

    if passed:
        scores = [float(r["score"]) for r in passed]
        scores.sort()
        avg = sum(scores) / len(scores)
        median = scores[len(scores) // 2]
        print(f"\n  Score distribution (passed projects):")
        print(f"    Min    : {min(scores):.1f}%")
        print(f"    Median : {median:.1f}%")
        print(f"    Mean   : {avg:.1f}%")
        print(f"    Max    : {max(scores):.1f}%")
        bands = [0, 10, 25, 50, 75, 101]
        labels = ["0-10%", "10-25%", "25-50%", "50-75%", ">75%"]
        print(f"\n  Score bands:")
        for i, label in enumerate(labels):
            count = sum(1 for s in scores if bands[i] <= s < bands[i + 1])
            bar = "#" * count
            print(f"    {label:10s}: {bar} {count}")


def _print_failures(rows: list[dict], full: bool) -> None:
    _section("2. ANALISE DE FALHAS")

    failed = [r for r in rows if r["status"] != "passed"]
    reasons = Counter(r["failure_reason"] for r in failed)

    print(f"  Total de falhas: {len(failed)}\n")
    print(f"  {'Razao':<35} {'Count':>5}  {'%':>5}  Accao sugerida")
    print(f"  {SEP2}")

    actions = {
        "oom": "Usar --jvm-max-heap 2048m --retry-oom (automatico)",
        "oom_retry_failed:oom": "Projeto demasiado grande; tentar --max-classes 5",
        "compile_error": "Usar --retry-compile para testar source levels 1.7/1.6",
        "pit_missing_csv": "Verificar logs; pode ser OOM (exit=3221226091) ou timeout",
        "command_failed": "Ver stdout_tail no CSV; pode ser problema de classpath",
        "zero_tests": "Projeto sem testes - gerar com EvoSuite manualmente",
        "no_target_classes": "Melhorar descoberta de classes alvo",
        "timeout": "Aumentar --timeout-seconds ou reduzir --max-classes",
        "no_test_sources": "Modo de teste incompativel com estrutura do projeto",
    }

    for reason, count in reasons.most_common():
        pct = count / len(failed) * 100
        action = actions.get(reason, "Investigar manualmente")
        print(f"  {reason:<35} {count:>5}  {pct:>4.0f}%  {action}")

    if full:
        print(f"\n  Projetos por razao de falha:")
        by_reason: dict[str, list[str]] = defaultdict(list)
        for r in failed:
            by_reason[r["failure_reason"]].append(r["project"])
        for reason, projects in sorted(by_reason.items()):
            print(f"\n  [{reason}]")
            for p in sorted(projects):
                print(f"    {p}")


def _print_quick_wins(rows: list[dict], full: bool) -> None:
    _section("3. QUICK WINS — projetos com maior potencial de melhoria")

    passed = [r for r in rows if r["status"] == "passed"]

    # A) Projects with good strength but high no_coverage (better target class selection would help)
    print("  A) Bom test strength mas alto no_coverage (melhor selecao de classes alvo):")
    print(f"     -> Problema: os testes cobrem bem o que testam, mas ha muito codigo ignorado")
    print(f"     -> Fix: --max-classes 0 (sem limite), ou melhorar _derive_target_classes\n")

    candidate_a = []
    for r in passed:
        total = int(r["total_mutations"]) if r["total_mutations"] else 0
        no_cov = int(r["no_coverage"]) if r["no_coverage"] else 0
        strength = float(r["test_strength"])
        score = float(r["score"])
        if total > 50 and no_cov / total > 0.6 and strength > 60:
            candidate_a.append((score, strength, no_cov / total * 100, total, r["project"]))

    candidate_a.sort(reverse=True)
    print(f"     {'Projeto':<28} {'Score':>6} {'Strength':>9} {'No_Cov':>8} {'Mut':>6}")
    print(f"     {'-'*60}")
    for score, strength, nc_pct, total, proj in candidate_a[:10]:
        print(f"     {proj:<28} {score:>5.1f}% {strength:>8.1f}% {nc_pct:>7.0f}%  {total:>5}")

    # B) Low-score projects with high no_coverage but large codebase (most impact if fixed)
    print(f"\n  B) Projetos com baixo score por no_coverage mas grande codebase (alto impacto):")
    print(f"     -> Fix: verificar se todas as classes alvo estao a ser incluidas no PIT\n")
    candidate_b = []
    for r in passed:
        total = int(r["total_mutations"]) if r["total_mutations"] else 0
        no_cov = int(r["no_coverage"]) if r["no_coverage"] else 0
        score = float(r["score"])
        if total > 500 and score < 20:
            potential = float(r["test_strength"]) * (1 - no_cov / total) if total > 0 else 0
            candidate_b.append((total, score, no_cov / total * 100, r["project"]))

    candidate_b.sort(reverse=True)
    print(f"     {'Projeto':<28} {'Tot_Mut':>7} {'Score':>6} {'No_Cov':>8}")
    print(f"     {'-'*55}")
    for total, score, nc_pct, proj in candidate_b[:8]:
        print(f"     {proj:<28} {total:>7} {score:>5.1f}% {nc_pct:>7.0f}%")


def _print_top_bottom(rows: list[dict], n: int = 10) -> None:
    _section(f"4. TOP {n} e BOTTOM {n} (projetos que passaram)")

    passed = sorted(
        [r for r in rows if r["status"] == "passed"],
        key=lambda r: float(r["score"]),
        reverse=True,
    )

    print(f"  {'TOP ' + str(n):<28} {'Score':>6} {'Strength':>9} {'Killed':>7} {'Survived':>9} {'No_Cov':>7} {'Total':>6}")
    print(f"  {'-'*75}")
    for r in passed[:n]:
        print(
            f"  {r['project']:<28} {float(r['score']):>5.1f}% "
            f"{float(r['test_strength']):>8.1f}% {r['killed']:>7} "
            f"{r['survived']:>9} {r['no_coverage']:>7} {r['total_mutations']:>6}"
        )

    print()
    print(f"  {'BOTTOM ' + str(n):<28} {'Score':>6} {'Strength':>9} {'No_Cov%':>8} {'Total':>6}")
    print(f"  {'-'*60}")
    for r in reversed(passed[-n:]):
        total = int(r["total_mutations"])
        nc_pct = int(r["no_coverage"]) / total * 100 if total > 0 else 0
        print(
            f"  {r['project']:<28} {float(r['score']):>5.1f}% "
            f"{float(r['test_strength']):>8.1f}% {nc_pct:>7.0f}% {total:>6}"
        )


def _print_improvement_plan(rows: list[dict]) -> None:
    _section("5. PLANO DE MELHORIA — impacto estimado")

    passed = [r for r in rows if r["status"] == "passed"]
    failed = [r for r in rows if r["status"] != "passed"]

    total_killed = sum(int(r["killed"]) for r in passed)
    total_relevant = sum(int(r["total_relevant"]) for r in passed)
    current_score = total_killed / total_relevant * 100 if total_relevant > 0 else 0

    reasons = Counter(r["failure_reason"] for r in failed)
    oom_count = sum(v for k, v in reasons.items() if "oom" in k)
    compile_count = reasons.get("compile_error", 0)
    pit_csv_count = reasons.get("pit_missing_csv", 0)

    # Estimate: if we recover N failed projects, assume they score ~20% avg
    # (conservative, based on current avg of passed projects)
    avg_passed_score = sum(float(r["score"]) for r in passed) / len(passed) if passed else 0
    avg_relevant_per_project = total_relevant / len(passed) if passed else 0
    conservative_score = avg_passed_score * 0.5  # recovered projects typically have lower coverage

    print(f"  Score atual: {current_score:.2f}%  (killed={total_killed:,} / relevant={total_relevant:,})")
    print(f"  Media de score dos projetos que passaram: {avg_passed_score:.1f}%")
    print()
    print(f"  Estrategia                         Projetos  Score esperado adicional")
    print(f"  {'-'*70}")

    plans = [
        (
            "Retry OOM (--retry-oom, ja ativo)",
            oom_count,
            "ALTO — 18 projetos recuperaveis imediatamente",
            "run_sf110_mutation_testing.py --retry-oom --jvm-max-heap 2048m --resume",
        ),
        (
            "Retry compile (--retry-compile)",
            compile_count,
            "MEDIO — depende de compatibilidade Java 1.7/1.6",
            "run_sf110_mutation_testing.py --retry-compile --resume",
        ),
        (
            "Sem limite de classes (--max-classes 0)",
            len([r for r in passed if float(r["score"]) < 20]),
            "MEDIO — melhora projetos com no_coverage alto",
            "run_sf110_mutation_testing.py --max-classes 0 --resume",
        ),
        (
            "Mais timeout (--timeout-seconds 2400)",
            reasons.get("timeout", 0),
            "BAIXO — so para projetos grandes",
            "run_sf110_mutation_testing.py --timeout-seconds 2400 --resume",
        ),
    ]

    for strategy, n_projects, impact, cmd in plans:
        print(f"  {strategy:<38} {n_projects:>8}  {impact}")
        print(f"    Comando: python python/scripts/{cmd}")
        print()

    print(f"  RECOMENDACAO PRIORITARIA:")
    print(f"  python python/scripts/run_sf110_mutation_testing.py \\")
    print(f"    --test-mode both \\")
    print(f"    --retry-oom \\")
    print(f"    --oom-retry-heap 2048m \\")
    print(f"    --oom-retry-classes 10 \\")
    print(f"    --retry-compile \\")
    print(f"    --jvm-max-heap 1536m \\")
    print(f"    --resume \\")
    print(f"    --output-root outputs/mutation_experiments/sf110_improved")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse SF110 mutation testing results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=str,
        default="outputs/mutation_experiments/sf110_full_optimized",
        help="Directory containing sf110_mutation_results.csv and sf110_mutation_summary.json",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Show full project lists per failure category",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERRO] Directory not found: {results_dir}")
        sys.exit(1)

    rows, summary = _load(results_dir)

    print(SEP)
    print(f"  SF110 MUTATION TESTING — ANALISE DE RESULTADOS")
    print(f"  Dir: {results_dir}")
    print(SEP)

    _print_summary(rows, summary)
    _print_failures(rows, full=args.full)
    _print_quick_wins(rows, full=args.full)
    _print_top_bottom(rows, n=10)
    _print_improvement_plan(rows)


if __name__ == "__main__":
    main()
