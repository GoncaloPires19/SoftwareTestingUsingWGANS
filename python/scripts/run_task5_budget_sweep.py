#!/usr/bin/env python3
"""
Tarefa 5 — cobertura e deteção de defeitos versus ORÇAMENTO DE TESTES.

The main experiment fixes the budget at 200 inputs per arm. That single point
hides the effect that decides how the generated arms should be read, because
the arms do not scale alike:

    CORPUS  is capped by its own diversity. The corpus holds 121 distinct
            inputs, so resampling it can never yield more than 121 however
            large the budget; at 200 draws it already repeats itself down to
            about 76 distinct values. Its curve must flatten.
    WGAN
    KDE     draw from a continuous density and keep producing novel inputs, so
            their curves should keep climbing past that ceiling.
    RANDOM  covers the range but rarely lands on the boundary values that drive
            branches, so it should trail throughout.

If a generative model is worth its cost, this is where it shows: not in beating
a fixed corpus at a budget the corpus can still fill, but in continuing to
improve after the corpus has run out of distinct material. Small budgets matter
too, and for the opposite reason — the practical appeal of generated tests is
getting good coverage cheaply.

The number of distinct inputs actually emitted is recorded per point, so the
saturation can be read off the results rather than assumed.

Efficiency note: each generator is fitted once per seed and `max(budgets)`
inputs are drawn from it; each budget N then takes the first N. Because the
samples are i.i.d., a prefix of a larger draw is distributed identically to an
independent draw of size N, so this is equivalent to regenerating per budget at
a fraction of the cost.

Usage:
    python python/scripts/run_task5_budget_sweep.py --seeds 5 --budgets 5,10,20,50,100,200
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
MVN = shutil.which("mvn") or "mvn"
JACOCO = "org.jacoco:jacoco-maven-plugin:0.8.11"
TARGET_CLASS = "dataloader.MathOperations"
SEP = "=" * 78

VEC_COLS = ["arr_len"] + [f"v{i}" for i in range(8)] + ["s0", "s1", "s2"]

# (chave do gerador, classe de testes emitida, rotulo no CSV)
ARMS = [
    ("wgan", "MathOperationsWganTests", "WGAN"),
    ("random", "MathOperationsRandomTests", "RANDOM"),
    ("corpus", "MathOperationsCorpusTests", "CORPUS"),
    ("kde", "MathOperationsKdeTests", "KDE"),
]

FIELDNAMES = [
    "arm", "seed", "budget", "n_distinct", "n_tests",
    "line_coverage", "branch_coverage", "method_coverage",
    "killed", "survived", "no_coverage", "mutation_score", "test_strength",
]


def _run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
                          timeout=timeout, errors="replace")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _py(script: str, *args: str) -> tuple[int, str]:
    return _run([sys.executable, "-u", str(SCRIPT_DIR / script), *args])


def _parse_jacoco(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row["CLASS"] != "MathOperations":
                continue
            lc, lm = int(row["LINE_COVERED"]), int(row["LINE_MISSED"])
            bc, bm = int(row["BRANCH_COVERED"]), int(row["BRANCH_MISSED"])
            mc, mm = int(row["METHOD_COVERED"]), int(row["METHOD_MISSED"])
            return {
                "line_coverage": round(lc / (lc + lm), 4) if lc + lm else 0.0,
                "branch_coverage": round(bc / (bc + bm), 4) if bc + bm else 0.0,
                "method_coverage": round(mc / (mc + mm), 4) if mc + mm else 0.0,
            }
    return {}


def _parse_pit(path: Path) -> dict:
    if not path.exists():
        return {}
    counts = {"KILLED": 0, "SURVIVED": 0, "NO_COVERAGE": 0, "TIMED_OUT": 0}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            status = row[-2].strip()
            if status in counts:
                counts[status] += 1
    killed = counts["KILLED"] + counts["TIMED_OUT"]
    relevant = killed + counts["SURVIVED"] + counts["NO_COVERAGE"]
    covered = killed + counts["SURVIVED"]

    # Zero mutations means the PIT run failed, not that the suite scored zero;
    # recording it as 0.0 would average a failure in as a real measurement.
    if relevant == 0:
        return {}

    return {
        "killed": killed, "survived": counts["SURVIVED"], "no_coverage": counts["NO_COVERAGE"],
        "mutation_score": round(killed / relevant, 4) if relevant else 0.0,
        "test_strength": round(killed / covered, 4) if covered else 0.0,
    }


def _truncate(src: Path, dst: Path, n: int) -> int:
    """Write the first n input rows of src to dst; return how many are distinct.

    The distinct count is what exposes the CORPUS ceiling: its curve flattens
    once resampling starts returning inputs the suite already has.
    """
    with open(src, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:n]
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VEC_COLS)
        writer.writeheader()
        writer.writerows(rows)
    return len({tuple(r[c] for c in VEC_COLS) for r in rows})


def _measure(arm: str, seed: int, budget: int, test_class: str) -> dict:
    row = {"arm": arm, "seed": seed, "budget": budget}
    short = test_class.replace("dataloader.", "")

    # The agent appends to jacoco.exec by default; a stale exec file would make
    # each measurement report the union of everything measured before it.
    shutil.rmtree(REPO_ROOT / "target" / "site" / "jacoco", ignore_errors=True)
    (REPO_ROOT / "target" / "jacoco.exec").unlink(missing_ok=True)
    _run([MVN, "-q", "-B", f"{JACOCO}:prepare-agent", "test", f"{JACOCO}:report",
          f"-Dtest={short}", "-Dmaven.test.failure.ignore=true", "-DfailIfNoTests=false",
          "-Djacoco.append=false"])
    row.update(_parse_jacoco(REPO_ROOT / "target" / "site" / "jacoco" / "jacoco.csv"))

    shutil.rmtree(REPO_ROOT / "target" / "pit-reports", ignore_errors=True)
    _run([MVN, "-q", "-B", "org.pitest:pitest-maven:mutationCoverage",
          f"-DtargetClasses={TARGET_CLASS}", f"-DtargetTests={test_class}",
          "-DmutationThreshold=0", "-DoutputFormats=CSV", "-DtimestampedReports=false"])
    row.update(_parse_pit(REPO_ROOT / "target" / "pit-reports" / "mutations.csv"))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep test budget for the GAN and RANDOM arms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--budgets", type=str, default="5,10,20,50,100,200")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--out", type=str, default="outputs/task5/budget_sweep.csv")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    max_budget = max(budgets)
    seeds = [args.base_seed + i for i in range(args.seeds)]
    task5 = REPO_ROOT / "outputs" / "task5"

    print(SEP)
    print("  TAREFA 5 — cobertura e detecao vs ORCAMENTO de testes")
    print(SEP)
    print(f"\n  Seeds: {seeds}")
    print(f"  Orcamentos (nr de inputs): {budgets}")

    results: list[dict] = []
    start = time.time()

    for seed in seeds:
        print(f"\n  seed {seed}: a ajustar geradores e tirar {max_budget} inputs ...", flush=True)
        code, out = _py("generate_test_inputs_wgan.py", "--seed", str(seed),
                        "--n-generate", str(max_budget), "--epochs", str(args.epochs))
        if code != 0:
            print(f"    [ERRO] {out[-400:]}")
            continue

        for key, _, _ in ARMS:
            shutil.copy(task5 / f"inputs_{key}.csv", task5 / f"inputs_{key}_full.csv")

        for budget in budgets:
            for key, cls, label in ARMS:
                n_distinct = _truncate(task5 / f"inputs_{key}_full.csv",
                                       task5 / f"inputs_{key}.csv", budget)

                code, out = _run(["java", "-cp", "target/classes;target/test-classes",
                                  "tools.OracleRunner",
                                  f"outputs/task5/inputs_{key}.csv",
                                  f"outputs/task5/oracle_{key}.csv"])
                if code != 0:
                    print(f"    [ERRO] oracle: {out[-300:]}")
                    continue

                code, out = _py("emit_junit_from_oracle.py",
                                "--oracle", f"outputs/task5/oracle_{key}.csv",
                                "--class-name", cls,
                                "--out", f"src/test/java/dataloader/{cls}.java",
                                "--arm", label)
                if code != 0:
                    print(f"    [ERRO] emissao: {out[-300:]}")
                    continue

                code, out = _run([MVN, "-q", "-B", "test-compile", "-DskipTests"])
                if code != 0:
                    print(f"    [ERRO] compilacao: {out[-400:]}")
                    continue

                row = _measure(label, seed, budget, f"dataloader.{cls}")
                row["n_distinct"] = n_distinct
                results.append(row)
                print(f"    budget={budget:>4} {label:<7} distintos={n_distinct:>4} "
                      f"line={row.get('line_coverage', 0):.1%} "
                      f"branch={row.get('branch_coverage', 0):.1%} "
                      f"mut={row.get('mutation_score', 0):.1%}", flush=True)

            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)

    print(f"\n{SEP}")
    print(f"  Concluido em {round(time.time() - start, 1)}s — {len(results)} medicoes")
    print(f"  CSV: {args.out}")
    print(SEP)


if __name__ == "__main__":
    main()
