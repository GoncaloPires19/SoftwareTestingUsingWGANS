#!/usr/bin/env python3
"""
Tarefa 5 — orchestrates the full experiment: does GAN-generated test data
improve code coverage and defect detection?

Six arms, all measured identically against dataloader.MathOperations:

    MANUAL    hand-written suites already in the repo          (fixed)
    EVOSUITE  EvoSuite-derived suite already in the repo       (fixed)
    RANDOM    uniform inputs over the corpus's per-dimension range
    CORPUS    the real corpus resampled with replacement
    KDE       a cross-validated Gaussian KDE fitted to the corpus
    WGAN      the WGAN-GP under evaluation

The four generated arms form a ladder of increasingly demanding baselines, and
each answers a different objection:

    vs RANDOM  did the WGAN-GP learn anything beyond the value range?
    vs CORPUS  is the WGAN-GP worth more than reusing the inputs it trained on?
    vs KDE     does the problem need an adversarial model, or would a density
               model do? (the same question the predictor ladder asked of the
               RandomForest, which turned out to add only 0.007 AUC)

Beating RANDOM alone would be a weak claim, since uniform sampling over a wide
integer range almost never lands on the boundary values that drive branches.

All four generated arms are re-run across seeds so the comparison has a spread
rather than a single point. MANUAL and EVOSUITE are fixed suites, measured once.

Per arm: JaCoCo (line/branch/method coverage), PIT (mutation score, test
strength), a per-mutator breakdown, and the generator's own fidelity/novelty
metrics carried over from the generation step, so distributional fidelity can
be correlated against practical utility.

Usage:
    python python/scripts/run_task5_experiment.py --seeds 15
    python python/scripts/run_task5_experiment.py --seeds 15 --fm-weight 0  # ablacao
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
MVN = shutil.which("mvn") or "mvn"
JACOCO = "org.jacoco:jacoco-maven-plugin:0.8.11"
# Descritores por alvo experimental. Blocos explicitos e nao um sistema
# generico: o caminho mathops sustenta a tese e ja esta validado.
TARGETS = {
    "mathops": {
        "class": "dataloader.MathOperations",
        "oracle": "tools.OracleRunner",
        "suffix": "",                       # MathOperationsWganTests, etc.
        "out_dir": "outputs/task5",
        "has_fixed_arms": True,
    },
    "arith": {
        "class": "dataloader.ArithmeticUtils",
        "oracle": "tools.ArithmeticOracleRunner",
        "suffix": "Arith",                  # ArithmeticWganTests, etc.
        "out_dir": "outputs/task5b",
        # A suite do Commons Math e JUnit 4 e serve so para recolher o corpus;
        # nao entra como braco de referencia, e nao ha EvoSuite para este alvo.
        "has_fixed_arms": False,
    },
}

TARGET_CLASS = TARGETS["mathops"]["class"]

SEP = "=" * 78

MANUAL_TESTS = (
    "dataloader.MathOperationsTests,dataloader.AdvancedTests,"
    "dataloader.ExtendedTests,dataloader.RobustTests,"
    "dataloader.AdvancedValidationTests,dataloader.ExampleTests"
)
EVOSUITE_TESTS = "dataloader.MathOperationsEvoSuiteTests"

# (chave do gerador, nome da classe de testes emitida, rotulo no CSV)
def _arms_for(target: str) -> list[tuple[str, str, str]]:
    """(chave do gerador, classe de testes emitida, rotulo no CSV)."""
    prefixo = "MathOperations" if target == "mathops" else "Arithmetic"
    return [(k, f"{prefixo}{c}Tests", r) for k, c, r in
            [("wgan", "Wgan", "WGAN"), ("random", "Random", "RANDOM"),
             ("corpus", "Corpus", "CORPUS"), ("kde", "Kde", "KDE")]]


GENERATED_ARMS = _arms_for("mathops")

FIELDNAMES = [
    "arm", "seed", "n_tests",
    "lines_covered", "lines_total", "line_coverage",
    "branches_covered", "branches_total", "branch_coverage",
    "methods_covered", "methods_total", "method_coverage",
    "killed", "survived", "no_coverage", "timed_out",
    "mutation_score", "test_strength",
    # metricas do gerador, para cruzar fidelidade com utilidade
    "w1", "n_unique", "nn_dist_median", "n_exact_copies",
]

MUTATOR_FIELDNAMES = ["arm", "seed", "mutator", "killed", "survived", "no_coverage"]


def _run(cmd: list[str], cwd: Path = REPO_ROOT, timeout: int = 3600) -> tuple[int, str]:
    """Correr um subprocesso, tratando o timeout como falha da medicao.

    O timeout e generoso porque mede tempo de relogio: uma medicao do PIT demora
    cerca de um minuto, mas se a maquina suspender a meio o relogio continua a
    andar e qualquer limite razoavel seria excedido. Ja aconteceu duas vezes.

    Devolver um codigo de erro em vez de propagar a excepcao e deliberado. Antes,
    um unico timeout matava a execucao inteira: uma fraccao do regime de poucos
    dados perdeu onze das doze seeds por causa de uma medicao. Agora perde-se a
    seed afectada e o resto continua.

    A saida vai para ficheiro em vez de um pipe, e o timeout mata a arvore toda
    com taskkill /T. As duas coisas resolvem o mesmo impasse, observado a matar
    32 horas da seed 49. O subprocess.run com capture_output abre pipes que os
    netos herdam; ao expirar o timeout ele mata so o filho directo, os JVM do
    PIT ficam orfaos a segurar o pipe, e o communicate() interno bloqueia para
    sempre a espera que feche. Pior, os orfaos continuavam a lancar JVM novos,
    pelo que o impasse se alimentava a si proprio e nada no log o denunciava.
    Sem pipe nao ha nada para esvaziar, e matar a arvore nao deixa orfaos.
    """
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f,
                                stderr=subprocess.STDOUT, text=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=120)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            return -1, f"TIMEOUT apos {timeout}s: {' '.join(cmd[:4])}"
        f.seek(0)
        return code, f.read()


def _py(script: str, *args: str) -> tuple[int, str]:
    return _run([sys.executable, "-u", str(SCRIPT_DIR / script), *args])


def _classpath() -> str:
    """Classpath para invocar o oraculo fora do Maven.

    O segundo alvo depende do commons-math3 para as excepcoes que lanca, pelo
    que target/classes e target/test-classes deixaram de chegar. O ficheiro e
    gerado por dependency:build-classpath no arranque da execucao.
    """
    base = "target/classes;target/test-classes"
    cp = REPO_ROOT / "target" / "cp.txt"
    return f"{base};{cp.read_text(encoding='utf-8').strip()}" if cp.exists() else base


def _parse_jacoco(path: Path) -> dict:
    """Pull the target class's row out of jacoco.csv."""
    if not path.exists():
        return {}
    simples = TARGET_CLASS.rsplit(".", 1)[-1]
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row["CLASS"] != simples:
                continue
            lc, lm = int(row["LINE_COVERED"]), int(row["LINE_MISSED"])
            bc, bm = int(row["BRANCH_COVERED"]), int(row["BRANCH_MISSED"])
            mc, mm = int(row["METHOD_COVERED"]), int(row["METHOD_MISSED"])
            return {
                "lines_covered": lc, "lines_total": lc + lm,
                "line_coverage": round(lc / (lc + lm), 4) if lc + lm else 0.0,
                "branches_covered": bc, "branches_total": bc + bm,
                "branch_coverage": round(bc / (bc + bm), 4) if bc + bm else 0.0,
                "methods_covered": mc, "methods_total": mc + mm,
                "method_coverage": round(mc / (mc + mm), 4) if mc + mm else 0.0,
            }
    return {}


def _parse_pit(path: Path) -> dict:
    """Count mutation outcomes from PIT's mutations.csv."""
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

    # TIMED_OUT counts as killed: the mutant changed observable behaviour.
    killed = counts["KILLED"] + counts["TIMED_OUT"]
    relevant = killed + counts["SURVIVED"] + counts["NO_COVERAGE"]
    covered = killed + counts["SURVIVED"]

    # PIT always finds mutations in this class, so finding none means the run
    # failed and wrote an empty report -- not that the suite scored zero. The
    # two must not be conflated: returning 0.0 here records a failure as a
    # legitimate measurement, and a single such row drags an arm's mean down
    # far enough to reverse the ranking. Leave the fields missing instead, so
    # the analysis skips them via nanmean rather than averaging in a fiction.
    if relevant == 0:
        return {}

    return {
        "killed": killed, "survived": counts["SURVIVED"],
        "no_coverage": counts["NO_COVERAGE"], "timed_out": counts["TIMED_OUT"],
        "mutation_score": round(killed / relevant, 4) if relevant else 0.0,
        "test_strength": round(killed / covered, 4) if covered else 0.0,
    }


def _parse_pit_by_mutator(path: Path, arm: str, seed) -> list[dict]:
    """Break the same PIT report down per mutation operator.

    Lets us ask *which* mutants a generator kills better, which is what links
    this experiment back to the operator curation done on SF110: if the gain
    concentrated in operators that were dropped there, the two results would be
    in tension.
    """
    if not path.exists():
        return []
    acc: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            mutator = row[2].strip().split(".")[-1].replace("Mutator", "")
            status = row[-2].strip()
            rec = acc.setdefault(
                mutator,
                {"arm": arm, "seed": seed, "mutator": mutator,
                 "killed": 0, "survived": 0, "no_coverage": 0},
            )
            if status in ("KILLED", "TIMED_OUT"):
                rec["killed"] += 1
            elif status == "SURVIVED":
                rec["survived"] += 1
            elif status == "NO_COVERAGE":
                rec["no_coverage"] += 1
    return sorted(acc.values(), key=lambda r: r["mutator"])


def _generation_metrics(arm_key: str) -> dict:
    """Fidelity/novelty of the generator that produced this arm's inputs."""
    path = REPO_ROOT / "outputs" / "task5" / "generation_metrics.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("arms", {}).get(arm_key, {})
    except (json.JSONDecodeError, OSError):
        return {}


def _count_tests(target_tests: str) -> int:
    """Read the executed test count from Surefire's own reports.

    Parsing Maven's stdout does not work here: the build runs with -q, which
    suppresses the very summary line the count would come from, so the column
    silently recorded 0 for every arm. Surefire still writes one report per test
    class regardless of -q, and reading those also verifies that the arms really
    are budget-matched rather than assuming it from the input count.
    """
    reports = REPO_ROOT / "target" / "surefire-reports"
    total = 0
    for cls in target_tests.split(","):
        path = reports / f"{cls.strip()}.txt"
        if not path.exists():
            continue
        m = re.search(r"Tests run:\s*(\d+)",
                      path.read_text(encoding="utf-8", errors="replace"))
        if m:
            total += int(m.group(1))
    return total


def _measure(arm: str, seed, target_tests: str) -> tuple[dict, list[dict]]:
    """Run JaCoCo then PIT for one arm; return the summary row and the
    per-mutator breakdown of the same PIT report."""
    row = {"arm": arm, "seed": seed}

    surefire_filter = target_tests.replace("dataloader.", "")

    # --- JaCoCo ---
    # The agent appends to jacoco.exec by default, so a stale exec file makes
    # every arm report the union of all arms measured before it. Delete both the
    # exec data and the rendered report, and pass append=false as a belt-and-braces.
    shutil.rmtree(REPO_ROOT / "target" / "site" / "jacoco", ignore_errors=True)
    (REPO_ROOT / "target" / "jacoco.exec").unlink(missing_ok=True)
    code, out = _run([
        MVN, "-q", "-B", f"{JACOCO}:prepare-agent", "test", f"{JACOCO}:report",
        f"-Dtest={surefire_filter}", "-Dmaven.test.failure.ignore=true",
        "-DfailIfNoTests=false", "-Djacoco.skip=false", "-Djacoco.append=false",
    ])
    row["n_tests"] = _count_tests(target_tests)
    row.update(_parse_jacoco(REPO_ROOT / "target" / "site" / "jacoco" / "jacoco.csv"))

    # --- PIT ---
    shutil.rmtree(REPO_ROOT / "target" / "pit-reports", ignore_errors=True)
    code, out = _run([
        MVN, "-q", "-B", "org.pitest:pitest-maven:mutationCoverage",
        f"-DtargetClasses={TARGET_CLASS}", f"-DtargetTests={target_tests}",
        "-DmutationThreshold=0", "-DoutputFormats=CSV", "-DtimestampedReports=false",
    ])
    pit_csv = REPO_ROOT / "target" / "pit-reports" / "mutations.csv"
    row.update(_parse_pit(pit_csv))

    return row, _parse_pit_by_mutator(pit_csv, arm, seed)


def _first_errors(out: str, n: int = 12) -> str:
    """The FIRST error lines, not the last.

    Printing out[-800:] hid the root cause of a compile failure behind Maven's
    "re-run with -X" epilogue: javac reports the originating error first and the
    cascade after it, so the tail is the least informative part of the log.
    """
    lines = [l for l in out.splitlines() if "ERROR" in l or "error:" in l]
    return "\n".join(lines[:n]) if lines else out[-800:]


class MutantPopulationChanged(RuntimeError):
    """PIT started mutating a different set of mutants part-way through a run.

    The mutation score is killed/total, so a run whose total changes has two
    incompatible denominators and its means are arithmetic on different
    quantities. This already happened once: seeds 42-46 were measured against
    114 mutants and 47-56 against 118, because the first batch ran against
    stale compiled classes. The blended averages went into the dissertation and
    had to be withdrawn.

    Raised rather than warned deliberately. A warning at the top of a two-hour
    log is a warning nobody reads.
    """


def _build_generated_arm(seed: int, n_generate: int, epochs: int,
                         fm_weight: float, conditional: bool = False,
                         corpus_frac: float = 1.0, target: str = "mathops") -> bool:
    """Regenerate inputs for this seed, re-run the oracle, re-emit every suite."""
    t = TARGETS[target]
    extra = ["--conditional"] if conditional else []
    if corpus_frac < 1.0:
        extra += ["--corpus-frac", str(corpus_frac)]
    code, out = _py("generate_test_inputs_wgan.py", "--target", target,
                    "--seed", str(seed), "--n-generate", str(n_generate),
                    "--epochs", str(epochs), "--fm-weight", str(fm_weight), *extra)
    if code != 0:
        print(f"    [ERRO] geracao falhou: {out[-500:]}")
        return False

    for arm_key, cls, _ in GENERATED_ARMS:
        code, out = _run([
            "java", "-cp", _classpath(), t["oracle"],
            f"{t['out_dir']}/inputs_{arm_key}.csv",
            f"{t['out_dir']}/oracle_{arm_key}.csv",
        ])
        if code != 0:
            print(f"    [ERRO] oracle {arm_key} falhou: {out[-500:]}")
            return False

        code, out = _py("emit_junit_from_oracle.py", "--target", target,
                        "--oracle", f"{t['out_dir']}/oracle_{arm_key}.csv",
                        "--class-name", cls,
                        "--out", f"src/test/java/dataloader/{cls}.java",
                        "--arm", arm_key.upper())
        if code != 0:
            print(f"    [ERRO] emissao {arm_key} falhou: {out[-500:]}")
            return False

    # Retried once because this step has a flaky external competitor: the VS Code
    # Java language server also builds into target/, and a concurrent write has
    # been observed to make javac lose classes it had just resolved, producing a
    # "cannot find symbol" that does not reproduce a minute later. One retry
    # costs half a minute; losing a seed costs six.
    for attempt in (1, 2):
        code, out = _run([MVN, "-q", "-B", "test-compile", "-DskipTests"])
        if code == 0:
            return True
        if attempt == 1:
            print("    [aviso] compilacao falhou, a repetir uma vez ...", flush=True)
            time.sleep(5)
    print(f"    [ERRO] compilacao falhou duas vezes:\n{_first_errors(out)}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Tarefa 5 experiment across all arms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-generate", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--fm-weight", type=float, default=0.3,
                        help="Feature matching weight; pass 0 for the ablation")
    parser.add_argument("--conditional", action="store_true",
                        help="Condition the WGAN-GP on arr_len. Leaves the RANDOM, CORPUS "
                             "and KDE arms untouched, so they act as the regression control.")
    parser.add_argument("--corpus-frac", type=float, default=1.0,
                        help="Train every generator on this fraction of the corpus. The "
                             "CORPUS arm is restricted to the same subset, so all arms see "
                             "equal data. Below 1.0 this is the low-data regime, the only "
                             "setting where CORPUS loses its ceiling advantage.")
    parser.add_argument("--target", type=str, default="mathops",
                        choices=sorted(TARGETS),
                        help="Alvo experimental. arith usa o ArithmeticUtils do "
                             "Commons Math, com corpus recolhido por instrumentacao.")
    parser.add_argument("--out", type=str, default="outputs/task5/results.csv")
    parser.add_argument("--skip-fixed", action="store_true",
                        help="Skip the MANUAL/EVOSUITE arms (they don't vary per seed)")
    args = parser.parse_args()

    # Fixar o alvo antes de qualquer medicao.
    global TARGET_CLASS, GENERATED_ARMS
    alvo = TARGETS[args.target]
    TARGET_CLASS = alvo["class"]
    GENERATED_ARMS = _arms_for(args.target)
    if not alvo["has_fixed_arms"]:
        args.skip_fixed = True

    results: list[dict] = []
    per_mutator: list[dict] = []
    start = time.time()
    # Fixed on the first measurement; every later one must agree with it.
    population: dict[str, int] = {}

    def _check_population(row: dict) -> None:
        try:
            total = int(row["killed"]) + int(row["survived"]) + int(row["no_coverage"])
        except (KeyError, TypeError, ValueError):
            return
        if "n" not in population:
            population["n"] = total
            print(f"  Populacao de mutantes: {total}")
        elif total != population["n"]:
            raise MutantPopulationChanged(
                f"PIT reportou {total} mutantes em {row.get('arm')}/{row.get('seed')}, "
                f"mas {population['n']} nas medicoes anteriores. O mutation score e "
                f"killed/total, por isso estas medicoes nao sao comparaveis. Causa "
                f"habitual: classes compiladas em cache. Corrigir com 'mvn clean' e "
                f"repetir a execucao inteira."
            )

    out_path = Path(args.out)
    mut_path = out_path.with_name(out_path.stem + "_by_mutator.csv")

    def _flush() -> None:
        """Persist after every seed: a 15-seed run is hours long and an
        interrupted one should still leave usable partial results."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        with open(mut_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MUTATOR_FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            w.writerows(per_mutator)

    print(SEP)
    print("  TAREFA 5 — impacto dos dados de teste gerados por GAN")
    print(SEP)
    print(f"\n  Bracos gerados : {', '.join(a[2] for a in GENERATED_ARMS)}")
    print(f"  fm_weight      : {args.fm_weight}"
          f"{'   (ABLACAO: feature matching desligado)' if args.fm_weight == 0 else ''}")
    if args.corpus_frac < 1.0:
        print(f"  corpus_frac    : {args.corpus_frac}   (REGIME DE POUCOS DADOS)")

    # Stale .class files are what produced the 114-vs-118 split, and they are
    # invisible until the totals are compared after the fact. Half a minute of
    # rebuild is cheap against a run that has to be thrown away.
    #
    # The rebuild is not optional after the clean: _build_generated_arm invokes
    # tools.OracleRunner from target/test-classes before anything compiles, so a
    # bare clean leaves the very next step with no classes to load.
    print("\n  mvn clean + test-compile (o PIT tem de mutilar bytecode actual) ...",
          flush=True)
    code, out = _run([MVN, "-q", "-B", "clean", "test-compile", "-DskipTests"])
    if code == 0:
        # Escrito para target/cp.txt e usado por _classpath() ao invocar o
        # oraculo fora do Maven.
        _run([MVN, "-q", "-B", "dependency:build-classpath",
              "-Dmdep.outputFile=target/cp.txt", "-Dmdep.includeScope=test"])
    if code != 0:
        print("  [ERRO] mvn clean/test-compile falhou:")
        print(_first_errors(out))
        sys.exit(1)

    if not args.skip_fixed:
        print("\n  Arms fixos (medidos uma vez)")
        for arm, tests in (("MANUAL", MANUAL_TESTS), ("EVOSUITE", EVOSUITE_TESTS)):
            print(f"    {arm} ...", flush=True)
            row, muts = _measure(arm, "-", tests)
            _check_population(row)
            results.append(row)
            per_mutator.extend(muts)
            print(f"      line={row.get('line_coverage', 0):.1%} "
                  f"branch={row.get('branch_coverage', 0):.1%} "
                  f"mut={row.get('mutation_score', 0):.1%} "
                  f"strength={row.get('test_strength', 0):.1%}")

    seeds = [args.base_seed + i for i in range(args.seeds)]
    print(f"\n  Arms gerados ({len(seeds)} seeds: {seeds})")
    for seed in seeds:
        print(f"\n    seed {seed}: a gerar inputs + oraculo + suites ...", flush=True)
        if not _build_generated_arm(seed, args.n_generate, args.epochs, args.fm_weight,
                                    args.conditional, args.corpus_frac, args.target):
            continue
        # As comparacoes sao emparelhadas por seed, por isso uma seed so entra se
        # TODOS os bracos foram medidos. Guardar uma seed incompleta enviesaria
        # qualquer media que a apanhasse.
        do_seed, muts_seed, falhou = [], [], None
        for arm_key, cls, label in GENERATED_ARMS:
            row, muts = _measure(label, seed, f"dataloader.{cls}")
            if not row.get("mutation_score") and not row.get("killed"):
                falhou = label
                break
            _check_population(row)
            row.update(_generation_metrics(arm_key))
            do_seed.append(row)
            muts_seed.extend(muts)
            print(f"      {label:<7} line={row.get('line_coverage', 0):.1%} "
                  f"branch={row.get('branch_coverage', 0):.1%} "
                  f"mut={row.get('mutation_score', 0):.1%} "
                  f"strength={row.get('test_strength', 0):.1%} "
                  f"W1={row.get('w1', float('nan')):.3f}", flush=True)

        if falhou:
            print(f"      [AVISO] seed {seed} descartada: medicao de {falhou} falhou",
                  flush=True)
            continue
        results.extend(do_seed)
        per_mutator.extend(muts_seed)
        _flush()

    _flush()
    elapsed = round(time.time() - start, 1)
    print(f"\n{SEP}")
    print(f"  Concluido em {elapsed}s — {len(results)} medicoes")
    print(f"  CSV          : {out_path}")
    print(f"  Por mutator  : {mut_path}")
    print(SEP)


if __name__ == "__main__":
    main()
