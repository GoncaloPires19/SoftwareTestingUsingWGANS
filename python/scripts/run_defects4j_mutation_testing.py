#!/usr/bin/env python3
"""
Defects4J mutation testing + structural metrics harness (cross-benchmark for SF110).

Why this script exists:
- The SF110 testability predictor (train_testability_predictor.py) was trained and
  validated only within SF110 (leave-projects-out). To claim the structural signal
  generalises, we need a genuinely different benchmark: Defects4J (17 mature
  real-world projects, real bugs, real developer test suites — not EvoSuite).
- Defects4J ships its OWN mutation testing tool (Major), reachable via
  `defects4j mutation` once a bug is checked out. This replaces the custom
  PIT/Maven wrapper that run_sf110_mutation_testing.py needed for Ant-based SF110
  projects — Defects4J's `defects4j` CLI already handles checkout/build/mutate.

Must run inside WSL/Linux (defects4j is Perl/Unix-only). Typical invocation:
    wsl -e bash -lc '
        export PATH=$PATH:$HOME/tools/defects4j/framework/bin
        python3 "python/scripts/run_defects4j_mutation_testing.py" \
            --output-root outputs/mutation_experiments/defects4j --workers 4
    '

Output schema matches outputs/mutation_experiments/bridge_dataset_111.csv exactly
(same FEATURES order from sf110_lib.py) so the SF110-trained pickle model can be
applied to this dataset with zero adaptation.

Strategy per bug (fixed revision, one row per class modified by the fix):
1) `defects4j checkout -p <PID> -v <bid>f -w <workdir>` (native Linux fs scratch).
2) `defects4j mutation` (bundles ant compile+mutate+test internally).
3) Parse summary.csv / kill.csv / mutants.log -> per-class killed/survived/no_coverage.
4) `defects4j export -p classes.modified` + `dir.src.classes` -> locate .java source.
5) Extract structural features via the same regex extractor as extract_class_metrics.py.
6) Append row to the results CSV + checkpoint (resumable), then delete the workdir.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

ALL_PROJECTS = [
    "Chart", "Cli", "Closure", "Codec", "Collections", "Compress", "Csv", "Gson",
    "JacksonCore", "JacksonDatabind", "JacksonXml", "Jsoup", "JxPath", "Lang",
    "Math", "Mockito", "Time",
]

# --- Structural feature regexes (kept identical to extract_class_metrics.py) ---
_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)
_METHOD_LIKE_RE = re.compile(r"\b(\w+)\s*\([^()]*\)\s*(?:throws[\w,.\s]+?)?\{")
_CONTROL_KEYWORDS = {"if", "for", "while", "switch", "catch", "synchronized"}
_BRANCH_KEYWORDS = re.compile(r"\b(if|for|while|case|catch)\b")
_BOOL_OPS = re.compile(r"(&&|\|\||\?)")
_IO_IMPORT_RE = re.compile(r"import\s+(java\.net\.|java\.io\.|javax\.swing|java\.awt|java\.rmi|java\.sql\.)")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_UNTESTABLE_SUPERCLASSES = re.compile(
    r"\bextends\s+(JFrame|JPanel|JDialog|JApplet|Applet|Frame|Thread|Window|"
    r"JComponent|Container|TimerTask|Thread|HttpServlet|JWindow)\b"
)
_IMPLEMENTS_RUNNABLE = re.compile(r"\bimplements\s+[\w,\s]*\b(Runnable|ActionListener|MouseListener|WindowListener)\b")
_ABSTRACT_RE = re.compile(r"\b(abstract)\s+(class|interface)\b|\binterface\s+\w+")
_STATIC_METHOD_RE = re.compile(r"\bstatic\s+[\w<>\[\].]+\s+\w+\s*\([^()]*\)\s*\{")
_DEFAULT_CTOR_HINT = re.compile(r"\bpublic\s+(\w+)\s*\(\s*\)")
_IMPORT_TYPE_RE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", re.MULTILINE)
_CTOR_PARAMS_RE = re.compile(r"\b(?:public|protected|private)\s+\w+\s*\(([^()]*)\)\s*(?:throws[\w,.\s]+?)?\{")

# kill.csv status -> outcome bucket (mirrors PIT's KILLED/SURVIVED/NO_COVERAGE)
_KILLED_KILL_STATUSES = {"FAIL", "TIME", "EXC"}
_SURVIVED_KILL_STATUSES = {"LIVE"}
_NO_COVERAGE_KILL_STATUSES = {"UNCOV"}


def _strip_noise(source: str) -> str:
    source = _BLOCK_COMMENT_RE.sub(" ", source)
    source = _LINE_COMMENT_RE.sub(" ", source)
    source = _STRING_RE.sub('""', source)
    return source


def _count_loc(raw_source: str) -> int:
    no_block = _BLOCK_COMMENT_RE.sub("", raw_source)
    loc = 0
    for line in no_block.splitlines():
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("*") or s.startswith("/*"):
            continue
        loc += 1
    return loc


def _extract_structural_features(java_file: Path) -> dict | None:
    """Same feature set/semantics as extract_class_metrics.py's _extract_file_metrics."""
    try:
        raw = java_file.read_text(encoding="ISO-8859-1", errors="ignore")
    except OSError:
        return None

    clean = _strip_noise(raw)

    method_like = _METHOD_LIKE_RE.findall(clean)
    methods = sum(1 for name in method_like if name not in _CONTROL_KEYWORDS)
    branches = len(_BRANCH_KEYWORDS.findall(clean)) + len(_BOOL_OPS.findall(clean))
    statements = clean.count(";")
    loc = _count_loc(raw)
    has_io = 1 if _IO_IMPORT_RE.search(raw) else 0

    n_fields = len(re.findall(
        r"\b(?:private|protected|public)\s+(?:static\s+|final\s+)*[\w<>\[\].]+\s+\w+\s*(?:=|;)",
        clean,
    ))

    extends_framework = 1 if _UNTESTABLE_SUPERCLASSES.search(clean) else 0
    implements_callback = 1 if _IMPLEMENTS_RUNNABLE.search(clean) else 0
    is_abstract = 1 if _ABSTRACT_RE.search(clean) else 0
    n_static_methods = len(_STATIC_METHOD_RE.findall(clean))
    has_default_ctor = 1 if _DEFAULT_CTOR_HINT.search(clean) else 0

    imports = _IMPORT_TYPE_RE.findall(raw)
    non_jdk = [i for i in imports if not i.startswith(("java.", "javax."))]
    n_imports = len(imports)
    n_external_deps = len(set(non_jdk))

    max_ctor_params = 0
    for params in _CTOR_PARAMS_RE.findall(clean):
        params = params.strip()
        if params:
            max_ctor_params = max(max_ctor_params, params.count(",") + 1)

    static_ratio = round(n_static_methods / methods, 3) if methods else 0.0

    return {
        "methods": methods,
        "branches": branches,
        "statements": statements,
        "loc": loc,
        "n_fields": n_fields,
        "has_io": has_io,
        "extends_framework": extends_framework,
        "implements_callback": implements_callback,
        "is_abstract": is_abstract,
        "n_static_methods": n_static_methods,
        "static_ratio": static_ratio,
        "has_default_ctor": has_default_ctor,
        "n_imports": n_imports,
        "n_external_deps": n_external_deps,
        "max_ctor_params": max_ctor_params,
    }


@dataclass
class BugResult:
    project: str
    bug_id: str
    status: str  # passed | failed
    rows: list[dict] = field(default_factory=list)
    error: str = ""
    failure_reason: str = ""
    runtime_seconds: float = 0.0


def _run(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
    # `defects4j`/ant fork JVM children (javac, mutation runner); a plain
    # subprocess timeout only kills the immediate shell process, leaving those
    # children running and the output pipes open past the deadline. Run in its
    # own process group so the whole tree can be killed on timeout.
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
    except FileNotFoundError as exc:
        return 127, "", str(exc)

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        return 124, stdout, stderr + "\n[TIMEOUT]"


def _read_active_bugs(d4j_home: Path, pid: str) -> list[str]:
    csv_path = d4j_home / "framework" / "projects" / pid / "active-bugs.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["bug.id"] for row in reader]


def _parse_kill_csv(kill_csv: Path) -> dict[int, str]:
    statuses: dict[int, str] = {}
    if not kill_csv.exists():
        return statuses
    with open(kill_csv, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                statuses[int(row[0])] = row[1].strip().upper()
            except ValueError:
                continue
    return statuses


def _parse_mutants_log(mutants_log: Path) -> dict[int, str]:
    """
    mutant_id -> top-level FQCN, parsed from Major's
    `id:OP:...:FQCN@MethodSignature(params):line:...` format.

    Field 4 (0-based, colon-split) is `ClassName@MethodSignature` — the method
    signature itself never contains ':', so the split index is stable. Strip the
    '@method(...)' suffix and any inner-class '$Name' so mutants inside nested/
    inner classes are attributed to the enclosing top-level .java file (which is
    what `classes.modified` reports and what structural features are extracted
    from).
    """
    mapping: dict[int, str] = {}
    if not mutants_log.exists():
        return mapping
    with open(mutants_log, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split(":")
            if len(parts) < 5:
                continue
            try:
                mutant_id = int(parts[0])
            except ValueError:
                continue
            class_locator = parts[4].split("@")[0].split("$")[0]
            mapping[mutant_id] = class_locator
    return mapping


def _process_bug(
    defects4j_bin: str, d4j_home: Path, work_root: Path, pid: str, bug_id: str,
    timeout: int, keep_workdir: bool,
) -> BugResult:
    start = time.time()
    workdir = work_root / f"{pid}_{bug_id}f"
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)

    try:
        code, out, err = _run(
            [defects4j_bin, "checkout", "-p", pid, "-v", f"{bug_id}f", "-w", str(workdir)],
            cwd=work_root, timeout=timeout,
        )
        if code != 0:
            return BugResult(
                project=pid, bug_id=bug_id, status="failed", failure_reason="checkout_failed",
                error=(out + err)[-1500:], runtime_seconds=round(time.time() - start, 2),
            )

        code, out, err = _run(
            [defects4j_bin, "export", "-p", "classes.modified"], cwd=workdir, timeout=120,
        )
        modified_classes = [c.strip() for c in out.splitlines() if c.strip() and "Running ant" not in c]
        if not modified_classes:
            return BugResult(
                project=pid, bug_id=bug_id, status="failed", failure_reason="no_modified_classes",
                error=(out + err)[-1000:], runtime_seconds=round(time.time() - start, 2),
            )

        code, out, err = _run(
            [defects4j_bin, "export", "-p", "dir.src.classes"], cwd=workdir, timeout=120,
        )
        src_dir_lines = [c.strip() for c in out.splitlines() if c.strip() and "Running ant" not in c]
        src_dir = src_dir_lines[0] if src_dir_lines else "src/main/java"

        code, out, err = _run([defects4j_bin, "mutation"], cwd=workdir, timeout=timeout)
        summary_csv = workdir / "summary.csv"
        if code != 0 or not summary_csv.exists():
            combined = (out + err).lower()
            # Check the unambiguous timeout signal (our own [TIMEOUT] marker / exit
            # code 124) before sniffing for "compil" — a killed-mid-compile process
            # still has "Compiling..." in its residual output, which would otherwise
            # misclassify a timeout as a compile_error.
            if code == 124 or "timeout" in combined:
                reason = "timeout"
            elif "compil" in combined:
                reason = "compile_error"
            else:
                reason = "mutation_failed"
            return BugResult(
                project=pid, bug_id=bug_id, status="failed", failure_reason=reason,
                error=(out + err)[-1500:], runtime_seconds=round(time.time() - start, 2),
            )

        kill_statuses = _parse_kill_csv(workdir / "kill.csv")
        mutant_classes = _parse_mutants_log(workdir / "mutants.log")

        rows: list[dict] = []
        for fqcn in modified_classes:
            mutant_ids = [mid for mid, cls in mutant_classes.items() if cls == fqcn]
            if not mutant_ids:
                continue
            killed = survived = no_coverage = 0
            for mid in mutant_ids:
                status = kill_statuses.get(mid, "")
                if status in _KILLED_KILL_STATUSES:
                    killed += 1
                elif status in _SURVIVED_KILL_STATUSES:
                    survived += 1
                elif status in _NO_COVERAGE_KILL_STATUSES:
                    no_coverage += 1
            relevant = killed + survived + no_coverage
            if relevant == 0:
                continue

            src_file = workdir / src_dir / (fqcn.replace(".", "/") + ".java")
            features = _extract_structural_features(src_file)
            if features is None:
                continue

            coverage = (killed + survived) / relevant
            mutation_score = killed / relevant
            test_strength = killed / (killed + survived) if (killed + survived) else 0.0

            rows.append({
                "project": pid,
                "class_name": fqcn,
                "bug_id": bug_id,
                **features,
                "n_mutations": relevant,
                "coverage": round(coverage, 4),
                "mutation_score": round(mutation_score, 4),
                "test_strength": round(test_strength, 4),
                "testable": int(coverage >= 0.5),
            })

        if not rows:
            return BugResult(
                project=pid, bug_id=bug_id, status="failed", failure_reason="no_class_rows",
                runtime_seconds=round(time.time() - start, 2),
            )

        return BugResult(
            project=pid, bug_id=bug_id, status="passed", rows=rows,
            runtime_seconds=round(time.time() - start, 2),
        )
    except Exception as exc:  # defensive: never let one bug crash the whole run
        return BugResult(
            project=pid, bug_id=bug_id, status="failed", failure_reason="unhandled_exception",
            error=repr(exc), runtime_seconds=round(time.time() - start, 2),
        )
    finally:
        if not keep_workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


_BRIDGE_FIELDNAMES = [
    "project", "class_name", "bug_id",
    "methods", "branches", "statements", "loc", "n_fields", "has_io",
    "extends_framework", "implements_callback", "is_abstract",
    "n_static_methods", "static_ratio", "has_default_ctor",
    "n_imports", "n_external_deps", "max_ctor_params", "n_mutations",
    "coverage", "mutation_score", "test_strength", "testable",
]


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"processed": [], "rows": [], "failures": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_bridge_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_BRIDGE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Defects4J (Major) mutation testing across all bugs, "
                    "producing a bridge_dataset-compatible CSV for cross-benchmark validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--defects4j-bin", type=str,
                         default=str(Path.home() / "tools/defects4j/framework/bin/defects4j"))
    parser.add_argument("--d4j-home", type=str, default=str(Path.home() / "tools/defects4j"))
    parser.add_argument("--work-root", type=str, default=str(Path.home() / "d4j_work/run"))
    parser.add_argument("--output-root", type=str, default="outputs/mutation_experiments/defects4j")
    parser.add_argument("--projects", type=str, default=",".join(ALL_PROJECTS))
    parser.add_argument("--max-bugs-per-project", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()

    d4j_home = Path(args.d4j_home)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_root / "checkpoint.json"
    bridge_csv_path = output_root / "bridge_dataset_d4j.csv"
    failures_csv_path = output_root / "failures.csv"
    summary_json_path = output_root / "summary.json"

    checkpoint = _load_checkpoint(checkpoint_path) if args.resume else {"processed": [], "rows": [], "failures": []}
    processed: set[str] = set(checkpoint["processed"])
    all_rows: list[dict] = checkpoint["rows"]
    all_failures: list[dict] = checkpoint["failures"]

    projects = [p.strip() for p in args.projects.split(",") if p.strip()]
    worklist: list[tuple[str, str]] = []
    for pid in projects:
        bug_ids = _read_active_bugs(d4j_home, pid)
        if args.max_bugs_per_project > 0:
            bug_ids = bug_ids[: args.max_bugs_per_project]
        for bid in bug_ids:
            key = f"{pid}_{bid}"
            if key in processed:
                continue
            worklist.append((pid, bid))

    total = len(worklist) + len(processed)
    print(f"Worklist: {len(worklist)} bugs pendentes ({len(processed)} ja no checkpoint, {total} total)")

    lock = Lock()
    done_count = len(processed)

    def _persist() -> None:
        _save_checkpoint(checkpoint_path, {
            "processed": sorted(processed), "rows": all_rows, "failures": all_failures,
        })
        _write_bridge_csv(all_rows, bridge_csv_path)
        with open(failures_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["project", "bug_id", "failure_reason", "error"])
            writer.writeheader()
            writer.writerows(all_failures)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_bug, args.defects4j_bin, d4j_home, work_root, pid, bid,
                args.timeout_seconds, args.keep_workdir,
            ): (pid, bid)
            for pid, bid in worklist
        }
        for future in as_completed(futures):
            pid, bid = futures[future]
            result = future.result()
            with lock:
                processed.add(f"{pid}_{bid}")
                done_count += 1
                if result.status == "passed":
                    all_rows.extend(result.rows)
                    print(f"[{done_count}/{total}] PASS {pid}-{bid}: {len(result.rows)} classe(s), "
                          f"{result.runtime_seconds}s")
                else:
                    all_failures.append({
                        "project": pid, "bug_id": bid,
                        "failure_reason": result.failure_reason, "error": result.error,
                    })
                    print(f"[{done_count}/{total}] FAIL {pid}-{bid}: {result.failure_reason} "
                          f"({result.runtime_seconds}s)")
                _persist()

    n_testable = sum(r["testable"] for r in all_rows)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "projects": projects,
        "total_bugs_attempted": total,
        "classes_extracted": len(all_rows),
        "bugs_failed": len(all_failures),
        "testable_classes": n_testable,
        "testable_pct": round(n_testable / len(all_rows) * 100, 1) if all_rows else 0.0,
        "failure_reason_breakdown": {
            reason: sum(1 for f in all_failures if f["failure_reason"] == reason)
            for reason in {f["failure_reason"] for f in all_failures}
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print(f"Classes extraidas: {len(all_rows)} | Falhas: {len(all_failures)}")
    print(f"Testaveis (coverage>=50%): {n_testable} ({summary['testable_pct']}%)")
    print(f"Bridge CSV: {bridge_csv_path}")
    print(f"Summary:    {summary_json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
