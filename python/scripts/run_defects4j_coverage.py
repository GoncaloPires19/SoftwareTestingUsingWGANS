#!/usr/bin/env python3
"""
Defects4J line/branch/method coverage harness.

run_defects4j_mutation_testing.py gives mutation score but no traditional
coverage. `defects4j coverage` runs the developer test suite under Cobertura
and reports line + branch ("condition") coverage for the classes modified by
the bug fix; method coverage isn't printed directly but is derivable from
coverage.xml (fraction of <method> elements with line-rate > 0).

Output schema uses (project, bug_id, class_name) so it can be joined with
bridge_dataset_d4j.csv (from run_defects4j_mutation_testing.py) for combined
analysis. Same infra as that script (process-group timeout kill, checkpoint/
resume, single class per bug from classes.modified).

Must run inside WSL/Linux. Typical invocation:
    wsl -e bash -lc '
        export PATH=$PATH:$HOME/tools/defects4j/framework/bin
        python3 "python/scripts/run_defects4j_coverage.py" \
            --output-root outputs/mutation_experiments/defects4j --workers 1
    '
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
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

ALL_PROJECTS = [
    "Chart", "Cli", "Closure", "Codec", "Collections", "Compress", "Csv", "Gson",
    "JacksonCore", "JacksonDatabind", "JacksonXml", "Jsoup", "JxPath", "Lang",
    "Math", "Mockito", "Time",
]


def _run(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
    # See run_defects4j_mutation_testing.py: `defects4j`/ant fork JVM children;
    # run in its own process group so the whole tree can be killed on timeout.
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


def _parse_coverage_xml(path: Path, modified_classes: set[str]) -> list[dict]:
    """One row per <class> in coverage.xml that matches a class the bug fix modified."""
    tree = ET.parse(path)
    rows = []
    for cls in tree.getroot().iter("class"):
        fqcn = cls.get("name", "")
        if fqcn not in modified_classes:
            continue
        lines_total = lines_covered = branches_total = branches_covered = 0
        methods_total = methods_covered = 0
        for method in cls.iter("method"):
            methods_total += 1
            method_covered = False
            for line in method.iter("line"):
                lines_total += 1
                hits = int(line.get("hits", "0"))
                if hits > 0:
                    lines_covered += 1
                    method_covered = True
                if line.get("branch") == "true":
                    cc = line.get("condition-coverage", "")
                    m = re.search(r"\((\d+)/(\d+)\)", cc)
                    if m:
                        branches_covered += int(m.group(1))
                        branches_total += int(m.group(2))
            if method_covered:
                methods_covered += 1
        rows.append({
            "class_name": fqcn,
            "lines_total": lines_total, "lines_covered": lines_covered,
            "branches_total": branches_total, "branches_covered": branches_covered,
            "methods_total": methods_total, "methods_covered": methods_covered,
        })
    return rows


@dataclass
class BugResult:
    project: str
    bug_id: str
    status: str  # passed | failed
    rows: list[dict] = field(default_factory=list)
    error: str = ""
    failure_reason: str = ""
    runtime_seconds: float = 0.0


def _process_bug(
    defects4j_bin: str, work_root: Path, pid: str, bug_id: str,
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
        modified_classes = {c.strip() for c in out.splitlines() if c.strip() and "Running ant" not in c}
        if not modified_classes:
            return BugResult(
                project=pid, bug_id=bug_id, status="failed", failure_reason="no_modified_classes",
                error=(out + err)[-1000:], runtime_seconds=round(time.time() - start, 2),
            )

        code, out, err = _run([defects4j_bin, "coverage"], cwd=workdir, timeout=timeout)
        coverage_xml = workdir / "coverage.xml"
        if code != 0 or not coverage_xml.exists():
            combined = (out + err).lower()
            if "compil" in combined:
                reason = "compile_error"
            elif "timeout" in combined or code == 124:
                reason = "timeout"
            else:
                reason = "coverage_failed"
            return BugResult(
                project=pid, bug_id=bug_id, status="failed", failure_reason=reason,
                error=(out + err)[-1500:], runtime_seconds=round(time.time() - start, 2),
            )

        rows = _parse_coverage_xml(coverage_xml, modified_classes)
        if not rows:
            return BugResult(
                project=pid, bug_id=bug_id, status="failed", failure_reason="no_class_rows",
                runtime_seconds=round(time.time() - start, 2),
            )
        for r in rows:
            r["project"] = pid
            r["bug_id"] = bug_id

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


_FIELDNAMES = [
    "project", "class_name", "bug_id",
    "lines_total", "lines_covered", "line_coverage",
    "branches_total", "branches_covered", "branch_coverage",
    "methods_total", "methods_covered", "method_coverage",
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


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            out["line_coverage"] = round(r["lines_covered"] / r["lines_total"], 4) if r["lines_total"] else 0.0
            out["branch_coverage"] = round(r["branches_covered"] / r["branches_total"], 4) if r["branches_total"] else 0.0
            out["method_coverage"] = round(r["methods_covered"] / r["methods_total"], 4) if r["methods_total"] else 0.0
            writer.writerow({k: out[k] for k in _FIELDNAMES})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Defects4J (Cobertura) line/branch/method coverage across bugs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--defects4j-bin", type=str,
                         default=str(Path.home() / "tools/defects4j/framework/bin/defects4j"))
    parser.add_argument("--d4j-home", type=str, default=str(Path.home() / "tools/defects4j"))
    parser.add_argument("--work-root", type=str, default=str(Path.home() / "d4j_work/coverage"))
    parser.add_argument("--output-root", type=str, default="outputs/mutation_experiments/defects4j")
    parser.add_argument("--projects", type=str, default=",".join(ALL_PROJECTS))
    parser.add_argument("--max-bugs-per-project", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()

    d4j_home = Path(args.d4j_home)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_root / "checkpoint_coverage.json"
    csv_path = output_root / "coverage_dataset_d4j.csv"
    failures_csv_path = output_root / "coverage_failures.csv"
    summary_json_path = output_root / "coverage_summary.json"

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
        _write_csv(all_rows, csv_path)
        with open(failures_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["project", "bug_id", "failure_reason", "error"])
            writer.writeheader()
            writer.writerows(all_failures)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_bug, args.defects4j_bin, work_root, pid, bid,
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

    total_lines_cov = sum(r["lines_covered"] for r in all_rows)
    total_lines_tot = sum(r["lines_total"] for r in all_rows)
    total_branches_cov = sum(r["branches_covered"] for r in all_rows)
    total_branches_tot = sum(r["branches_total"] for r in all_rows)
    total_methods_cov = sum(r["methods_covered"] for r in all_rows)
    total_methods_tot = sum(r["methods_total"] for r in all_rows)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "projects": projects,
        "total_bugs_attempted": total,
        "classes_extracted": len(all_rows),
        "bugs_failed": len(all_failures),
        "line_coverage_global": round(total_lines_cov / total_lines_tot * 100, 1) if total_lines_tot else 0.0,
        "branch_coverage_global": round(total_branches_cov / total_branches_tot * 100, 1) if total_branches_tot else 0.0,
        "method_coverage_global": round(total_methods_cov / total_methods_tot * 100, 1) if total_methods_tot else 0.0,
        "failure_reason_breakdown": {
            reason: sum(1 for f in all_failures if f["failure_reason"] == reason)
            for reason in {f["failure_reason"] for f in all_failures}
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print(f"Classes: {len(all_rows)} | Falhas: {len(all_failures)}")
    print(f"Line coverage: {summary['line_coverage_global']}% | "
          f"Branch coverage: {summary['branch_coverage_global']}% | "
          f"Method coverage: {summary['method_coverage_global']}%")
    print(f"CSV: {csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
