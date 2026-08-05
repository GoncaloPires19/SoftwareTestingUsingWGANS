#!/usr/bin/env python3
"""
SF110 line/branch/method coverage via JaCoCo, reusing the already-built Maven
projects from the PIT mutation testing run (outputs/mutation_experiments/
sf110_full111/workdir/<project>/, each with a working pom.xml + target/).

PIT (run_sf110_mutation_testing.py) only gives line coverage + mutation
coverage (extracted separately by extract_pit_line_coverage.py). JaCoCo adds
real branch and method coverage, which PIT doesn't report.

Runs entirely natively on Windows (no WSL needed — this SF110 pipeline was
always Windows-native, unlike Defects4J which requires WSL/Perl).

Per project: `mvn org.jacoco:jacoco-maven-plugin:PREPARE_AGENT test
org.jacoco:jacoco-maven-plugin:REPORT -Dmaven.test.failure.ignore=true`,
then parse target/site/jacoco/jacoco.csv (one row per class: INSTRUCTION/
BRANCH/LINE/METHOD missed+covered).

Usage:
    python python/scripts/run_sf110_jacoco_coverage.py \
        --workdir outputs/mutation_experiments/sf110_full111/workdir \
        --workers 2 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

JACOCO_VERSION = "0.8.11"
SEP = "=" * 72
MVN_BIN = shutil.which("mvn") or "mvn"  # shutil.which resolves mvn.cmd on Windows; bare "mvn" fails via Popen there


def _run(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, creationflags=creationflags,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        stdout, stderr = proc.communicate()
        return 124, stdout, stderr + "\n[TIMEOUT]"


def _parse_jacoco_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rows.append({
                "package": row["PACKAGE"],
                "class_name": row["CLASS"],
                "lines_missed": int(row["LINE_MISSED"]), "lines_covered": int(row["LINE_COVERED"]),
                "branches_missed": int(row["BRANCH_MISSED"]), "branches_covered": int(row["BRANCH_COVERED"]),
                "methods_missed": int(row["METHOD_MISSED"]), "methods_covered": int(row["METHOD_COVERED"]),
            })
    return rows


def _process_project(project_dir: Path, timeout: int) -> dict:
    start = time.time()
    name = project_dir.name
    cmd = [
        MVN_BIN, "-q", "-B",
        f"org.jacoco:jacoco-maven-plugin:{JACOCO_VERSION}:prepare-agent",
        "test",
        f"org.jacoco:jacoco-maven-plugin:{JACOCO_VERSION}:report",
        "-Dmaven.test.failure.ignore=true",
    ]
    code, out, err = _run(cmd, cwd=project_dir, timeout=timeout)
    jacoco_csv = project_dir / "target" / "site" / "jacoco" / "jacoco.csv"

    if not jacoco_csv.exists():
        combined = (out + err).lower()
        if "timeout" in combined or code == 124:
            reason = "timeout"
        elif code != 0:
            reason = "build_failed"
        else:
            reason = "no_jacoco_csv"
        return {
            "project": name, "status": "failed", "failure_reason": reason,
            "error": (out + err)[-1500:], "rows": [],
            "runtime_seconds": round(time.time() - start, 2),
        }

    rows = _parse_jacoco_csv(jacoco_csv)
    for r in rows:
        r["project"] = name
    return {
        "project": name, "status": "passed", "failure_reason": "", "error": "",
        "rows": rows, "runtime_seconds": round(time.time() - start, 2),
    }


_FIELDNAMES = [
    "project", "package", "class_name",
    "lines_missed", "lines_covered", "line_coverage",
    "branches_missed", "branches_covered", "branch_coverage",
    "methods_missed", "methods_covered", "method_coverage",
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
            lt = r["lines_missed"] + r["lines_covered"]
            bt = r["branches_missed"] + r["branches_covered"]
            mt = r["methods_missed"] + r["methods_covered"]
            out = dict(r)
            out["line_coverage"] = round(r["lines_covered"] / lt, 4) if lt else 0.0
            out["branch_coverage"] = round(r["branches_covered"] / bt, 4) if bt else 0.0
            out["method_coverage"] = round(r["methods_covered"] / mt, 4) if mt else 0.0
            writer.writerow({k: out[k] for k in _FIELDNAMES})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run JaCoCo line/branch/method coverage across already-built SF110 projects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workdir", type=str, default="outputs/mutation_experiments/sf110_full111/workdir")
    parser.add_argument("--output-root", type=str, default="outputs/mutation_experiments")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-projects", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_root / "checkpoint_sf110_jacoco.json"
    csv_path = output_root / "sf110_jacoco_coverage.csv"
    failures_csv_path = output_root / "sf110_jacoco_failures.csv"
    summary_json_path = output_root / "sf110_jacoco_summary.json"

    checkpoint = _load_checkpoint(checkpoint_path) if args.resume else {"processed": [], "rows": [], "failures": []}
    processed: set[str] = set(checkpoint["processed"])
    all_rows: list[dict] = checkpoint["rows"]
    all_failures: list[dict] = checkpoint["failures"]

    projects = sorted(p.parent for p in workdir.glob("*/pom.xml"))
    if args.max_projects > 0:
        projects = projects[: args.max_projects]
    worklist = [p for p in projects if p.name not in processed]

    total = len(worklist) + len(processed)
    print(f"Worklist: {len(worklist)} projetos pendentes ({len(processed)} ja no checkpoint, {total} total)")

    lock = Lock()
    done_count = len(processed)

    def _persist() -> None:
        _save_checkpoint(checkpoint_path, {
            "processed": sorted(processed), "rows": all_rows, "failures": all_failures,
        })
        _write_csv(all_rows, csv_path)
        with open(failures_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["project", "failure_reason", "error"])
            writer.writeheader()
            writer.writerows(all_failures)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_project, p, args.timeout_seconds): p for p in worklist}
        for future in as_completed(futures):
            project_dir = futures[future]
            result = future.result()
            with lock:
                processed.add(result["project"])
                done_count += 1
                if result["status"] == "passed":
                    all_rows.extend(result["rows"])
                    print(f"[{done_count}/{total}] PASS {result['project']}: "
                          f"{len(result['rows'])} classe(s), {result['runtime_seconds']}s")
                else:
                    all_failures.append({
                        "project": result["project"],
                        "failure_reason": result["failure_reason"], "error": result["error"],
                    })
                    print(f"[{done_count}/{total}] FAIL {result['project']}: "
                          f"{result['failure_reason']} ({result['runtime_seconds']}s)")
                _persist()

    total_l_cov = sum(r["lines_covered"] for r in all_rows)
    total_l = total_l_cov + sum(r["lines_missed"] for r in all_rows)
    total_b_cov = sum(r["branches_covered"] for r in all_rows)
    total_b = total_b_cov + sum(r["branches_missed"] for r in all_rows)
    total_m_cov = sum(r["methods_covered"] for r in all_rows)
    total_m = total_m_cov + sum(r["methods_missed"] for r in all_rows)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "projects_attempted": total,
        "projects_passed": total - len(all_failures),
        "classes_extracted": len(all_rows),
        "line_coverage_global": round(total_l_cov / total_l * 100, 1) if total_l else 0.0,
        "branch_coverage_global": round(total_b_cov / total_b * 100, 1) if total_b else 0.0,
        "method_coverage_global": round(total_m_cov / total_m * 100, 1) if total_m else 0.0,
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
