"""
Run Java tests for extracted SF110 projects (Ant-based) and produce a summary report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _bootstrap_ant_local(repo_root: Path) -> Path:
    ant_version = "1.10.15"
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        tools_dir = Path(local_app_data) / "sf110-tools"
    else:
        tools_dir = repo_root / "python" / "tools"

    ant_home = tools_dir / f"apache-ant-{ant_version}"
    ant_bat = ant_home / "bin" / "ant.bat"
    if ant_bat.exists():
        return ant_bat

    tools_dir.mkdir(parents=True, exist_ok=True)

    zip_path = tools_dir / f"apache-ant-{ant_version}-bin.zip"
    url = f"https://dlcdn.apache.org/ant/binaries/apache-ant-{ant_version}-bin.zip"

    print(f"[INFO] Ant not found in PATH. Downloading Apache Ant {ant_version}...")
    urllib.request.urlretrieve(url, str(zip_path))

    print("[INFO] Extracting Apache Ant locally...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(tools_dir))

    if not ant_bat.exists():
        raise RuntimeError(f"Failed to bootstrap Apache Ant at expected path: {ant_bat}")

    return ant_bat


def _resolve_ant_cmd() -> str:
    if os.name != "nt":
        return "ant"

    # Windows commonly exposes ant.bat.
    for candidate in ("ant", "ant.bat", "ant.cmd"):
        try:
            subprocess.run([candidate, "-version"], capture_output=True, text=True, check=False)
            return candidate
        except FileNotFoundError:
            continue

    ant_home = os.getenv("ANT_HOME")
    if ant_home:
        ant_bat = Path(ant_home) / "bin" / "ant.bat"
        if ant_bat.exists():
            return str(ant_bat)

    repo_root = Path(__file__).resolve().parents[2]
    ant_bat = _bootstrap_ant_local(repo_root)
    return str(ant_bat)


def _discover_projects(sf110_root: Path) -> list[Path]:
    projects = []
    for build_file in sf110_root.rglob("build.xml"):
        project_dir = build_file.parent
        has_tests = (project_dir / "evosuite-tests").exists() or (project_dir / "src" / "test").exists()
        if has_tests:
            projects.append(project_dir)

    # unique + stable order
    unique = sorted({p.resolve() for p in projects})
    return [Path(p) for p in unique]


def _choose_ant_target(ant_cmd: str, project_dir: Path, preferred: str | None) -> str | None:
    if preferred:
        return preferred

    code, out, _ = _run([ant_cmd, "-p"], project_dir)
    if code != 0:
        return None

    lower = out.lower()
    for target in ("test", "tests", "junit", "junit-test", "run-tests"):
        if f" {target} " in lower or f"\n{target} " in lower or f" {target}\n" in lower:
            return target
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Java tests for SF110 extracted projects")
    parser.add_argument(
        "--sf110-root",
        type=str,
        default="python/data/raw/evosuite/sf110/extracted/SF110-20130704-src/SF110-20130704-src",
        help="Root folder containing SF110 project directories",
    )
    parser.add_argument("--max-projects", type=int, default=20, help="Maximum projects to test (0 = all)")
    parser.add_argument("--project-filter", type=str, default=None, help="Filter project paths containing this text")
    parser.add_argument("--ant-target", type=str, default=None, help="Force Ant target (default: auto detect)")
    parser.add_argument(
        "--report-path",
        type=str,
        default="outputs/sf110_hypotheses/summaries/sf110_java_test_report.json",
        help="Output JSON report path",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if individual projects fail")
    parser.add_argument(
        "--javac-source",
        type=str,
        default="1.8",
        help="Override compile.source and ant.build.javac.source (empty to disable)",
    )
    parser.add_argument(
        "--javac-target",
        type=str,
        default="1.8",
        help="Override compile.target and ant.build.javac.target (empty to disable)",
    )
    args = parser.parse_args()

    sf110_root = Path(args.sf110_root)
    if not sf110_root.exists():
        raise FileNotFoundError(f"SF110 root not found: {sf110_root}")

    ant_cmd = _resolve_ant_cmd()

    projects = _discover_projects(sf110_root)
    if args.project_filter:
        needle = args.project_filter.lower()
        projects = [p for p in projects if needle in str(p).lower()]

    if args.max_projects > 0:
        projects = projects[: args.max_projects]

    if not projects:
        raise RuntimeError("No SF110 projects with tests/build.xml were found for execution")

    report = {
        "sf110_root": str(sf110_root).replace("\\", "/"),
        "ant_cmd": ant_cmd,
        "projects_selected": len(projects),
        "results": [],
    }

    passed = 0
    failed = 0

    for project_dir in projects:
        target = _choose_ant_target(ant_cmd, project_dir, args.ant_target)
        ant_overrides: list[str] = []
        if args.javac_source:
            ant_overrides.extend(["-Dcompile.source=" + args.javac_source, "-Dant.build.javac.source=" + args.javac_source])
        if args.javac_target:
            ant_overrides.extend(["-Dcompile.target=" + args.javac_target, "-Dant.build.javac.target=" + args.javac_target])

        if target:
            cmd = [ant_cmd, *ant_overrides, target]
        else:
            cmd = [ant_cmd, *ant_overrides]

        code, out, err = _run(cmd, project_dir)
        ok = code == 0

        if ok:
            passed += 1
        else:
            failed += 1

        report["results"].append(
            {
                "project": project_dir.name,
                "path": str(project_dir).replace("\\", "/"),
                "target": target if target else "default",
                "status": "passed" if ok else "failed",
                "exit_code": code,
                "stdout_tail": out[-4000:],
                "stderr_tail": err[-4000:],
            }
        )

        print(f"[{ 'PASS' if ok else 'FAIL' }] {project_dir.name} (target={target if target else 'default'})")

        if (not ok) and (not args.continue_on_error):
            break

    report["summary"] = {
        "passed": passed,
        "failed": failed,
        "executed": passed + failed,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print("✅ SF110 Java test execution finished")
    print(f"   Executed: {report['summary']['executed']}")
    print(f"   Passed:   {report['summary']['passed']}")
    print(f"   Failed:   {report['summary']['failed']}")
    print(f"   Report:   {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
