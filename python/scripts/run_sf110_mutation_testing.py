#!/usr/bin/env python3
"""
Large-scale mutation testing runner for SF110 (Ant-based projects).

Why this script exists:
- Existing PIT workflow in this repository targets a single local Maven project.
- SF110 projects are Ant-based and need per-project adaptation for PIT execution.

Strategy:
1) Discover SF110 projects with build.xml and source folders.
2) For each project, generate a temporary Maven wrapper project.
3) Run PIT via pitest-maven against selected test sources (user/evosuite/both).
4) Parse mutations.csv and aggregate per-project + global metrics.
5) Persist progress checkpoint so long runs can resume safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PIT_STATUS_FALLBACK_COLUMN = 5
PIT_STATUS_COLUMN_CANDIDATES = ["status", "result", "mutation_status"]
COMPILE_ERROR_PATTERNS = (
    "compilation failure",
    "mojofailureexception",
    "incompatible types",
    "cannot find symbol",
    "package does not exist",
    "no suitable method found",
)

FAILING_TESTS_PATTERNS = (
    "tests did not pass without mutation",
    "mutation testing requires a green suite",
    "test(s) did not pass",
)


@dataclass
class PitStats:
    killed: int
    survived: int
    no_coverage: int
    timed_out: int
    memory_error: int
    non_viable: int
    total_mutations: int
    total_relevant: int
    score: float
    test_strength: float  # KILLED/(KILLED+SURVIVED)*100, excludes NO_COVERAGE


@dataclass
class RunResult:
    project: str
    path: str
    status: str
    test_mode: str
    target_classes: str
    test_sources: str
    killed: int = 0
    survived: int = 0
    no_coverage: int = 0
    timed_out: int = 0
    memory_error: int = 0
    non_viable: int = 0
    total_mutations: int = 0
    total_relevant: int = 0
    tests_ran: int = -1
    score: float = 0.0
    test_strength: float = 0.0  # KILLED/(KILLED+SURVIVED)*100, excludes NO_COVERAGE
    runtime_seconds: float = 0.0
    pit_csv: str = ""
    workdir: str = ""
    exit_code: int | None = None
    error: str = ""
    failure_reason: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""


def _discover_projects(sf110_root: Path) -> list[Path]:
    projects = []
    for build_file in sf110_root.rglob("build.xml"):
        project_dir = build_file.parent
        src_main = project_dir / "src" / "main" / "java"
        if src_main.exists():
            projects.append(project_dir.resolve())

    unique = list({str(p) for p in projects})

    def _numeric_key(path_str: str) -> tuple[int, str]:
        m = re.search(r"[/\\](\d+)_", path_str)
        return (int(m.group(1)), path_str) if m else (999_999, path_str)

    unique.sort(key=_numeric_key)
    return [Path(p) for p in unique]


def _collect_test_sources(project_dir: Path, test_mode: str) -> list[Path]:
    candidates: list[Path] = []
    user_tests = project_dir / "src" / "test" / "java"
    evo_tests = project_dir / "evosuite-tests"

    if test_mode in ("user", "both") and user_tests.exists():
        candidates.append(user_tests.resolve())
    if test_mode in ("evosuite", "both") and evo_tests.exists():
        candidates.append(evo_tests.resolve())

    return candidates


def _collect_local_jars(project_dir: Path) -> list[Path]:
    jar_dirs = [
        project_dir / "lib",
        project_dir / "test-lib",
        project_dir.parent / "lib",  # shared SF110 lib dir used by build.xml
    ]

    jars: list[Path] = []
    for jar_dir in jar_dirs:
        if not jar_dir.exists():
            continue
        jars.extend(jar_dir.glob("*.jar"))

    unique = sorted({str(p.resolve()) for p in jars})
    return [Path(p) for p in unique]


def _limit_patterns(patterns: set[str], max_patterns: int) -> list[str]:
    ordered = sorted(patterns)
    if len(ordered) > max_patterns:
        return ordered[:max_patterns]
    return ordered


def _package_pattern_from_relative_java_path(rel: Path) -> str | None:
    if len(rel.parts) <= 1:
        return None

    package_parts = rel.parts[:-1]
    if not package_parts:
        return None

    if len(package_parts) >= 2:
        package_prefix = ".".join(package_parts[:2])
    else:
        package_prefix = package_parts[0]

    if not package_prefix:
        return None
    return f"{package_prefix}.*"


def _target_class_from_test_relative_java_path(rel: Path) -> str | None:
    if not rel.parts:
        return None

    stem = rel.stem
    suffixes = ("EvoSuiteTest", "_ESTest", "_ESTest_scaffolding", "Test")
    class_name = stem
    for suffix in suffixes:
        if class_name.endswith(suffix):
            class_name = class_name[: -len(suffix)]
            break

    if not class_name:
        return None

    package_parts = list(rel.parts[:-1])
    if package_parts:
        return ".".join(package_parts + [class_name])
    return class_name


def _target_class_priority(fqcn: str) -> tuple[int, str]:
    safe_prefixes = (
        "org.json.",
        "org.pf.",
        "org.bouncycastle.",
        "org.apache.",
        "com.aelitis.azureus.core.util.",
        "com.aelitis.azureus.core.neuronal.",
        "com.aelitis.azureus.core.util.average.",
        "com.aelitis.azureus.core.util.png.",
    )
    risky_markers = (
        ".devices.",
        ".networkmanager.",
        ".ui.swt.",
        ".speedmanager.",
        ".proxy.",
        ".instancemanager.",
        ".dht.",
        ".upnp.",
        ".udp.",
        ".lws.",
    )

    if fqcn.startswith(safe_prefixes):
        return (0, fqcn)
    if any(marker in fqcn for marker in risky_markers):
        return (2, fqcn)
    return (1, fqcn)


def _derive_target_classes_from_tests(
    test_sources: list[Path], max_classes: int = 60, exclude_risky: bool = True
) -> list[str]:
    classes: set[str] = set()
    for test_root in test_sources:
        if not test_root.exists():
            continue
        for java_file in test_root.rglob("*.java"):
            rel = java_file.relative_to(test_root)
            fqcn = _target_class_from_test_relative_java_path(rel)
            if fqcn:
                classes.add(fqcn)

    if not classes:
        return []

    ordered = sorted(classes, key=_target_class_priority)
    if exclude_risky:
        ordered = [c for c in ordered if _target_class_priority(c)[0] < 2]
    if len(ordered) > max_classes:
        return ordered[:max_classes]
    return ordered


def _derive_target_classes(project_dir: Path, max_patterns: int = 20) -> list[str]:
    src_main = project_dir / "src" / "main" / "java"
    if not src_main.exists():
        return []

    patterns: set[str] = set()
    default_pkg_classes: set[str] = set()
    for java_file in src_main.rglob("*.java"):
        rel = java_file.relative_to(src_main)
        pkg_pattern = _package_pattern_from_relative_java_path(rel)
        if pkg_pattern is None:
            # Some SF110 projects keep classes in the default package.
            default_pkg_classes.add(java_file.stem)
            continue

        patterns.add(pkg_pattern)

    if patterns:
        return _limit_patterns(patterns, max_patterns)

    if not default_pkg_classes:
        return []

    defaults = {f"{name}*" for name in default_pkg_classes}
    return _limit_patterns(defaults, max_patterns)


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


_EVOSUITE_SHIM_FILES: dict[str, str] = {
    "org/evosuite/junit/EvoSuiteRunner.java": """\
package org.evosuite.junit;

import org.junit.runners.BlockJUnit4ClassRunner;
import org.junit.runners.model.InitializationError;

/** Shim: replaces missing EvoSuiteRunner with a plain JUnit4 runner. */
public class EvoSuiteRunner extends BlockJUnit4ClassRunner {
    public EvoSuiteRunner(Class<?> clazz) throws InitializationError {
        super(clazz);
    }
}
""",
    "org/evosuite/Properties.java": """\
package org.evosuite;

/** Shim: stub for org.evosuite.Properties used by EvoSuite-generated tests. */
public class Properties {
    public static boolean REPLACE_CALLS = false;
    public static boolean RESET_STATIC_FIELDS = false;
    public static boolean USE_SEPARATE_CLASSLOADER = false;

    public enum SandboxMode { OFF, RECOMMENDED, IO }
    public static SandboxMode SANDBOX_MODE = SandboxMode.OFF;
}
""",
    "org/evosuite/sandbox/Sandbox.java": """\
package org.evosuite.sandbox;

/** Shim: stub sandbox for EvoSuite-generated tests. */
public class Sandbox {
    public static void initializeSecurityManagerForSUT() {}
    public static void deinitializeSecurityManagerForSUT() {}
    public static void resetDefaultSecurityManager() {}
    public static void goingToExecuteSUTCode() {}
    public static void doneWithExecutingSUTCode() {}
}
""",
    "org/evosuite/testcase/CodeUnderTestException.java": """\
package org.evosuite.testcase;

/** Shim: stub exception for EvoSuite-generated tests. */
public class CodeUnderTestException extends RuntimeException {
    public CodeUnderTestException(Throwable cause) { super(cause); }
}
""",
    "org/evosuite/instrumentation/InstrumentingClassLoader.java": """\
package org.evosuite.instrumentation;

/** Shim: minimal loader type referenced by some EvoSuite-generated tests. */
public class InstrumentingClassLoader extends ClassLoader {
    public InstrumentingClassLoader() {
        super(InstrumentingClassLoader.class.getClassLoader());
    }
}
""",
}


def _write_evosuite_shim(workdir: Path) -> Path:
    """Write EvoSuite shim sources into workdir/evosuite-shim and return the dir."""
    shim_dir = workdir / "evosuite-shim"
    for rel_path, content in _EVOSUITE_SHIM_FILES.items():
        target = shim_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return shim_dir


def _preprocess_evosuite_test_content(content: str) -> str:
    replacements = (
        ("import org.junit.runner.RunWith;\n", ""),
        ("import org.evosuite.junit.EvoSuiteRunner;\n", ""),
        ("import org.evosuite.Properties.SandboxMode;\n", ""),
        ("import org.evosuite.sandbox.Sandbox;\n", ""),
        ("import org.evosuite.instrumentation.InstrumentingClassLoader;\n", ""),
        ("@RunWith(EvoSuiteRunner.class)\n", ""),
        ("    org.evosuite.Properties.REPLACE_CALLS = true; \n", ""),
        ("    org.evosuite.Properties.SANDBOX_MODE = SandboxMode.RECOMMENDED; \n", ""),
        ("    Sandbox.initializeSecurityManagerForSUT(); \n", ""),
        ("    Sandbox.resetDefaultSecurityManager(); \n", ""),
        ("    Sandbox.goingToExecuteSUTCode(); \n", ""),
        ("    Sandbox.doneWithExecutingSUTCode(); \n", ""),
    )

    updated = content
    for old, new in replacements:
        updated = updated.replace(old, new)

    updated = updated.replace("Collections.EmptyList", "Object")
    updated = re.sub(r"\bInstrumentingClassLoader\b", "ClassLoader", updated)
    updated = updated.replace("Class<?> class0 =", "Class class0 =")
    updated = updated.replace(".setViewClass(class0);", ".setViewClass((Class) class0);")
    return updated


def _prepare_test_sources(
    test_sources: list[Path],
    project_workdir: Path,
    target_classes: list[str] | None = None,
) -> list[Path]:
    prepared_sources: list[Path] = []
    allowed_targets = set(target_classes or [])
    for source_dir in test_sources:
        if source_dir.name != "evosuite-tests":
            prepared_sources.append(source_dir)
            continue

        dest_dir = project_workdir / "preprocessed_evosuite_tests"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for path in source_dir.rglob("*"):
            rel_path = path.relative_to(source_dir)
            dest_path = dest_dir / rel_path
            if path.is_dir():
                dest_path.mkdir(parents=True, exist_ok=True)
                continue

            if path.suffix == ".java" and allowed_targets:
                test_target = _target_class_from_test_relative_java_path(rel_path)
                if test_target not in allowed_targets:
                    continue

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".java":
                content = path.read_text(encoding="ISO-8859-1", errors="ignore")
                dest_path.write_text(_preprocess_evosuite_test_content(content), encoding="utf-8")
            else:
                shutil.copy2(path, dest_path)

        prepared_sources.append(dest_dir)

    return prepared_sources


def _generate_temp_pom(
    project_dir: Path,
    workdir: Path,
    test_sources: list[Path],
    jars: list[Path],
    target_classes: list[str],
    reports_dir: Path,
    source_level: str,
    target_level: str,
    source_encoding: str,
    pit_version: str,
    include_evosuite_runtime: bool = False,
) -> Path:
    src_main = (project_dir / "src" / "main" / "java").resolve()
    main_test_src = test_sources[0].resolve()
    extra_test_sources = [p.resolve() for p in test_sources[1:]]

    dependency_blocks = [
        """
        <dependency>
          <groupId>junit</groupId>
          <artifactId>junit</artifactId>
          <version>4.13.2</version>
          <scope>test</scope>
        </dependency>
        """.strip(),
        """
        <dependency>
          <groupId>org.hamcrest</groupId>
          <artifactId>hamcrest-core</artifactId>
          <version>1.3</version>
          <scope>test</scope>
        </dependency>
        """.strip(),
    ]

    if include_evosuite_runtime:
        dependency_blocks.append(
            """
        <dependency>
          <groupId>org.evosuite</groupId>
          <artifactId>evosuite-standalone-runtime</artifactId>
          <version>1.0.6</version>
          <scope>test</scope>
        </dependency>
        """.strip()
        )

    for idx, jar in enumerate(jars):
        dep = f"""
        <dependency>
          <groupId>sf110.local</groupId>
          <artifactId>lib-{idx}</artifactId>
          <version>1.0.0</version>
          <scope>system</scope>
          <systemPath>{_xml_escape(str(jar.resolve()))}</systemPath>
        </dependency>
        """.strip()
        dependency_blocks.append(dep)

    target_class_params = "\n".join(f"            <param>{_xml_escape(p)}</param>" for p in target_classes)
    add_test_source_block = ""
    if extra_test_sources:
        source_tags = "\n".join(
            f"                  <source>{_xml_escape(str(p))}</source>" for p in extra_test_sources
        )
        add_test_source_block = f"""
          <plugin>
            <groupId>org.codehaus.mojo</groupId>
            <artifactId>build-helper-maven-plugin</artifactId>
            <version>3.5.0</version>
            <executions>
              <execution>
                <id>add-test-sources</id>
                <phase>generate-test-sources</phase>
                <goals>
                  <goal>add-test-source</goal>
                </goals>
                <configuration>
                  <sources>
{source_tags}
                  </sources>
                </configuration>
              </execution>
            </executions>
          </plugin>
        """

    pom = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<project xmlns=\"http://maven.apache.org/POM/4.0.0\"
         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"
         xsi:schemaLocation=\"http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd\">
  <modelVersion>4.0.0</modelVersion>

  <groupId>sf110.temp</groupId>
  <artifactId>{_xml_escape(project_dir.name.lower())}-pit-wrapper</artifactId>
  <version>1.0.0</version>
  <packaging>jar</packaging>

  <properties>
    <maven.compiler.source>{_xml_escape(source_level)}</maven.compiler.source>
    <maven.compiler.target>{_xml_escape(target_level)}</maven.compiler.target>
    <project.build.sourceEncoding>{_xml_escape(source_encoding)}</project.build.sourceEncoding>
    <pitest.version>{_xml_escape(pit_version)}</pitest.version>
  </properties>

  <build>
    <sourceDirectory>{_xml_escape(str(src_main))}</sourceDirectory>
    <testSourceDirectory>{_xml_escape(str(main_test_src))}</testSourceDirectory>

    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-compiler-plugin</artifactId>
        <version>3.11.0</version>
        <configuration>
          <source>{_xml_escape(source_level)}</source>
          <target>{_xml_escape(target_level)}</target>
          <encoding>{_xml_escape(source_encoding)}</encoding>
                    <fork>true</fork>
                    <meminitial>256m</meminitial>
                    <maxmem>2048m</maxmem>
        </configuration>
      </plugin>

{add_test_source_block}

      <plugin>
        <groupId>org.pitest</groupId>
        <artifactId>pitest-maven</artifactId>
        <version>${{pitest.version}}</version>
        <configuration>
          <outputFormats>
            <param>CSV</param>
            <param>XML</param>
            <param>HTML</param>
          </outputFormats>
          <reportsDirectory>{_xml_escape(str(reports_dir.resolve()))}</reportsDirectory>
          <timestampedReports>false</timestampedReports>
          <threads>1</threads>
          <targetClasses>
{target_class_params}
          </targetClasses>
          <targetTests>
            <param>*</param>
          </targetTests>
          <failWhenNoMutations>false</failWhenNoMutations>
          <mutationThreshold>0</mutationThreshold>
          <skipFailingTests>true</skipFailingTests>
          <timeoutFactor>1.5</timeoutFactor>
          <timeoutConstant>3000</timeoutConstant>
          <jvmArgs>
            <jvmArg>-Xmx1024m</jvmArg>
            <jvmArg>-Xss8m</jvmArg>
            <jvmArg>-XX:+UseSerialGC</jvmArg>
          </jvmArgs>
        </configuration>
      </plugin>
    </plugins>
  </build>

  <dependencies>
{os.linesep.join('    ' + d.replace(os.linesep, os.linesep + '    ') for d in dependency_blocks)}
  </dependencies>
</project>
"""

    workdir.mkdir(parents=True, exist_ok=True)
    pom_path = workdir / "pom.xml"
    pom_path.write_text(pom, encoding="utf-8")
    return pom_path


def _find_latest_mutations_csv(reports_dir: Path) -> Path | None:
    direct_csv = reports_dir / "mutations.csv"
    timestamped_csv = list(reports_dir.glob("*/mutations.csv"))

    candidates: list[Path] = []
    if direct_csv.exists():
        candidates.append(direct_csv)
    candidates.extend([p for p in timestamped_csv if p.exists()])

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_status_column(csv_path: Path) -> list[str]:
    # Read as DictReader first to support header CSV.
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            lowered = {str(c).strip().lower(): c for c in reader.fieldnames}
            for candidate in PIT_STATUS_COLUMN_CANDIDATES:
                if candidate in lowered:
                    key = lowered[candidate]
                    return [str(row.get(key, "")).strip().upper() for row in reader]

    # Fall back to positional parsing without header.
    statuses: list[str] = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        raw_reader = csv.reader(f)
        for row in raw_reader:
            if len(row) > PIT_STATUS_FALLBACK_COLUMN:
                statuses.append(str(row[PIT_STATUS_FALLBACK_COLUMN]).strip().upper())
    return statuses


def _parse_pit_results(csv_path: Path) -> PitStats:
    statuses = _extract_status_column(csv_path)

    killed = sum(1 for s in statuses if s == "KILLED")
    survived = sum(1 for s in statuses if s == "SURVIVED")
    no_coverage = sum(1 for s in statuses if s == "NO_COVERAGE")
    timed_out = sum(1 for s in statuses if s == "TIMED_OUT")
    memory_error = sum(1 for s in statuses if s == "MEMORY_ERROR")
    non_viable = sum(1 for s in statuses if s == "NON_VIABLE")

    killed_total = killed + timed_out
    total_mutations = len(statuses)
    total_relevant = total_mutations - non_viable - memory_error
    score = (killed_total / total_relevant * 100.0) if total_relevant > 0 else 0.0
    # test_strength: of the mutants that WERE covered, what fraction was killed?
    covered = killed_total + survived
    test_strength = (killed_total / covered * 100.0) if covered > 0 else 0.0

    return PitStats(
        killed=killed_total,
        survived=survived,
        no_coverage=no_coverage,
        timed_out=timed_out,
        memory_error=memory_error,
        non_viable=non_viable,
        total_mutations=total_mutations,
        total_relevant=total_relevant,
        score=round(score, 2),
        test_strength=round(test_strength, 2),
    )


def _extract_tests_ran(stdout: str, stderr: str) -> int | None:
    combined = f"{stdout}\n{stderr}"
    matches = re.findall(r">>\s*Ran\s+(\d+)\s+tests", combined)
    if not matches:
        return None
    return int(matches[-1])


def _coerce_process_stream(value: str | bytes | None) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        proc.kill()


def _run_command(cmd: list[str], cwd: Path, timeout_seconds: int) -> tuple[int, str, str, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        return proc.returncode, _coerce_process_stream(stdout), _coerce_process_stream(stderr), ""
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

        timeout_stdout = _coerce_process_stream(exc.stdout) or _coerce_process_stream(stdout)
        timeout_stderr = _coerce_process_stream(exc.stderr) or _coerce_process_stream(stderr)
        return 124, timeout_stdout, timeout_stderr, f"Timeout after {timeout_seconds}s"


def _looks_like_compile_error(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    return any(token in combined for token in COMPILE_ERROR_PATTERNS)


def _looks_like_failing_tests(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    return any(token in combined for token in FAILING_TESTS_PATTERNS)


def _classify_command_failure(
    *,
    exit_code: int | None,
    error: str,
    stdout: str,
    stderr: str,
    pit_csv: Path | None,
) -> str:
    lowered_error = (error or "").lower()
    if exit_code == 124 or "timeout" in lowered_error:
        return "timeout"
    if _looks_like_compile_error(stdout, stderr):
        return "compile_error"
    if _looks_like_failing_tests(stdout, stderr):
        return "failing_tests"
    if not pit_csv or not pit_csv.exists():
        return "pit_missing_csv"
    return "command_failed"


def _tail(text: str, max_chars: int = 4000) -> str:
    if not text:
        return ""
    return text[-max_chars:]


def _safe_name(project_dir: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", project_dir.name)


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {"results": []}
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "results" not in data or not isinstance(data["results"], list):
        data["results"] = []
    return data


def _save_checkpoint(checkpoint_path: Path, data: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_results_csv(results: list[RunResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "project",
        "path",
        "status",
        "test_mode",
        "target_classes",
        "test_sources",
        "killed",
        "survived",
        "no_coverage",
        "timed_out",
        "memory_error",
        "non_viable",
        "total_mutations",
        "total_relevant",
        "tests_ran",
        "score",
        "test_strength",
        "runtime_seconds",
        "pit_csv",
        "workdir",
        "exit_code",
        "error",
        "failure_reason",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


def _build_summary(results: list[RunResult], args: argparse.Namespace, projects_total: int) -> dict[str, Any]:
    executed = len(results)
    passed = [r for r in results if r.status == "passed"]
    failed = [r for r in results if r.status != "passed"]

    killed_sum = sum(r.killed for r in passed)
    relevant_sum = sum(r.total_relevant for r in passed)
    covered_sum = sum(r.killed + r.survived for r in passed)
    weighted_score = (killed_sum / relevant_sum * 100.0) if relevant_sum > 0 else 0.0
    weighted_test_strength = (killed_sum / covered_sum * 100.0) if covered_sum > 0 else 0.0

    avg_score = (sum(r.score for r in passed) / len(passed)) if passed else 0.0
    avg_test_strength = (sum(r.test_strength for r in passed) / len(passed)) if passed else 0.0
    zero_test_projects = [r for r in results if r.tests_ran == 0]
    failure_reason_breakdown: dict[str, int] = {}
    for r in failed:
        reason = r.failure_reason or "unknown"
        failure_reason_breakdown[reason] = failure_reason_breakdown.get(reason, 0) + 1

    top_projects = sorted(passed, key=lambda r: r.score, reverse=True)[:10]
    worst_projects = sorted(passed, key=lambda r: r.score)[:10]

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "sf110_root": str(Path(args.sf110_root).resolve()).replace("\\", "/"),
            "test_mode": args.test_mode,
            "max_projects": args.max_projects,
            "project_filter": args.project_filter,
            "timeout_seconds": args.timeout_seconds,
            "source_level": args.source_level,
            "target_level": args.target_level,
            "source_encoding": args.source_encoding,
            "pit_version": args.pit_version,
            "resume": args.resume,
        },
        "projects_total_selected": projects_total,
        "projects_executed": executed,
        "projects_passed": len(passed),
        "projects_failed": len(failed),
        "overall_weighted_mutation_score": round(weighted_score, 2),
        "overall_weighted_test_strength": round(weighted_test_strength, 2),
        "average_project_score": round(avg_score, 2),
        "average_project_test_strength": round(avg_test_strength, 2),
        "killed_sum": killed_sum,
        "relevant_mutations_sum": relevant_sum,
        "projects_with_zero_tests": len(zero_test_projects),
        "failure_reason_breakdown": failure_reason_breakdown,
        "top_projects": [
            {
                "project": r.project,
                "score": r.score,
                "test_strength": r.test_strength,
                "killed": r.killed,
                "survived": r.survived,
                "no_coverage": r.no_coverage,
                "total_relevant": r.total_relevant,
            }
            for r in top_projects
        ],
        "worst_projects": [
            {
                "project": r.project,
                "score": r.score,
                "test_strength": r.test_strength,
                "killed": r.killed,
                "survived": r.survived,
                "no_coverage": r.no_coverage,
                "total_relevant": r.total_relevant,
            }
            for r in worst_projects
        ],
        "failed_projects": [
            {
                "project": r.project,
                "error": r.error,
                "exit_code": r.exit_code,
                "failure_reason": r.failure_reason,
            }
            for r in failed
        ],
    }


def _project_to_result(obj: dict[str, Any]) -> RunResult:
    return RunResult(
        project=str(obj.get("project", "")),
        path=str(obj.get("path", "")),
        status=str(obj.get("status", "failed")),
        test_mode=str(obj.get("test_mode", "")),
        target_classes=str(obj.get("target_classes", "")),
        test_sources=str(obj.get("test_sources", "")),
        killed=int(obj.get("killed", 0)),
        survived=int(obj.get("survived", 0)),
        no_coverage=int(obj.get("no_coverage", 0)),
        timed_out=int(obj.get("timed_out", 0)),
        memory_error=int(obj.get("memory_error", 0)),
        non_viable=int(obj.get("non_viable", 0)),
        total_mutations=int(obj.get("total_mutations", 0)),
        total_relevant=int(obj.get("total_relevant", 0)),
        tests_ran=int(obj.get("tests_ran", -1)),
        score=float(obj.get("score", 0.0)),
        test_strength=float(obj.get("test_strength", 0.0)),
        runtime_seconds=float(obj.get("runtime_seconds", 0.0)),
        pit_csv=str(obj.get("pit_csv", "")),
        workdir=str(obj.get("workdir", "")),
        exit_code=(None if obj.get("exit_code") is None else int(obj.get("exit_code"))),
        error=str(obj.get("error", "")),
        failure_reason=str(obj.get("failure_reason", "")),
        stdout_tail=str(obj.get("stdout_tail", "")),
        stderr_tail=str(obj.get("stderr_tail", "")),
    )


def _persist_checkpoint(checkpoint_path: Path, run_results: list[RunResult], selected_total: int) -> None:
    checkpoint = {
        "results": [r.__dict__ for r in run_results],
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "projects_total_selected": selected_total,
    }
    _save_checkpoint(checkpoint_path, checkpoint)


def _build_failure_result(
    project_dir: Path,
    test_mode: str,
    context: tuple[list[str], list[Path]],
    runtime_seconds: float,
    error: str,
    failure_reason: str,
    *,
    stats: PitStats | None = None,
    tests_ran: int = -1,
    pit_csv: str = "",
    workdir: str = "",
    exit_code: int | None = None,
    stdout_tail: str = "",
    stderr_tail: str = "",
) -> RunResult:
    target_classes, test_sources = context

    return RunResult(
        project=project_dir.name,
        path=str(project_dir).replace("\\", "/"),
        status="failed",
        test_mode=test_mode,
        target_classes=",".join(target_classes),
        test_sources=",".join(str(p).replace("\\", "/") for p in test_sources),
        killed=(stats.killed if stats else 0),
        survived=(stats.survived if stats else 0),
        no_coverage=(stats.no_coverage if stats else 0),
        timed_out=(stats.timed_out if stats else 0),
        memory_error=(stats.memory_error if stats else 0),
        non_viable=(stats.non_viable if stats else 0),
        total_mutations=(stats.total_mutations if stats else 0),
        total_relevant=(stats.total_relevant if stats else 0),
        tests_ran=tests_ran,
        score=(stats.score if stats else 0.0),
        test_strength=(stats.test_strength if stats else 0.0),
        runtime_seconds=runtime_seconds,
        pit_csv=pit_csv,
        workdir=workdir,
        exit_code=exit_code,
        error=error,
        failure_reason=failure_reason,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def _build_success_result(
    project_dir: Path,
    test_mode: str,
    target_classes: list[str],
    test_sources: list[Path],
    stats: PitStats,
    tests_ran: int,
    runtime_seconds: float,
    pit_csv: Path,
    project_workdir: Path,
    exit_code: int,
    stdout: str,
    stderr: str,
) -> RunResult:
    return RunResult(
        project=project_dir.name,
        path=str(project_dir).replace("\\", "/"),
        status="passed",
        test_mode=test_mode,
        target_classes=",".join(target_classes),
        test_sources=",".join(str(p).replace("\\", "/") for p in test_sources),
        killed=stats.killed,
        survived=stats.survived,
        no_coverage=stats.no_coverage,
        timed_out=stats.timed_out,
        memory_error=stats.memory_error,
        non_viable=stats.non_viable,
        total_mutations=stats.total_mutations,
        total_relevant=stats.total_relevant,
        tests_ran=tests_ran,
        score=stats.score,
        test_strength=stats.test_strength,
        runtime_seconds=runtime_seconds,
        pit_csv=str(pit_csv).replace("\\", "/"),
        workdir=str(project_workdir).replace("\\", "/"),
        exit_code=exit_code,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
    )


def _run_single_project(
    project_dir: Path,
    idx: int,
    selected_total: int,
    args: argparse.Namespace,
    mvn_cmd: str,
    work_root: Path,
) -> RunResult:
    start = time.time()
    test_sources = _collect_test_sources(project_dir, args.test_mode)
    target_classes = _derive_target_classes_from_tests(
        test_sources, max_classes=args.max_classes, exclude_risky=True
    )
    if not target_classes:
        target_classes = _derive_target_classes(project_dir)

    if not test_sources:
        return _build_failure_result(
            project_dir=project_dir,
            test_mode=args.test_mode,
            context=(target_classes, test_sources),
            runtime_seconds=round(time.time() - start, 2),
            error="No compatible test sources found for selected test mode",
            failure_reason="no_test_sources",
        )

    if not target_classes:
        return _build_failure_result(
            project_dir=project_dir,
            test_mode=args.test_mode,
            context=(target_classes, test_sources),
            runtime_seconds=round(time.time() - start, 2),
            error="Could not infer targetClasses patterns from source tree",
            failure_reason="no_target_classes",
        )

    jars = _collect_local_jars(project_dir)
    safe_name = _safe_name(project_dir)
    project_workdir = work_root / safe_name
    reports_dir = project_workdir / "pit-reports"
    effective_test_sources = _prepare_test_sources(test_sources, project_workdir, target_classes=target_classes)

    evo_tests_dir = project_dir / "evosuite-tests"
    include_evosuite_runtime = evo_tests_dir.exists() and args.test_mode in ("evosuite", "both")
    if include_evosuite_runtime:
        shim_dir = _write_evosuite_shim(project_workdir)
        effective_test_sources.append(shim_dir)
    pom_path = _generate_temp_pom(
        project_dir=project_dir,
        workdir=project_workdir,
        test_sources=effective_test_sources,
        jars=jars,
        target_classes=target_classes,
        reports_dir=reports_dir,
        source_level=args.source_level,
        target_level=args.target_level,
        source_encoding=args.source_encoding,
        pit_version=args.pit_version,
        include_evosuite_runtime=include_evosuite_runtime,
    )

    cmd = [
        mvn_cmd,
        "-f",
        str(pom_path),
        "clean",
        "test-compile",
        "org.pitest:pitest-maven:mutationCoverage",
        "-DtimestampedReports=false",
    ]

    print(f"[{idx}/{selected_total}] Running PIT for {project_dir.name}")
    exit_code, stdout, stderr, error = _run_command(cmd, cwd=project_workdir, timeout_seconds=args.timeout_seconds)
    runtime = round(time.time() - start, 2)
    pit_csv = _find_latest_mutations_csv(reports_dir)

    if exit_code != 0 or not pit_csv or not pit_csv.exists():
        failure_reason = _classify_command_failure(
            exit_code=exit_code,
            error=error,
            stdout=stdout,
            stderr=stderr,
            pit_csv=pit_csv,
        )
        return _build_failure_result(
            project_dir=project_dir,
            test_mode=args.test_mode,
            context=(target_classes, test_sources),
            runtime_seconds=runtime,
            pit_csv=(str(pit_csv).replace("\\", "/") if pit_csv else ""),
            workdir=str(project_workdir).replace("\\", "/"),
            exit_code=exit_code,
            error=error or "Maven/PIT command failed or mutations.csv not generated",
            failure_reason=failure_reason,
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
        )

    try:
        stats = _parse_pit_results(pit_csv)
    except Exception as exc:
        return _build_failure_result(
            project_dir=project_dir,
            test_mode=args.test_mode,
            context=(target_classes, test_sources),
            runtime_seconds=runtime,
            pit_csv=str(pit_csv).replace("\\", "/"),
            workdir=str(project_workdir).replace("\\", "/"),
            exit_code=exit_code,
            error=f"Failed parsing PIT CSV: {exc}",
            failure_reason="parse_error",
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
        )

    tests_ran = _extract_tests_ran(stdout, stderr)
    if tests_ran is not None and tests_ran == 0:
        return _build_failure_result(
            project_dir=project_dir,
            test_mode=args.test_mode,
            context=(target_classes, test_sources),
            runtime_seconds=runtime,
            error="PIT reported 0 executed tests (likely initialization/classpath issue)",
            failure_reason="zero_tests",
            stats=stats,
            tests_ran=0,
            pit_csv=str(pit_csv).replace("\\", "/"),
            workdir=str(project_workdir).replace("\\", "/"),
            exit_code=exit_code,
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
        )

    return _build_success_result(
        project_dir=project_dir,
        test_mode=args.test_mode,
        target_classes=target_classes,
        test_sources=test_sources,
        stats=stats,
        tests_ran=(tests_ran if tests_ran is not None else -1),
        runtime_seconds=runtime,
        pit_csv=pit_csv,
        project_workdir=project_workdir,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large-scale mutation testing on SF110")
    parser.add_argument(
        "--sf110-root",
        type=str,
        default="python/data/raw/evosuite/sf110/extracted/SF110-20130704-src/SF110-20130704-src",
        help="Root folder containing SF110 project directories",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/mutation_experiments/sf110",
        help="Output folder for checkpoint/results/summary",
    )
    parser.add_argument(
        "--test-mode",
        choices=["user", "evosuite", "both"],
        default="evosuite",
        help="Which tests to compile/use for PIT",
    )
    parser.add_argument("--max-projects", type=int, default=0, help="Max projects to execute (0 = all)")
    parser.add_argument("--project-filter", type=str, default=None, help="Only run projects containing this text")
    parser.add_argument("--timeout-seconds", type=int, default=1200, help="Per-project timeout in seconds")
    parser.add_argument("--source-level", type=str, default="1.8", help="Maven compiler source level")
    parser.add_argument("--target-level", type=str, default="1.8", help="Maven compiler target level")
    parser.add_argument("--source-encoding", type=str, default="ISO-8859-1", help="Source encoding")
    parser.add_argument("--pit-version", type=str, default="1.15.8", help="pitest-maven version")
    parser.add_argument("--max-classes", type=int, default=60, help="Max target classes per project (0 = unlimited)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if it exists")
    parser.add_argument("--reset", action="store_true", help="Delete output root before running")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on first project failure (default continues)",
    )
    return parser.parse_args()


def _prepare_output_paths(output_root: Path) -> tuple[Path, Path, Path, Path, Path]:
    work_root = output_root / "workdir"
    checkpoint_path = output_root / "checkpoint.json"
    results_csv = output_root / "sf110_mutation_results.csv"
    summary_json = output_root / "sf110_mutation_summary.json"
    return output_root, work_root, checkpoint_path, results_csv, summary_json


def _select_projects(sf110_root: Path, project_filter: str | None, max_projects: int) -> list[Path]:
    projects = _discover_projects(sf110_root)
    if project_filter:
        needle = project_filter.lower()
        projects = [p for p in projects if needle in str(p).lower()]

    if max_projects > 0:
        projects = projects[:max_projects]

    if not projects:
        raise RuntimeError("No SF110 projects found with build.xml + src/main/java")

    return projects


def _load_previous_results(resume: bool, checkpoint_path: Path) -> tuple[list[RunResult], set[str]]:
    checkpoint = _load_checkpoint(checkpoint_path) if resume else {"results": []}
    previous_results = [_project_to_result(r) for r in checkpoint.get("results", [])]
    processed = {r.project for r in previous_results}
    return previous_results, processed


def _run_projects_batch(
    projects: list[Path],
    args: argparse.Namespace,
    mvn_cmd: str,
    work_root: Path,
    checkpoint_path: Path,
    results_csv: Path,
    summary_json: Path,
    run_results: list[RunResult],
    processed: set[str],
) -> list[RunResult]:
    selected_total = len(projects)

    for idx, project_dir in enumerate(projects, start=1):
        project_name = project_dir.name
        if project_name in processed:
            print(f"[SKIP] {project_name} already in checkpoint")
            continue

        try:
            result = _run_single_project(
                project_dir=project_dir,
                idx=idx,
                selected_total=selected_total,
                args=args,
                mvn_cmd=mvn_cmd,
                work_root=work_root,
            )
        except Exception as exc:
            result = _build_failure_result(
                project_dir=project_dir,
                test_mode=args.test_mode,
                context=([], []),
                runtime_seconds=0.0,
                error=f"Unhandled runner exception: {exc}",
                failure_reason="command_failed",
                stderr_tail=_tail(repr(exc)),
            )

        run_results.append(result)
        _persist_checkpoint(checkpoint_path, run_results, selected_total)
        _write_results_csv(run_results, results_csv)
        partial_summary = _build_summary(run_results, args, selected_total)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(partial_summary, f, indent=2)

        if result.status == "passed":
            print(
                f"[PASS] {project_name}: score={result.score}% strength={result.test_strength}% "
                f"(killed={result.killed}, survived={result.survived}, no_cov={result.no_coverage})"
            )
        elif result.exit_code is not None:
            print(f"[FAIL] {project_name}: exit={result.exit_code} {result.error}")
        else:
            print(f"[FAIL] {project_name}: {result.error}")

        if result.status != "passed" and args.stop_on_error:
            break

    return run_results


def _print_summary(
    summary: dict[str, Any],
    selected_total: int,
    results_csv: Path,
    summary_json: Path,
    checkpoint_path: Path,
) -> None:
    print("=" * 72)
    print("SF110 mutation testing finished")
    print(f"Projects selected: {selected_total}")
    print(f"Projects executed: {summary['projects_executed']}")
    print(f"Passed:            {summary['projects_passed']}")
    print(f"Failed:            {summary['projects_failed']}")
    print(f"Weighted score:    {summary['overall_weighted_mutation_score']}%")
    print(f"Weighted strength: {summary['overall_weighted_test_strength']}%  (excl. no_coverage)")
    print(f"Results CSV:       {results_csv}")
    print(f"Summary JSON:      {summary_json}")
    print(f"Checkpoint:        {checkpoint_path}")
    print("=" * 72)


def main() -> None:
    args = _parse_args()

    sf110_root = Path(args.sf110_root).resolve()
    if not sf110_root.exists():
        raise FileNotFoundError(f"SF110 root not found: {sf110_root}")

    output_root, work_root, checkpoint_path, results_csv, summary_json = _prepare_output_paths(
        Path(args.output_root).resolve()
    )
    if args.reset and output_root.exists():
        shutil.rmtree(output_root)

    mvn_cmd = "mvn.cmd" if os.name == "nt" else "mvn"

    all_projects = _select_projects(sf110_root, args.project_filter, args.max_projects)
    previous_results, processed = _load_previous_results(args.resume, checkpoint_path)

    if args.resume and previous_results:
        print(f"[INFO] Resuming from checkpoint: {len(previous_results)} projects already recorded")

    selected_total = len(all_projects)
    run_results: list[RunResult] = previous_results.copy()

    run_results = _run_projects_batch(
        projects=all_projects,
        args=args,
        mvn_cmd=mvn_cmd,
        work_root=work_root,
        checkpoint_path=checkpoint_path,
        results_csv=results_csv,
        summary_json=summary_json,
        run_results=run_results,
        processed=processed,
    )

    _write_results_csv(run_results, results_csv)
    summary = _build_summary(run_results, args, selected_total)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary, selected_total, results_csv, summary_json, checkpoint_path)


if __name__ == "__main__":
    main()
