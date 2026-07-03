#!/usr/bin/env python3
"""
Large-scale mutation testing runner for SF110 (Ant-based projects).

Why this script exists:
- Existing PIT workflow in this repository targets a single local Maven project.
- SF110 projects are Ant-based and need per-project adaptation for PIT execution.

Strategy:
1) Discover SF110 projects with build.xml and source folders.
2) For each project, generate a temporary Maven wrapper project with a unified test directory.
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
    "cannot access",
    "error: package",
    "error: class",
    "error: cannot find",
)

FAILING_TESTS_PATTERNS = (
    "tests did not pass without mutation",
    "mutation testing requires a green suite",
    "test(s) did not pass",
)

# Windows process-killed / OOM exit codes.
# 0xC000013A = STATUS_CONTROL_C_EXIT (also used for OOM kills by the OS)
# 3221226391 = 0xC0000017 = STATUS_NO_MEMORY
_OOM_EXIT_CODES: frozenset[int] = frozenset({3221226091, 3221226391, 137})

# Source levels tried in order when compile retry is enabled
# Source levels to try in retry order. Modern Maven (3.9+) rejects 1.6 outright
# ("Source option 6 is no longer supported. Use 7 or later."), so trying 1.6
# silently breaks projects that compile fine at 1.8 -- the retry overwrites the
# good attempt with a failed one. We keep only levels Maven still accepts.
_SOURCE_LEVEL_FALLBACKS = ("1.8", "1.7")


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

    stem = java_file_stem = rel.stem
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


_TEST_FILENAME_SUFFIXES = (
    "Test.java", "Tests.java", "TestCase.java",
    "_ESTest.java", "_ESTest_scaffolding.java",
    "EvoSuiteTest.java", "ITCase.java",
)


def _looks_like_test_filename(name: str) -> bool:
    """Return True for filenames that clearly belong to test code."""
    return name.endswith(_TEST_FILENAME_SUFFIXES) or name.startswith("Test")


def _prepare_test_sources(
    test_sources: list[Path],
    project_workdir: Path,
    target_classes: list[str] | None = None,
    include_shims: bool = False,
) -> list[Path]:
    prepared_sources: list[Path] = []
    allowed_targets = set(target_classes or [])
    
    # Criação do diretório unificado de testes para estabilizar o classpath do Maven/PIT
    unified_test_dir = project_workdir / "unified_prepared_tests"
    if unified_test_dir.exists():
        shutil.rmtree(unified_test_dir)
    unified_test_dir.mkdir(parents=True, exist_ok=True)

    if include_shims:
        for rel_path, content in _EVOSUITE_SHIM_FILES.items():
            target = unified_test_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    for source_dir in test_sources:
        if not source_dir.exists():
            continue

        for path in source_dir.rglob("*"):
            if path.is_dir():
                continue

            rel_path = path.relative_to(source_dir)
            dest_path = unified_test_dir / rel_path

            # Some legacy SF110 projects mix main classes with tests under
            # src/test/java. Filter out files that are ALSO in src/main/java
            # (those are duplicated main classes, not test helpers). But always
            # include test helpers (mocks, fixtures) that only exist in test/.
            if path.suffix == ".java" and source_dir.name != "evosuite-tests":
                is_test_by_name = _looks_like_test_filename(path.name)
                if not is_test_by_name:
                    # Check if this file also exists in src/main/java (main class
                    # accidentally placed under test). If so, skip it.
                    # Determine the project root by going up from source_dir.
                    try:
                        main_counterpart = source_dir.parent.parent / "main" / "java" / rel_path
                    except Exception:
                        main_counterpart = None
                    if main_counterpart is not None and main_counterpart.exists():
                        # File exists in main sources — it's a duplicated main class, skip.
                        continue
                    # Otherwise it's a test helper (mock, fixture, base class).
                    # Include it so tests that depend on it can compile.

            if source_dir.name == "evosuite-tests" and path.suffix == ".java" and allowed_targets:
                test_target = _target_class_from_test_relative_java_path(rel_path)
                if test_target not in allowed_targets:
                    continue

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".java":
                content = path.read_text(encoding="ISO-8859-1", errors="ignore")
                dest_path.write_text(_preprocess_evosuite_test_content(content), encoding="utf-8")
            else:
                shutil.copy2(path, dest_path)

    prepared_sources.append(unified_test_dir)
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
    jvm_max_heap: str = "1536m",
    mutators: list[str] | None = None,
) -> Path:
    src_main = (project_dir / "src" / "main" / "java").resolve()
    main_test_src = test_sources[0].resolve()

    # Empty when omitted -> PIT uses its DEFAULTS. When given, pass an explicit
    # mutator list so the run is reproducible regardless of the PIT version's
    # default selection.
    if mutators:
        mutator_params = "\n".join(f"            <param>{_xml_escape(m)}</param>" for m in mutators)
        mutators_xml = f"<mutators>\n{mutator_params}\n          </mutators>"
    else:
        mutators_xml = ""

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

    # Defensive: skip JARs that no longer exist on disk. Without this guard, the
    # generated pom can reference stale paths (e.g. a JAR that was renamed or
    # replaced between collection and pom emission), which causes Maven to fail
    # the whole dependency resolution and the project counts as compile_error,
    # even though the actual source compiles fine.
    valid_jars = [j for j in jars if j.exists()]
    if len(valid_jars) != len(jars):
        # Surfaces the issue in the per-project output without crashing the run.
        missing = [j for j in jars if not j.exists()]
        print(f"  [pom] {len(missing)} JAR(s) ignorado(s) (caminho stale): "
              f"{', '.join(j.name for j in missing[:3])}{'...' if len(missing) > 3 else ''}")
    for idx, jar in enumerate(valid_jars):
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

    pom = f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
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
          {mutators_xml}
          <failWhenNoMutations>false</failWhenNoMutations>
          <mutationThreshold>0</mutationThreshold>
          <skipFailingTests>true</skipFailingTests>
          <timeoutFactor>3.0</timeoutFactor>
          <timeoutConstant>4000</timeoutConstant>
          <jvmArgs>
            <jvmArg>-Xmx{_xml_escape(jvm_max_heap)}</jvmArg>
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
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            lowered = {str(c).strip().lower(): c for c in reader.fieldnames}
            for candidate in PIT_STATUS_COLUMN_CANDIDATES:
                if candidate in lowered:
                    key = lowered[candidate]
                    return [str(row.get(key, "")).strip().upper() for row in reader]

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


def _is_oom_exit(exit_code: int | None) -> bool:
    """Return True when exit code indicates OS-level OOM or process kill."""
    return exit_code is not None and exit_code in _OOM_EXIT_CODES


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
    if _is_oom_exit(exit_code):
        return "oom"
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


def _head(text: str, max_chars: int = 2000) -> str:
    """Capture the beginning of output — useful for compile errors that print early."""
    if not text:
        return ""
    return text[:max_chars]


def _head_and_tail(text: str, head_chars: int = 1500, tail_chars: int = 2500) -> str:
    """Return head + '...' + tail for compile-error diagnosis."""
    if not text:
        return ""
    if len(text) <= head_chars + tail_chars:
        return text
    return text[:head_chars] + "\n[...]\n" + text[-tail_chars:]


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
        "project", "path", "status", "test_mode", "target_classes", "test_sources",
        "killed", "survived", "no_coverage", "timed_out", "memory_error", "non_viable",
        "total_mutations", "total_relevant", "tests_ran", "score", "test_strength",
        "runtime_seconds", "pit_csv", "workdir", "exit_code", "error", "failure_reason",
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
                "project": r.project, "score": r.score, "test_strength": r.test_strength,
                "killed": r.killed, "survived": r.survived, "no_coverage": r.no_coverage, "total_relevant": r.total_relevant,
            } for r in top_projects
        ],
        "worst_projects": [
            {
                "project": r.project, "score": r.score, "test_strength": r.test_strength,
                "killed": r.killed, "survived": r.survived, "no_coverage": r.no_coverage, "total_relevant": r.total_relevant,
            } for r in worst_projects
        ],
        "failed_projects": [
            {
                "project": r.project, "error": r.error, "exit_code": r.exit_code, "failure_reason": r.failure_reason,
            } for r in failed
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
    project_dir: Path, test_mode: str, context: tuple[list[str], list[Path]],
    runtime_seconds: float, error: str, failure_reason: str, *,
    stats: PitStats | None = None, tests_ran: int = -1, pit_csv: str = "",
    workdir: str = "", exit_code: int | None = None, stdout_tail: str = "", stderr_tail: str = "",
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
    project_dir: Path, test_mode: str, target_classes: list[str], test_sources: list[Path],
    stats: PitStats, tests_ran: int, runtime_seconds: float, pit_csv: Path,
    project_workdir: Path, exit_code: int, stdout: str, stderr: str,
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


def _run_pit(
    *,
    project_dir: Path,
    project_workdir: Path,
    test_sources: list[Path],
    target_classes: list[str],
    jars: list[Path],
    args: argparse.Namespace,
    mvn_cmd: str,
    source_level: str,
    jvm_max_heap: str,
    attempt_label: str,
) -> tuple[int | None, str, str, str, Path | None]:
    """
    Execute one Maven/PIT run and return (exit_code, stdout, stderr, error, pit_csv).

    Extracted so _run_single_project can call it multiple times (retry logic).
    """
    evo_tests_dir = project_dir / "evosuite-tests"
    include_evosuite_runtime = evo_tests_dir.exists() and args.test_mode in ("evosuite", "both")

    effective_test_sources = _prepare_test_sources(
        test_sources, project_workdir,
        target_classes=target_classes,
        include_shims=include_evosuite_runtime,
    )

    reports_dir = project_workdir / "pit-reports"
    pom_path = _generate_temp_pom(
        project_dir=project_dir,
        workdir=project_workdir,
        test_sources=effective_test_sources,
        jars=jars,
        target_classes=target_classes,
        reports_dir=reports_dir,
        source_level=source_level,
        target_level=source_level,
        source_encoding=args.source_encoding,
        pit_version=args.pit_version,
        include_evosuite_runtime=include_evosuite_runtime,
        jvm_max_heap=jvm_max_heap,
        mutators=getattr(args, "mutators_list", None),
    )

    cmd = [
        mvn_cmd, "-f", str(pom_path), "clean", "test-compile",
        "org.pitest:pitest-maven:mutationCoverage", "-DtimestampedReports=false",
    ]

    print(f"  [{attempt_label}] Xmx={jvm_max_heap} src={source_level} classes={len(target_classes)}")
    exit_code, stdout, stderr, error = _run_command(
        cmd, cwd=project_workdir, timeout_seconds=args.timeout_seconds
    )
    pit_csv = _find_latest_mutations_csv(reports_dir)
    return exit_code, stdout, stderr, error, pit_csv


def _load_guided_targets(guided_dir: Path | None, project_name: str) -> set[str] | None:
    """
    Read the per-project guided class list (one FQCN per line) produced by
    guided_class_selector.py. Returns None when no list exists, meaning the
    project should run unguided.
    """
    if not guided_dir:
        return None
    f = guided_dir / f"{project_name}.classes"
    if not f.exists():
        return None
    lines = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return set(lines) if lines else None


def _filter_with_guided(target_classes: list[str], guided_fqcns: set[str]) -> list[str]:
    """
    Intersect the test-derived target classes with the predictor's selection.

    Each test-derived FQCN may use slightly different package shorthand than the
    structural FQCN, so we match conservatively by either the full FQCN or by
    the simple class name suffix.
    """
    guided_simple = {fqcn.split(".")[-1] for fqcn in guided_fqcns}
    kept = []
    for cls in target_classes:
        if cls in guided_fqcns or cls.split(".")[-1] in guided_simple:
            kept.append(cls)
    return kept


def _run_single_project(
    project_dir: Path, idx: int, selected_total: int, args: argparse.Namespace,
    mvn_cmd: str, work_root: Path,
) -> RunResult:
    start = time.time()
    test_sources = _collect_test_sources(project_dir, args.test_mode)
    target_classes = _derive_target_classes_from_tests(
        test_sources, max_classes=args.max_classes, exclude_risky=True
    )
    if not target_classes:
        target_classes = _derive_target_classes(project_dir)

    # Guided selection: intersect derived classes with the predictor's whitelist.
    # Falls back to the full list if the intersection is empty (so we never produce
    # zero-target runs from a too-strict predictor).
    guided_dir = Path(args.guided_targets_dir) if args.guided_targets_dir else None
    guided_fqcns = _load_guided_targets(guided_dir, project_dir.name)
    if guided_fqcns:
        before = len(target_classes)
        filtered = _filter_with_guided(target_classes, guided_fqcns)
        if filtered:
            target_classes = filtered
            print(f"  [guided] {project_dir.name}: {before} -> {len(target_classes)} classes")
        else:
            print(f"  [guided] {project_dir.name}: sem intersecao, mantendo todas ({before} classes)")

    if not test_sources:
        return _build_failure_result(
            project_dir=project_dir, test_mode=args.test_mode, context=(target_classes, test_sources),
            runtime_seconds=round(time.time() - start, 2),
            error="No compatible test sources found for selected test mode",
            failure_reason="no_test_sources",
        )

    if not target_classes:
        return _build_failure_result(
            project_dir=project_dir, test_mode=args.test_mode, context=(target_classes, test_sources),
            runtime_seconds=round(time.time() - start, 2),
            error="Could not infer targetClasses patterns from source tree",
            failure_reason="no_target_classes",
        )

    jars = _collect_local_jars(project_dir)
    safe_name = _safe_name(project_dir)
    project_workdir = work_root / safe_name

    print(f"[{idx}/{selected_total}] Running PIT for {project_dir.name}")

    # ------------------------------------------------------------------
    # Attempt 1 — standard settings
    # ------------------------------------------------------------------
    exit_code, stdout, stderr, error, pit_csv = _run_pit(
        project_dir=project_dir,
        project_workdir=project_workdir,
        test_sources=test_sources,
        target_classes=target_classes,
        jars=jars,
        args=args,
        mvn_cmd=mvn_cmd,
        source_level=args.source_level,
        jvm_max_heap=args.jvm_max_heap,
        attempt_label=f"1/attempt",
    )
    runtime = round(time.time() - start, 2)

    failure_reason = (
        _classify_command_failure(
            exit_code=exit_code, error=error, stdout=stdout, stderr=stderr, pit_csv=pit_csv,
        )
        if (exit_code != 0 or not pit_csv or not pit_csv.exists())
        else None
    )

    # ------------------------------------------------------------------
    # Attempt 2 — OOM retry with reduced classes and more heap
    # ------------------------------------------------------------------
    if failure_reason == "oom" and args.retry_oom and len(target_classes) > args.oom_retry_classes:
        reduced_classes = target_classes[: args.oom_retry_classes]
        print(
            f"  [OOM] Retrying {project_dir.name} with {len(reduced_classes)} classes "
            f"and Xmx={args.oom_retry_heap}"
        )
        exit_code2, stdout2, stderr2, error2, pit_csv2 = _run_pit(
            project_dir=project_dir,
            project_workdir=project_workdir,
            test_sources=test_sources,
            target_classes=reduced_classes,
            jars=jars,
            args=args,
            mvn_cmd=mvn_cmd,
            source_level=args.source_level,
            jvm_max_heap=args.oom_retry_heap,
            attempt_label="2/oom-retry",
        )
        runtime = round(time.time() - start, 2)
        if exit_code2 == 0 and pit_csv2 and pit_csv2.exists():
            # Retry succeeded — use these results
            exit_code, stdout, stderr, error, pit_csv = exit_code2, stdout2, stderr2, error2, pit_csv2
            target_classes = reduced_classes
            failure_reason = None
            print(f"  [OOM] Retry succeeded for {project_dir.name}")
        else:
            failure_reason2 = _classify_command_failure(
                exit_code=exit_code2, error=error2, stdout=stdout2,
                stderr=stderr2, pit_csv=pit_csv2,
            )
            error = f"OOM on attempt 1; retry also failed ({failure_reason2}): {error2}"
            failure_reason = f"oom_retry_failed:{failure_reason2}"

    # ------------------------------------------------------------------
    # Attempt 3 — compile error: try fallback source levels
    # ------------------------------------------------------------------
    if failure_reason == "compile_error" and args.retry_compile:
        fallbacks = [lvl for lvl in _SOURCE_LEVEL_FALLBACKS if lvl != args.source_level]
        for fallback_src in fallbacks:
            print(f"  [COMPILE] Retrying {project_dir.name} with source level {fallback_src}")
            exit_code3, stdout3, stderr3, error3, pit_csv3 = _run_pit(
                project_dir=project_dir,
                project_workdir=project_workdir,
                test_sources=test_sources,
                target_classes=target_classes,
                jars=jars,
                args=args,
                mvn_cmd=mvn_cmd,
                source_level=fallback_src,
                jvm_max_heap=args.jvm_max_heap,
                attempt_label=f"compile-retry-{fallback_src}",
            )
            runtime = round(time.time() - start, 2)
            if exit_code3 == 0 and pit_csv3 and pit_csv3.exists():
                exit_code, stdout, stderr, error, pit_csv = exit_code3, stdout3, stderr3, error3, pit_csv3
                failure_reason = None
                print(f"  [COMPILE] Retry with {fallback_src} succeeded for {project_dir.name}")
                break
            if not _looks_like_compile_error(stdout3, stderr3):
                break  # Different error — no point trying more source levels

    # ------------------------------------------------------------------
    # Final failure check
    # ------------------------------------------------------------------
    if failure_reason or exit_code != 0 or not pit_csv or not pit_csv.exists():
        fr = failure_reason or _classify_command_failure(
            exit_code=exit_code, error=error, stdout=stdout, stderr=stderr, pit_csv=pit_csv,
        )
        # For compile errors, capture head (errors appear at the start of output)
        captured = (
            _head_and_tail(stdout) if "compile_error" in fr
            else _tail(stdout)
        )
        return _build_failure_result(
            project_dir=project_dir, test_mode=args.test_mode, context=(target_classes, test_sources),
            runtime_seconds=runtime, pit_csv=(str(pit_csv).replace("\\", "/") if pit_csv else ""),
            workdir=str(project_workdir).replace("\\", "/"), exit_code=exit_code,
            error=error or "Maven/PIT command failed or mutations.csv not generated",
            failure_reason=fr,
            stdout_tail=captured, stderr_tail=_tail(stderr),
        )

    try:
        stats = _parse_pit_results(pit_csv)
    except Exception as exc:
        return _build_failure_result(
            project_dir=project_dir, test_mode=args.test_mode, context=(target_classes, test_sources),
            runtime_seconds=runtime, pit_csv=str(pit_csv).replace("\\", "/"),
            workdir=str(project_workdir).replace("\\", "/"), exit_code=exit_code,
            error=f"Failed parsing PIT CSV: {exc}", failure_reason="parse_error",
            stdout_tail=_tail(stdout), stderr_tail=_tail(stderr),
        )

    tests_ran = _extract_tests_ran(stdout, stderr)
    if tests_ran is not None and tests_ran == 0:
        return _build_failure_result(
            project_dir=project_dir, test_mode=args.test_mode, context=(target_classes, test_sources),
            runtime_seconds=runtime,
            error="PIT reported 0 executed tests (likely initialization/classpath issue)",
            failure_reason="zero_tests", stats=stats, tests_ran=0,
            pit_csv=str(pit_csv).replace("\\", "/"),
            workdir=str(project_workdir).replace("\\", "/"), exit_code=exit_code,
            stdout_tail=_tail(stdout), stderr_tail=_tail(stderr),
        )

    return _build_success_result(
        project_dir=project_dir, test_mode=args.test_mode, target_classes=target_classes,
        test_sources=test_sources, stats=stats,
        tests_ran=(tests_ran if tests_ran is not None else -1),
        runtime_seconds=runtime, pit_csv=pit_csv, project_workdir=project_workdir,
        exit_code=exit_code, stdout=stdout, stderr=stderr,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large-scale mutation testing on SF110")
    parser.add_argument(
        "--sf110-root", type=str,
        default="python/data/raw/evosuite/sf110/extracted/SF110-20130704-src/SF110-20130704-src",
        help="Root folder containing SF110 project directories",
    )
    parser.add_argument(
        "--output-root", type=str, default="outputs/mutation_experiments/sf110",
        help="Output folder for checkpoint/results/summary",
    )
    parser.add_argument(
        "--test-mode", choices=["user", "evosuite", "both"], default="evosuite",
        help="Which tests to compile/use for PIT",
    )
    parser.add_argument("--max-projects", type=int, default=0, help="Max projects to execute (0 = all)")
    parser.add_argument("--project-filter", type=str, default=None, help="Only run projects containing this text")
    parser.add_argument("--timeout-seconds", type=int, default=1200, help="Per-project timeout in seconds")
    parser.add_argument("--source-level", type=str, default="1.8", help="Maven compiler source level")
    parser.add_argument("--target-level", type=str, default="1.8", help="Maven compiler target level")
    parser.add_argument("--source-encoding", type=str, default="ISO-8859-1", help="Source encoding")
    # CORREÇÃO: Alterado por omissão para 1.7.3 para assegurar compatibilidade estrita com Java 8 legado
    parser.add_argument("--pit-version", type=str, default="1.7.3", help="pitest-maven version")
    parser.add_argument("--max-classes", type=int, default=60, help="Max target classes per project (0 = unlimited)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if it exists")
    parser.add_argument("--reset", action="store_true", help="Delete output root before running")
    parser.add_argument(
        "--stop-on-error", action="store_true", help="Stop immediately on first project failure (default continues)",
    )
    # Memory and retry
    parser.add_argument(
        "--jvm-max-heap", type=str, default="1536m",
        help="JVM -Xmx for PIT worker (default: 1536m). Increase if you see OOM failures.",
    )
    parser.add_argument(
        "--retry-oom", action="store_true", default=True,
        help="Automatically retry projects that fail with OOM using reduced class set (default: enabled)",
    )
    parser.add_argument(
        "--no-retry-oom", dest="retry_oom", action="store_false",
        help="Disable OOM retry",
    )
    parser.add_argument(
        "--oom-retry-classes", type=int, default=10,
        help="Max target classes on OOM retry (default: 10). Smaller = less memory needed.",
    )
    parser.add_argument(
        "--oom-retry-heap", type=str, default="2048m",
        help="JVM -Xmx used on OOM retry (default: 2048m).",
    )
    parser.add_argument(
        "--retry-compile", action="store_true", default=False,
        help="Retry compile errors with fallback Java source levels (1.7, 1.6). Slower but may recover some projects.",
    )
    parser.add_argument(
        "--guided-targets-dir", type=str, default=None,
        help="Directory with per-project .classes files produced by guided_class_selector.py. "
             "When set, PIT targets are intersected with the predictor's whitelist.",
    )
    parser.add_argument(
        "--mutators", type=str, default=None,
        help="Comma-separated PIT mutator list to use. Examples: "
             "'STRONGER' (PIT default stronger set), "
             "'DEFAULTS' (current default), "
             "'CONDITIONALS_BOUNDARY,NEGATE_CONDITIONALS,MATH,INCREMENTS,INVERT_NEGS,RETURN_VALS,EMPTY_RETURNS,FALSE_RETURNS,TRUE_RETURNS,NULL_RETURNS,PRIMITIVE_RETURNS' "
             "(curated subset, drops weak mutators like VOID_METHOD_CALLS). "
             "Leave empty for PIT version default.",
    )
    args = parser.parse_args()
    args.mutators_list = [m.strip() for m in args.mutators.split(",")] if args.mutators else None
    return args


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
    projects: list[Path], args: argparse.Namespace, mvn_cmd: str, work_root: Path,
    checkpoint_path: Path, results_csv: Path, summary_json: Path,
    run_results: list[RunResult], processed: set[str],
) -> list[RunResult]:
    selected_total = len(projects)

    for idx, project_dir in enumerate(projects, start=1):
        project_name = project_dir.name
        if project_name in processed:
            print(f"[SKIP] {project_name} already in checkpoint")
            continue

        try:
            result = _run_single_project(
                project_dir=project_dir, idx=idx, selected_total=selected_total,
                args=args, mvn_cmd=mvn_cmd, work_root=work_root,
            )
        except Exception as exc:
            result = _build_failure_result(
                project_dir=project_dir, test_mode=args.test_mode, context=([], []),
                runtime_seconds=0.0, error=f"Unhandled runner exception: {exc}",
                failure_reason="command_failed", stderr_tail=_tail(repr(exc)),
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
    summary: dict[str, Any], selected_total: int, results_csv: Path,
    summary_json: Path, checkpoint_path: Path,
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
        projects=all_projects, args=args, mvn_cmd=mvn_cmd, work_root=work_root,
        checkpoint_path=checkpoint_path, results_csv=results_csv, summary_json=summary_json,
        run_results=run_results, processed=processed,
    )

    _write_results_csv(run_results, results_csv)
    summary = _build_summary(run_results, args, selected_total)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary, selected_total, results_csv, summary_json, checkpoint_path)


if __name__ == "__main__":
    main()