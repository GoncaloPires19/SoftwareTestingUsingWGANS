#!/usr/bin/env python3
"""
Java Class Structural Metric Extractor

Extracts lightweight structural metrics for every class in the SF110 source
tree, producing the *structural* half of the bridge dataset. These metrics are
available BEFORE running mutation testing, so a model trained on them can
predict testability and guide PIT target selection up front.

Metrics per class (matching the WGAN feature space: methods, branches, statements):
  - methods    : number of method/constructor declarations
  - branches   : decision points (if/for/while/case/catch/&&/||/?:)
  - statements : statement count (semicolons outside comments/strings, approx.)
  - loc        : non-blank, non-comment lines of code
  - n_fields   : field declarations (proxy for state/coupling)
  - has_io     : 1 if the class imports java.net / java.io / javax.swing / java.awt
                 (structural signal — NOT used as a hard rule, just a feature)

This is regex-based (no full Java parser dependency) — accurate enough for
coarse structural metrics used as ML features.

Usage:
    python python/scripts/extract_class_metrics.py \
        --sf110-root python/data/raw/evosuite/sf110/extracted/SF110-20130704-src \
        --output outputs/mutation_experiments/class_structural_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_EXT_PREFIX = "\\\\?\\"  # Windows extended-length path prefix (bypasses MAX_PATH)


def _long(p: Path) -> str:
    """Return an extended-length absolute path string on Windows to bypass MAX_PATH."""
    ap = os.path.abspath(str(p))
    if os.name == "nt" and not ap.startswith(_EXT_PREFIX):
        return _EXT_PREFIX + ap
    return ap

# Regex patterns (coarse but stable for metric extraction).
# Kept deliberately simple to avoid catastrophic backtracking on large/minified files.
_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)
_CLASS_DECL_RE = re.compile(r"\b(?:class|interface|enum)\s+(\w+)")
# A method is a `name(...) {` preceded by an access/return-type token. We count
# the simple, linear-time signature shape `\w+\s*\([^()]*\)\s*\{` and subtract
# obvious control-flow keywords that share that shape (if/for/while/switch/catch).
_METHOD_LIKE_RE = re.compile(r"\b(\w+)\s*\([^()]*\)\s*(?:throws[\w,.\s]+?)?\{")
_CONTROL_KEYWORDS = {"if", "for", "while", "switch", "catch", "synchronized"}
_BRANCH_KEYWORDS = re.compile(r"\b(if|for|while|case|catch)\b")
_BOOL_OPS = re.compile(r"(&&|\|\||\?)")
_IO_IMPORT_RE = re.compile(
    r"import\s+(java\.net\.|java\.io\.|javax\.swing|java\.awt|java\.rmi|java\.sql\.)",
)

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')

# --- Richer testability features (v2) ---
# Frameworks that make a class hard/impossible to unit-test when extended.
_UNTESTABLE_SUPERCLASSES = re.compile(
    r"\bextends\s+(JFrame|JPanel|JDialog|JApplet|Applet|Frame|Thread|Window|"
    r"JComponent|Container|TimerTask|Thread|HttpServlet|JWindow)\b"
)
_IMPLEMENTS_RUNNABLE = re.compile(r"\bimplements\s+[\w,\s]*\b(Runnable|ActionListener|MouseListener|WindowListener)\b")
_ABSTRACT_RE = re.compile(r"\b(abstract)\s+(class|interface)\b|\binterface\s+\w+")
_STATIC_METHOD_RE = re.compile(r"\bstatic\s+[\w<>\[\].]+\s+\w+\s*\([^()]*\)\s*\{")
_DEFAULT_CTOR_HINT = re.compile(r"\bpublic\s+(\w+)\s*\(\s*\)")
# distinct imported types as a coupling proxy
_IMPORT_TYPE_RE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", re.MULTILINE)
# constructor with the most parameters (mockability — more deps = harder)
_CTOR_PARAMS_RE = re.compile(r"\b(?:public|protected|private)\s+\w+\s*\(([^()]*)\)\s*(?:throws[\w,.\s]+?)?\{")


def _strip_noise(source: str) -> str:
    """Remove comments and string literals so they don't pollute metric counts."""
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


def _extract_file_metrics(java_file: Path, src_root: Path) -> dict | None:
    try:
        with open(_long(java_file), "r", encoding="ISO-8859-1", errors="ignore") as fh:
            raw = fh.read()
    except OSError:
        return None

    clean = _strip_noise(raw)

    class_match = _CLASS_DECL_RE.search(clean)
    if not class_match:
        return None
    class_simple = class_match.group(1)

    pkg_match = _PACKAGE_RE.search(clean)
    package = pkg_match.group(1) if pkg_match else ""
    fqcn = f"{package}.{class_simple}" if package else class_simple

    # Count method-like signatures, then subtract control-flow keywords that
    # share the `keyword(...) {` shape (linear-time, no backtracking).
    method_like = _METHOD_LIKE_RE.findall(clean)
    methods = sum(1 for name in method_like if name not in _CONTROL_KEYWORDS)
    branches = len(_BRANCH_KEYWORDS.findall(clean)) + len(_BOOL_OPS.findall(clean))
    statements = clean.count(";")
    loc = _count_loc(raw)
    has_io = 1 if _IO_IMPORT_RE.search(raw) else 0

    # field declarations: lines ending in ; at class level with a type and name,
    # approximated as semicolons minus method bodies is unreliable — use a simple
    # heuristic: declarations with access modifier not followed by '('
    n_fields = len(re.findall(
        r"\b(?:private|protected|public)\s+(?:static\s+|final\s+)*[\w<>\[\].]+\s+\w+\s*(?:=|;)",
        clean,
    ))

    # --- Richer testability features (v2) ---
    extends_framework = 1 if _UNTESTABLE_SUPERCLASSES.search(clean) else 0
    implements_callback = 1 if _IMPLEMENTS_RUNNABLE.search(clean) else 0
    is_abstract = 1 if _ABSTRACT_RE.search(clean) else 0
    n_static_methods = len(_STATIC_METHOD_RE.findall(clean))
    has_default_ctor = 1 if _DEFAULT_CTOR_HINT.search(clean) else 0

    # Coupling: distinct non-JDK imported types
    imports = _IMPORT_TYPE_RE.findall(raw)
    non_jdk = [i for i in imports if not i.startswith(("java.", "javax."))]
    n_imports = len(imports)
    n_external_deps = len(set(non_jdk))

    # Mockability: max constructor parameter count
    max_ctor_params = 0
    for params in _CTOR_PARAMS_RE.findall(clean):
        params = params.strip()
        if params:
            max_ctor_params = max(max_ctor_params, params.count(",") + 1)

    # static ratio — pure static utility classes are highly testable
    static_ratio = round(n_static_methods / methods, 3) if methods else 0.0

    return {
        "project": _project_of(java_file, src_root),
        "class_name": fqcn,
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


def _project_of(java_file: Path, sf110_root: Path) -> str:
    """Return the SF110 project directory name for a given java file."""
    try:
        rel = java_file.relative_to(sf110_root)
        return rel.parts[0]
    except ValueError:
        return java_file.parent.name


_MAX_FILE_BYTES = 2_000_000  # skip pathologically large/minified files


def _extract_all(sf110_root: Path) -> list[dict]:
    rows: list[dict] = []
    project_dirs = [p for p in sf110_root.iterdir() if p.is_dir() and (p / "build.xml").exists()]
    total = len(project_dirs)

    for i, proj in enumerate(sorted(project_dirs), start=1):
        src_main = proj / "src" / "main" / "java"
        if not src_main.exists():
            print(f"  [{i:>2}/{total}] {proj.name:<28} (sem src/main/java)", flush=True)
            continue

        java_files = list(src_main.rglob("*.java"))
        n_before = len(rows)
        skipped = 0
        for java_file in java_files:
            try:
                if os.path.getsize(_long(java_file)) > _MAX_FILE_BYTES:
                    skipped += 1
                    continue
            except OSError:
                skipped += 1
                continue
            metrics = _extract_file_metrics(java_file, sf110_root)
            if metrics:
                rows.append(metrics)

        extracted = len(rows) - n_before
        skip_note = f" ({skipped} skip grandes)" if skipped else ""
        print(f"  [{i:>2}/{total}] {proj.name:<28} {extracted:>4} classes{skip_note}", flush=True)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structural metrics per Java class",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sf110-root", type=str,
        default="python/data/raw/evosuite/sf110/extracted/SF110-20130704-src",
        help="SF110 source root",
    )
    parser.add_argument(
        "--output", type=str,
        default="outputs/mutation_experiments/class_structural_metrics.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    sf110_root = Path(args.sf110_root)
    if not sf110_root.exists():
        print(f"[ERRO] SF110 root not found: {sf110_root}")
        sys.exit(1)

    print(f"Extracting structural metrics from {sf110_root}...")
    rows = _extract_all(sf110_root)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "project", "class_name", "methods", "branches", "statements", "loc",
        "n_fields", "has_io", "extends_framework", "implements_callback",
        "is_abstract", "n_static_methods", "static_ratio", "has_default_ctor",
        "n_imports", "n_external_deps", "max_ctor_params",
    ]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} classes from {len({r['project'] for r in rows})} projects")
    print(f"Output: {output}")
    # Quick sanity stats
    if rows:
        io_count = sum(r["has_io"] for r in rows)
        print(f"  Classes com I/O imports: {io_count} ({io_count/len(rows)*100:.0f}%)")
        avg_methods = sum(r["methods"] for r in rows) / len(rows)
        avg_branches = sum(r["branches"] for r in rows) / len(rows)
        print(f"  Media methods/classe: {avg_methods:.1f}, branches/classe: {avg_branches:.1f}")


if __name__ == "__main__":
    main()
