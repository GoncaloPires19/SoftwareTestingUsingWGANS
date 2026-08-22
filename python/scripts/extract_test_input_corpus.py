#!/usr/bin/env python3
"""
Tarefa 5, passo 1 — build the training corpus of TEST INPUTS.

The WGAN in this project previously learnt aggregate test *metrics* (methods,
branches, statements). Objective 3 of the plan ("investigar o impacto dos
conjuntos de testes gerados pela GAN na cobertura do código e deteção de bugs")
needs the GAN to generate actual test *inputs* instead, so that generated tests
can be executed and measured.

This script harvests the inputs the existing suites already exercise (manual +
EvoSuite) from the JUnit sources, giving the GAN a corpus of "interesting"
inputs (boundaries, special cases) rather than a uniform random one.

Representation — every call is encoded as a fixed 12-dim vector so a single
generator can serve all method families (decoded per family downstream):

    [ arr_len, v0..v7, s0, s1, s2 ]

    arr_len : int[] argument length (0..8; -1 encodes a null array)
    v0..v7  : int[] element values (unused slots = 0)
    s0..s2  : scalar int arguments (unused slots = 0)

Method families:
    ARRAY      sum, max, min, average, median, reverseArray, bubbleSort
    ARRAY_INT  countOccurrences(int[], int)          -> uses s0
    INT1       isPrime, factorial, fibonacci, isPerfectSquare
    INT2       gcd(int, int)                          -> uses s0, s1
    INT3       clamp(int, int, int)                   -> uses s0, s1, s2

isPalindrome(String) is out of scope for generation (non-numeric); it is still
covered by the manual/EvoSuite arms and is reported separately.

Circularity guard — the generated suites live in the same directory this script
harvests, because that is where Maven expects test sources. Re-running the
extraction after an experiment would therefore fold hundreds of machine-made
inputs back into the corpus the generators are trained on, and every subsequent
comparison would be measuring a model against its own output. The generated
suites are skipped by name (see GENERATED_SUITES) and the ones actually read
are printed, so the corpus provenance is visible in the log rather than assumed.

Usage:
    python python/scripts/extract_test_input_corpus.py \
        --test-dir src/test/java/dataloader \
        --out outputs/task5/input_corpus.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

MAX_ARR = 8

FAMILY_OF: dict[str, str] = {
    "sum": "ARRAY", "max": "ARRAY", "min": "ARRAY", "average": "ARRAY",
    "median": "ARRAY", "reverseArray": "ARRAY", "bubbleSort": "ARRAY",
    "countOccurrences": "ARRAY_INT",
    "isPrime": "INT1", "factorial": "INT1", "fibonacci": "INT1",
    "isPerfectSquare": "INT1",
    "gcd": "INT2",
    "clamp": "INT3",
}

_CALL_RE = re.compile(r"MathOperations\.([a-zA-Z]+)\s*\(")
_NEW_ARR_LITERAL = re.compile(r"^new\s+int\s*\[\s*\]\s*\{([^}]*)\}$")
_NEW_ARR_SIZED = re.compile(r"^new\s+int\s*\[\s*(\d+)\s*\]$")
_INT_LITERAL = re.compile(r"^-?\d+$")

FIELDNAMES = ["method", "family", "arr_len"] + [f"v{i}" for i in range(MAX_ARR)] + ["s0", "s1", "s2"]

# Suites emitidas por emit_junit_from_oracle.py, uma por braco gerado. Vivem no
# mesmo diretorio das suites humanas porque e onde o Maven procura fontes de
# teste, por isso tem de ser excluidas explicitamente: incluir uma delas no
# corpus faria os geradores treinar sobre o proprio output.
GENERATED_SUITES = frozenset({
    # O nome antigo fica na lista: se um ficheiro de uma execucao anterior ao
    # rename sobreviver na arvore, tem de continuar a ser excluido do corpus.
    "MathOperationsGanTests.java",
    "MathOperationsWganTests.java",
    "MathOperationsRandomTests.java",
    "MathOperationsCorpusTests.java",
    "MathOperationsKdeTests.java",
})


def _split_args(text: str) -> list[str] | None:
    """Split a call's argument list on top-level commas. None if unbalanced."""
    args, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if depth != 0:
        return None
    tail = "".join(cur).strip()
    if tail:
        args.append(tail)
    return args


def _extract_call_args(source: str, open_paren_idx: int) -> str | None:
    """Return the raw text between the call's parentheses, or None if unbalanced."""
    depth = 0
    for i in range(open_paren_idx, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[open_paren_idx + 1:i]
    return None


def _parse_array(arg: str) -> list[int] | None:
    """Parse an int[] argument literal. Returns None if not a parseable literal."""
    arg = arg.strip()
    if arg == "null":
        return None  # caller distinguishes null via a sentinel
    m = _NEW_ARR_LITERAL.match(arg)
    if m:
        body = m.group(1).strip()
        if not body:
            return []
        vals = []
        for part in body.split(","):
            part = part.strip()
            if not _INT_LITERAL.match(part):
                return None
            vals.append(int(part))
        return vals
    m = _NEW_ARR_SIZED.match(arg)
    if m:
        return [0] * int(m.group(1))
    return None


def _parse_int(arg: str) -> int | None:
    arg = arg.strip()
    return int(arg) if _INT_LITERAL.match(arg) else None


def _encode(method: str, args: list[str]) -> dict | None:
    """Encode a parsed call into the 12-dim row, or None if not fully literal."""
    family = FAMILY_OF[method]
    row = {"method": method, "family": family, "arr_len": 0,
           **{f"v{i}": 0 for i in range(MAX_ARR)}, "s0": 0, "s1": 0, "s2": 0}

    if family in ("ARRAY", "ARRAY_INT"):
        if not args:
            return None
        raw = args[0].strip()
        if raw == "null":
            row["arr_len"] = -1
        else:
            vals = _parse_array(raw)
            if vals is None or len(vals) > MAX_ARR:
                return None
            row["arr_len"] = len(vals)
            for i, v in enumerate(vals):
                row[f"v{i}"] = v
        if family == "ARRAY_INT":
            if len(args) < 2:
                return None
            s0 = _parse_int(args[1])
            if s0 is None:
                return None
            row["s0"] = s0
        return row

    n_scalars = {"INT1": 1, "INT2": 2, "INT3": 3}[family]
    if len(args) < n_scalars:
        return None
    for i in range(n_scalars):
        val = _parse_int(args[i])
        if val is None:
            return None
        row[f"s{i}"] = val
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest test inputs from existing JUnit suites into a GAN training corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-dir", type=str, default="src/test/java/dataloader")
    parser.add_argument("--out", type=str, default="outputs/task5/input_corpus.csv")
    parser.add_argument("--dedup", action="store_true", default=True,
                        help="Drop duplicate input vectors (same method + same values)")
    parser.add_argument("--allow-generated", action="store_true",
                        help="Harvest the generated suites too. Only for diagnostics: "
                             "it makes the corpus circular and invalidates the experiment.")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"[ERRO] Test dir not found: {test_dir}")
        sys.exit(1)

    sources = sorted(test_dir.glob("*.java"))
    if not args.allow_generated:
        skipped = [p.name for p in sources if p.name in GENERATED_SUITES]
        sources = [p for p in sources if p.name not in GENERATED_SUITES]
        if skipped:
            print(f"  Suites geradas ignoradas: {', '.join(skipped)}")
    if not sources:
        print(f"[ERRO] Nenhuma suite humana em {test_dir}")
        sys.exit(1)
    print(f"  Corpus extraido de: {', '.join(p.name for p in sources)}\n")

    rows: list[dict] = []
    seen: set[tuple] = set()
    stats = Counter()

    for java_file in sources:
        source = java_file.read_text(encoding="utf-8", errors="replace")
        for m in _CALL_RE.finditer(source):
            method = m.group(1)
            stats["call_sites"] += 1
            if method not in FAMILY_OF:
                stats[f"skipped_{method}"] += 1
                continue
            raw_args = _extract_call_args(source, m.end() - 1)
            if raw_args is None:
                stats["unbalanced"] += 1
                continue
            parts = _split_args(raw_args)
            if parts is None:
                stats["unbalanced"] += 1
                continue
            row = _encode(method, parts)
            if row is None:
                stats["not_literal"] += 1
                continue
            key = tuple(row[f] for f in FIELDNAMES)
            if args.dedup and key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            rows.append(row)
            stats["extracted"] += 1

    if not rows:
        print("[ERRO] Nenhum input extraido.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 72)
    print("  CORPUS DE INPUTS — extraido dos testes existentes")
    print("=" * 72)
    print(f"\n  Call sites encontrados : {stats['call_sites']}")
    print(f"  Extraidos (literais)   : {stats['extracted']}")
    print(f"  Duplicados removidos   : {stats['duplicate']}")
    print(f"  Nao-literais (variavel): {stats['not_literal']}")
    skipped = {k: v for k, v in stats.items() if k.startswith("skipped_")}
    for k, v in sorted(skipped.items()):
        print(f"  Fora de scope ({k[8:]:<14}): {v}")

    by_family = Counter(r["family"] for r in rows)
    by_method = Counter(r["method"] for r in rows)
    print(f"\n  {'Familia':<12}{'Inputs':>8}")
    print(f"  {'-'*20}")
    for fam, n in sorted(by_family.items()):
        print(f"  {fam:<12}{n:>8}")

    print(f"\n  {'Metodo':<18}{'Inputs':>8}")
    print(f"  {'-'*26}")
    for meth, n in sorted(by_method.items(), key=lambda t: -t[1]):
        print(f"  {meth:<18}{n:>8}")

    print(f"\n  CSV: {out_path}")


if __name__ == "__main__":
    main()
