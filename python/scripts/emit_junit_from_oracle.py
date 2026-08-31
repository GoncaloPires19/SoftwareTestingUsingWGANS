#!/usr/bin/env python3
"""
Tarefa 5, passo 4 — emit an executable JUnit 5 suite from oracle records.

Takes the CSV produced by tools.OracleRunner (input vector + the outcome the
ORIGINAL implementation produced) and writes a JUnit test class where every
expected value is a compile-time literal. That literal is what lets the suite
kill mutants: a self-referential assertion would mutate on both sides and never
fail.

Emitted into package `dataloader` so the PIT config (targetTests=dataloader.*)
picks the suite up.

Usage:
    python python/scripts/emit_junit_from_oracle.py \
        --oracle outputs/task5/oracle_gan.csv \
        --class-name MathOperationsWganTests \
        --out src/test/java/dataloader/MathOperationsWganTests.java
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# method -> (return kind, argument kind)
RETURN_KIND = {
    "sum": "int", "max": "int", "min": "int", "countOccurrences": "int",
    "gcd": "int", "clamp": "int",
    "average": "double", "median": "double",
    "factorial": "long", "fibonacci": "long",
    "isPrime": "boolean", "isPerfectSquare": "boolean",
    "reverseArray": "int[]", "bubbleSort": "int[]",
}

ARG_KIND = {
    "sum": "ARRAY", "max": "ARRAY", "min": "ARRAY", "average": "ARRAY",
    "median": "ARRAY", "reverseArray": "ARRAY", "bubbleSort": "ARRAY",
    "countOccurrences": "ARRAY_INT",
    "isPrime": "INT1", "factorial": "INT1", "fibonacci": "INT1",
    "isPerfectSquare": "INT1",
    "gcd": "INT2", "clamp": "INT3",
}


def _array_literal(row: dict) -> str:
    arr_len = int(row["arr_len"])
    if arr_len < 0:
        return "null"
    if arr_len == 0:
        return "new int[0]"
    vals = ", ".join(row[f"v{i}"] for i in range(arr_len))
    return f"new int[]{{{vals}}}"


def _call_args(method: str, row: dict, target: dict = None) -> str:
    kind = (target or TARGETS["mathops"])["arg_kind"][method]
    if kind == "ARRAY":
        return _array_literal(row)
    if kind == "ARRAY_INT":
        return f"{_array_literal(row)}, {row['s0']}"
    if kind == "INT1":
        return row["s0"]
    if kind == "INT2":
        return f"{row['s0']}, {row['s1']}"
    if kind == "LONG2":
        # O sufixo L e o que faz o compilador escolher a sobrecarga long
        return f"{row['s0']}L, {row['s1']}L"
    return f"{row['s0']}, {row['s1']}, {row['s2']}"


def _assertion(method: str, row: dict, target: dict = None) -> str | None:
    """Build the assertion body, or None if the row can't be turned into a test."""
    target = target or TARGETS["mathops"]
    java = target.get("java_name", {}).get(method, method)
    call = f"{target['class']}.{java}({_call_args(method, row, target)})"

    if row["outcome"] == "EXCEPTION":
        return f"assertThrows({row['value']}.class, () -> {call});"

    kind = target["return_kind"][method]
    val = row["value"]

    if kind == "int":
        return f"assertEquals({val}, {call});"
    if kind == "long":
        return f"assertEquals({val}L, {call});"
    if kind == "double":
        if val in ("NaN", "Infinity", "-Infinity"):
            return None  # not expressible as a plain literal assertion
        return f"assertEquals({val}, {call}, 1e-9);"
    if kind == "boolean":
        return f"assertTrue({call});" if val == "true" else f"assertFalse({call});"
    if kind == "int[]":
        if val == "null":
            return f"assertNull({call});"
        elems = ", ".join(val.split(";")) if val else ""
        expected = f"new int[]{{{elems}}}" if elems else "new int[0]"
        return f"assertArrayEquals({expected}, {call});"
    return None


# Segundo alvo experimental. Seis metodos (int, int) que devolvem int, todos
# passiveis de lancar excepcao por overflow -- e por isso metade das assercoes
# emitidas sao assertThrows, ao contrario do primeiro alvo onde sao a excepcao.
ARITH_RETURN_KIND = {
    "gcd": "int", "lcm": "int", "pow": "int",
    "addAndCheck": "int", "subAndCheck": "int", "mulAndCheck": "int",
    # Sufixo L: sobrecarga long da mesma operacao, invocada com os argumentos
    # promovidos. Sem elas, dois tercos das mutacoes da classe ficariam sem
    # cobertura para todos os bracos.
    "gcdL": "long", "lcmL": "long", "powL": "long",
    "addAndCheckL": "long", "subAndCheckL": "long", "mulAndCheckL": "long",
}

# Nome Java real de cada entrada do oraculo (o sufixo L nao existe na classe).
ARITH_JAVA_NAME = {m: m[:-1] if m.endswith("L") else m for m in ARITH_RETURN_KIND}

TARGETS = {
    "mathops": {
        "class": "MathOperations",
        "return_kind": RETURN_KIND,
        "arg_kind": ARG_KIND,
        "input_cols": ["arr_len"] + [f"v{i}" for i in range(8)] + ["s0", "s1", "s2"],
        "imports": "",
    },
    "arith": {
        "class": "ArithmeticUtils",
        "return_kind": ARITH_RETURN_KIND,
        "arg_kind": {m: ("LONG2" if m.endswith("L") else "INT2")
                     for m in ARITH_RETURN_KIND},
        "java_name": ARITH_JAVA_NAME,
        "input_cols": ["s0", "s1"],
        # As excepcoes de overflow sao resultado esperado neste alvo, e cerca de
        # um em cada dezasseis testes emitidos e um assertThrows sobre elas.
        "imports": "import org.apache.commons.math3.exception.MathArithmeticException;\nimport org.apache.commons.math3.exception.NotPositiveException;\nimport org.apache.commons.math3.exception.NumberIsTooLargeException;\n",
    },
}


HEADER = """package dataloader;

import org.junit.jupiter.api.Test;
{imports}
import static org.junit.jupiter.api.Assertions.*;

/**
 * AUTO-GENERATED — do not edit by hand.
 *
 * <p>Generated by python/scripts/emit_junit_from_oracle.py for Tarefa 5
 * (impact of GAN-generated test data on coverage and defect detection).
 *
 * <p>Arm: {arm}
 * <p>Inputs: {n_inputs} | Tests: {n_tests}
 *
 * <p>Every expected value is a literal recorded from the original
 * implementation (regression oracle), so mutants that change behaviour fail.
 */
class {class_name} {{
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit a JUnit 5 suite from OracleRunner records",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--oracle", type=str, required=True)
    parser.add_argument("--class-name", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--arm", type=str, default="")
    parser.add_argument("--target", type=str, default="mathops",
                        choices=sorted(TARGETS))
    args = parser.parse_args()
    target = TARGETS[args.target]

    oracle_path = Path(args.oracle)
    if not oracle_path.exists():
        print(f"[ERRO] Oracle CSV not found: {oracle_path}")
        sys.exit(1)

    with open(oracle_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    bodies: list[str] = []
    stats = Counter()
    per_method = Counter()
    seen_inputs = set()

    for idx, row in enumerate(rows):
        method = row["method"]
        if method not in target["return_kind"]:
            stats["unknown_method"] += 1
            continue
        assertion = _assertion(method, row, target)
        if assertion is None:
            stats["skipped"] += 1
            continue
        seen_inputs.add(tuple(row[c] for c in target["input_cols"]))
        per_method[method] += 1
        stats["emitted"] += 1
        bodies.append(
            f"    @Test\n"
            f"    void {method}_{idx}() {{\n"
            f"        {assertion}\n"
            f"    }}\n"
        )

    if not bodies:
        print("[ERRO] Nenhum teste emitido.")
        sys.exit(1)

    arm = args.arm or oracle_path.stem
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(HEADER.format(imports=target["imports"], arm=arm,
                              class_name=args.class_name,
                              n_inputs=len(seen_inputs), n_tests=len(bodies)))
        f.write("\n".join(bodies))
        f.write("}\n")

    print("=" * 72)
    print(f"  SUITE GERADA — {args.class_name}")
    print("=" * 72)
    print(f"\n  Registos oracle : {len(rows)}")
    print(f"  Testes emitidos : {stats['emitted']}")
    print(f"  Ignorados       : {stats['skipped']}")
    print(f"  Inputs distintos: {len(seen_inputs)}")
    print(f"\n  {'Metodo':<18}{'Testes':>8}")
    print(f"  {'-'*26}")
    for method, n in sorted(per_method.items(), key=lambda t: -t[1]):
        print(f"  {method:<18}{n:>8}")
    print(f"\n  Ficheiro: {out_path}")


if __name__ == "__main__":
    main()
