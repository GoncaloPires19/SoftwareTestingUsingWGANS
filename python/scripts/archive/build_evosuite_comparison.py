#!/usr/bin/env python3
"""Build a complete with-vs-without EvoSuite comparison JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build with/without EvoSuite comparison JSON")
    parser.add_argument(
        "--with-file",
        default="outputs/mutation_experiments/with_evosuite_only.json",
        help="Path to with_evosuite JSON",
    )
    parser.add_argument(
        "--without-file",
        default="outputs/mutation_experiments/without_evosuite_only.json",
        help="Path to without_evosuite JSON",
    )
    parser.add_argument(
        "--output",
        default="outputs/mutation_experiments/with_without_evosuite_comparison.json",
        help="Output comparison JSON path",
    )
    args = parser.parse_args()

    with_data = load_json(Path(args.with_file))
    without_data = load_json(Path(args.without_file))

    delta = {
        "score": round(float(with_data.get("score", 0.0)) - float(without_data.get("score", 0.0)), 4),
        "killed": int(with_data.get("killed", 0)) - int(without_data.get("killed", 0)),
        "survived": int(with_data.get("survived", 0)) - int(without_data.get("survived", 0)),
        "no_coverage": int(with_data.get("no_coverage", 0)) - int(without_data.get("no_coverage", 0)),
        "total": int(with_data.get("total", 0)) - int(without_data.get("total", 0)),
    }

    output = {
        "comparison": {
            "with_evosuite": with_data,
            "without_evosuite": without_data,
            "delta_with_minus_without": delta,
        },
        "interpretation": (
            "No aggregate effect from EvoSuite in this snapshot"
            if delta["score"] == 0.0 and delta["killed"] == 0
            else "Aggregate metrics changed with EvoSuite"
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote comparison: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
