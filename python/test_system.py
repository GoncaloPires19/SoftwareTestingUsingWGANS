#!/usr/bin/env python3
"""Quick repository sanity checks for Python + data pipeline prerequisites."""

from pathlib import Path
import importlib
import sys


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "torch",
    "scipy",
    "matplotlib",
    "seaborn",
]

REQUIRED_PATHS = [
    Path("python/scripts/train_basic_profiles.py"),
    Path("python/scripts/generate_synthetic.py"),
    Path("python/scripts/experimental_analysis.py"),
    Path("python/data/junit_data_cleaned_strict.csv"),
]


def check_modules() -> list[str]:
    failures = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"Module import failed ({module_name}): {exc}")
    return failures


def check_paths() -> list[str]:
    failures = []
    for path in REQUIRED_PATHS:
        if not path.exists():
            failures.append(f"Required path missing: {path}")
    return failures


def main() -> int:
    failures = []
    failures.extend(check_modules())
    failures.extend(check_paths())

    if failures:
        print("System check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("System check passed.")
    print("- Core Python dependencies import correctly")
    print("- Key scripts and dataset files exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
