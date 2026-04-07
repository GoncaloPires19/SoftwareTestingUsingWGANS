#!/usr/bin/env python3
"""Compatibility trainer that delegates to train_basic_profiles optimized profile."""

import subprocess
import sys


def main() -> int:
    passthrough_args = sys.argv[1:]
    if "--profile" not in passthrough_args:
        passthrough_args = ["--profile", "optimized"] + passthrough_args

    cmd = [sys.executable, "python/scripts/train_basic_profiles.py"] + passthrough_args
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
