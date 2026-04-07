#!/usr/bin/env python3
"""Compatibility entry point that forwards to the archived profile-aware trainer."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    archived_script = Path(__file__).parent / "archive" / "train_profile_aware.py"
    if not archived_script.exists():
        print(f"Archived script not found: {archived_script}")
        return 1

    cmd = [sys.executable, str(archived_script)] + sys.argv[1:]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
