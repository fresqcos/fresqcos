#!/usr/bin/env python
"""
Run code-quality checks for the project.

This script wraps:
- pylint          (style / linting)
- mypy            (static type checking)
- docstr-coverage (docstring coverage)

Run from the repo root:

    python code_quality.py
"""

from __future__ import annotations

import subprocess
from typing import List, Tuple


Command = Tuple[str, List[str]]


def run_command(name: str, cmd: List[str]) -> int:
    """Run a command, stream output, and return its exit code."""
    print(f"\n=== Running {name}: {' '.join(cmd)} ===")
    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print(f"[WARNING] {name} is not installed or not on PATH.")
        return 1
    return result.returncode


def main() -> int:
    commands: List[Command] = [
        (
            "pylint",
            ["pylint", "fresqcos"],
        ),
        (
            "mypy",
            ["mypy"],
        ),
        (
            "docstr-coverage",
            [
                "docstr-coverage",
                "fresqcos",
                "--exclude",
                "--skip-private",
                "--skip-magic",
            ],
        ),
    ]

    overall_rc = 0
    for name, cmd in commands:
        rc = run_command(name, cmd)
        if rc != 0:
            overall_rc = rc

    if overall_rc == 0:
        print("\n✅ All code quality checks passed.")
    else:
        print("\n❌ Some checks failed (see output above).")

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
