from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fixture_path(name: str) -> Path:
    return repo_root() / "tests" / "fixtures" / name


def run_cli_json(args: list[str], cwd: Path) -> tuple[int, dict[str, Any], str]:
    completed = subprocess.run(
        [sys.executable, "cli_calculator.py", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )

    stdout = completed.stdout
    stderr = completed.stderr

    if not stdout:
        raise AssertionError(f"CLI produced no stdout. stderr:\n{stderr}")

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        snippet = stdout[:500]
        raise AssertionError(
            "CLI returned non-JSON stdout."
            f" stdout[:500]=\n{snippet}\n"
            f"stderr:\n{stderr}"
        ) from exc

    if not isinstance(payload, dict):
        raise AssertionError(
            "CLI JSON payload is not an object."
            f" type={type(payload).__name__} stderr:\n{stderr}"
        )

    return completed.returncode, payload, stderr
