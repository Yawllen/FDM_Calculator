from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from tests.helpers_cli import fixture_path, repo_root, run_cli_json


def _choose_fixture() -> Path:
    for name in ["sub.stl", "Untitled_v1.stl", "ksr_fdmtest_v4.3mf"]:
        candidate = fixture_path(name)
        if candidate.exists():
            return candidate
    pytest.skip("No known fixtures available in tests/fixtures")


def _assert_finite_non_negative(value: Any, label: str) -> None:
    assert isinstance(value, (int, float)) and not isinstance(
        value, bool
    ), f"{label} should be numeric, got {type(value).__name__}"
    assert math.isfinite(value), f"{label} should be finite, got {value}"
    assert value >= 0, f"{label} should be >= 0, got {value}"


def _maybe_assert_numeric_for_key(key: str, value: Any, *, include_time: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    key_lower = key.lower()
    if key_lower == "costs":
        return
    if "volume" in key_lower:
        _assert_finite_non_negative(value, f"{key} (volume)")
    if "gram" in key_lower or "weight" in key_lower:
        _assert_finite_non_negative(value, f"{key} (weight)")
    if include_time and "time" in key_lower:
        _assert_finite_non_negative(value, f"{key} (time)")
    if "total_rub" in key_lower or "cost" in key_lower:
        _assert_finite_non_negative(value, f"{key} (price)")


def test_cli_json_single_file_contract() -> None:
    fixture = _choose_fixture()
    args = [
        str(fixture),
        "--json",
        "--material",
        "Enduse PETG",
        "--infill",
        "10",
        "--qty",
        "1",
        "--setup-min",
        "10",
        "--post-min",
        "0",
    ]

    returncode, payload, stderr = run_cli_json(args, cwd=repo_root())

    assert returncode == 0, f"CLI failed with return code {returncode}. stderr:\n{stderr}"
    assert isinstance(payload, dict), "CLI JSON payload should be an object"
    assert "per_object" in payload, f"JSON payload missing 'per_object'. stderr:\n{stderr}"
    assert "summary" in payload, f"JSON payload missing 'summary'. stderr:\n{stderr}"

    if "success" in payload:
        assert isinstance(payload["success"], bool), "'success' should be a boolean"
    if "errors" in payload:
        assert isinstance(payload["errors"], list), "'errors' should be a list"
    if "count_ok" in payload:
        assert isinstance(payload["count_ok"], int), "'count_ok' should be an int"
    if "count_failed" in payload:
        assert isinstance(payload["count_failed"], int), "'count_failed' should be an int"

    summary = payload["summary"]
    if summary is not None:
        assert isinstance(summary, dict), "'summary' should be an object when present"
        for key, value in summary.items():
            _maybe_assert_numeric_for_key(key, value)

    per_object = payload["per_object"]
    if per_object is not None:
        assert isinstance(per_object, list), "'per_object' should be a list when present"
        assert per_object or summary is not None, (
            "'per_object' should contain entries unless summary is present"
        )
        if per_object:
            first = per_object[0]
            assert isinstance(first, dict), "Per-object item should be an object"
            assert "file" in first, "Per-object item should include 'file'"
            for key, value in first.items():
                _maybe_assert_numeric_for_key(key, value)
    else:
        assert summary is not None, "'summary' should be present when per_object is None"


def test_cli_json_per_object_mode_contract() -> None:
    fixture = _choose_fixture()
    args = [
        str(fixture),
        "--json",
        "--per-object",
        "--material",
        "Enduse PETG",
        "--infill",
        "10",
        "--qty",
        "1",
        "--setup-min",
        "10",
        "--post-min",
        "0",
    ]

    returncode, payload, stderr = run_cli_json(args, cwd=repo_root())

    assert returncode == 0, f"CLI failed with return code {returncode}. stderr:\n{stderr}"
    assert isinstance(payload, dict), "CLI JSON payload should be an object"
    assert "per_object" in payload, f"JSON payload missing 'per_object'. stderr:\n{stderr}"
    assert "summary" in payload, f"JSON payload missing 'summary'. stderr:\n{stderr}"

    per_object = payload["per_object"]
    assert isinstance(per_object, list), "'per_object' should be a list"
    assert len(per_object) >= 1, "'per_object' should contain at least one entry"

    first = per_object[0]
    assert isinstance(first, dict), "Per-object item should be an object"
    assert "file" in first, "Per-object item should include 'file'"
    assert isinstance(first["file"], str) and first["file"], "'file' should be a non-empty string"

    for key, value in first.items():
        _maybe_assert_numeric_for_key(key, value, include_time=True)


def test_cli_json_batch_partial_failure_emits_json(tmp_path: Path) -> None:
    fixture = _choose_fixture()
    missing = tmp_path / "missing_input.stl"

    args = [
        str(fixture),
        str(missing),
        "--json",
        "--material",
        "Enduse PETG",
        "--infill",
        "10",
        "--qty",
        "1",
        "--setup-min",
        "10",
        "--post-min",
        "0",
    ]

    returncode, payload, _ = run_cli_json(args, cwd=repo_root())
    assert returncode != 0, "Expected non-zero return code for partial failure"

    assert "per_object" in payload, "JSON payload missing 'per_object'"
    assert "summary" in payload, "JSON payload missing 'summary'"

    per_object = payload["per_object"]
    if per_object is not None:
        assert isinstance(per_object, list), "'per_object' should be a list"

    summary = payload["summary"]
    if summary is not None:
        assert isinstance(summary, dict), "'summary' should be an object"

    if "count_ok" in payload:
        assert payload["count_ok"] >= 1, "'count_ok' should be >= 1"
    if "count_failed" in payload:
        assert payload["count_failed"] >= 1, "'count_failed' should be >= 1"

    if "errors" in payload:
        errors = payload["errors"]
        assert isinstance(errors, list), "'errors' should be a list"
        assert errors, "'errors' should not be empty on partial failure"
        joined = " ".join(str(item) for item in errors).lower()
        assert any(token in joined for token in ["not found", "missing", "no such file"]), (
            "Expected missing-file hint in errors"
        )
