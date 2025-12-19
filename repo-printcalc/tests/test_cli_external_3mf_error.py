from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tests.helpers_cli import fixture_path, repo_root, run_cli_json


def _choose_fixture() -> Path:
    for name in ["sub.stl", "Untitled_v1.stl", "ksr_fdmtest_v4.3mf"]:
        candidate = fixture_path(name)
        if candidate.exists():
            return candidate
    pytest.skip("No known fixtures available in tests/fixtures")


def _write_external_missing(tmp_path: Path) -> Path:
    model_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<model unit=\"millimeter\" "
        "xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\" "
        "xmlns:p=\"http://schemas.microsoft.com/3dmanufacturing/production/2015/06\">\n"
        "  <resources>\n"
        "    <object id=\"1\" type=\"model\">\n"
        "      <mesh>\n"
        "        <vertices>\n"
        "          <vertex x=\"0\" y=\"0\" z=\"0\" />\n"
        "          <vertex x=\"1\" y=\"0\" z=\"0\" />\n"
        "          <vertex x=\"0\" y=\"1\" z=\"0\" />\n"
        "          <vertex x=\"0\" y=\"0\" z=\"1\" />\n"
        "        </vertices>\n"
        "        <triangles>\n"
        "          <triangle v1=\"0\" v2=\"1\" v3=\"2\" />\n"
        "          <triangle v1=\"0\" v2=\"1\" v3=\"3\" />\n"
        "          <triangle v1=\"0\" v2=\"2\" v3=\"3\" />\n"
        "          <triangle v1=\"1\" v2=\"2\" v3=\"3\" />\n"
        "        </triangles>\n"
        "      </mesh>\n"
        "    </object>\n"
        "    <object id=\"2\" type=\"model\">\n"
        "      <components>\n"
        "        <component objectid=\"1\" p:path=\"/3D/other.model\" />\n"
        "      </components>\n"
        "    </object>\n"
        "  </resources>\n"
        "  <build>\n"
        "    <item objectid=\"2\" />\n"
        "  </build>\n"
        "</model>\n"
    )
    path = tmp_path / "external_missing.3mf"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", model_xml)
    return path


def test_cli_batch_external_missing_model_reports_error(tmp_path: Path) -> None:
    fixture = _choose_fixture()
    external_missing = _write_external_missing(tmp_path)

    args = [
        str(fixture),
        str(external_missing),
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
    assert returncode != 0, "Expected non-zero return code for missing external model"

    errors = payload.get("errors") or []
    assert isinstance(errors, list), "Expected 'errors' to be a list"
    assert errors, "Expected errors to be reported for missing external model"

    joined = " ".join(str(item) for item in errors).lower()
    assert any(token in joined for token in ["external", "p:path", "missing model"]), (
        "Expected missing external model hint in errors"
    )

    if "count_failed" in payload:
        assert payload["count_failed"] >= 1
