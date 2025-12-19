from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import core_calc


def _write_3mf(tmp_path: Path, model_xml: str) -> Path:
    path = tmp_path / "sample_external_missing.3mf"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", model_xml)
    return path


def test_3mf_rejects_missing_external_component_model(tmp_path: Path) -> None:
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
    path = _write_3mf(tmp_path, model_xml)

    with pytest.raises(ValueError, match=r"external|p:path|missing model"):
        core_calc.parse_geometry(str(path))
