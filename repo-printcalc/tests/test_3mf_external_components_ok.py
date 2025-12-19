from __future__ import annotations

import math
import zipfile
from pathlib import Path

from core_calc import parse_3mf


def _mesh_xml(object_id: str) -> str:
    return (
        f"    <object id=\"{object_id}\" type=\"model\">\n"
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
    )


def test_external_component_path_with_leading_slash_is_accepted(tmp_path: Path) -> None:
    main_model = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<model unit=\"millimeter\" "
        "xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\" "
        "xmlns:p=\"http://schemas.microsoft.com/3dmanufacturing/production/2015/06\">\n"
        "  <resources>\n"
        f"{_mesh_xml('1')}"
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
    other_model = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<model unit=\"millimeter\" "
        "xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n"
        "  <resources>\n"
        f"{_mesh_xml('1')}"
        "  </resources>\n"
        "</model>\n"
    )
    path = tmp_path / "sample_external_ok.3mf"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", main_model)
        zf.writestr("3D/other.model", other_model)

    data = parse_3mf(str(path))
    assert data, "Expected parsed geometry for valid external component"
    assert math.isfinite(data[0][3])
    assert data[0][3] >= 0
