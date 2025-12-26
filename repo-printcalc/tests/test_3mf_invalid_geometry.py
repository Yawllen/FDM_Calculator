from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import core_calc


def _write_3mf(tmp_path: Path, model_xml: str, name: str) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", model_xml)
    return path


def test_3mf_rejects_non_finite_vertex(tmp_path: Path) -> None:
    model_xml = (
        "<model unit=\"millimeter\" "
        "xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">"
        "<resources>"
        "<object id=\"1\">"
        "<mesh>"
        "<vertices>"
        "<vertex x=\"0\" y=\"0\" z=\"0\"/>"
        "<vertex x=\"nan\" y=\"1\" z=\"0\"/>"
        "</vertices>"
        "<triangles><triangle v1=\"0\" v2=\"0\" v3=\"0\"/></triangles>"
        "</mesh>"
        "</object>"
        "</resources>"
        "<build/>"
        "</model>"
    )
    path = _write_3mf(tmp_path, model_xml, "invalid_vertex.3mf")

    with pytest.raises(ValueError, match=r"non-finite vertex in object 1 at index 1"):
        core_calc.parse_geometry(str(path))


def test_3mf_rejects_out_of_range_triangle_index(tmp_path: Path) -> None:
    model_xml = (
        "<model unit=\"millimeter\" "
        "xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">"
        "<resources>"
        "<object id=\"7\">"
        "<mesh>"
        "<vertices>"
        "<vertex x=\"0\" y=\"0\" z=\"0\"/>"
        "<vertex x=\"1\" y=\"0\" z=\"0\"/>"
        "<vertex x=\"0\" y=\"1\" z=\"0\"/>"
        "</vertices>"
        "<triangles><triangle v1=\"0\" v2=\"1\" v3=\"5\"/></triangles>"
        "</mesh>"
        "</object>"
        "</resources>"
        "<build/>"
        "</model>"
    )
    path = _write_3mf(tmp_path, model_xml, "invalid_triangle.3mf")

    with pytest.raises(ValueError, match=r"Invalid triangle in object 7 at index 0"):
        core_calc.parse_geometry(str(path))
