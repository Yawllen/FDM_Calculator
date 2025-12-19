import zipfile

import pytest

import core_calc


def test_3mf_rejects_object_without_id(tmp_path):
    model_xml = (
        "<model unit=\"millimeter\" "
        "xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">"
        "<resources>"
        "<object>"
        "<mesh>"
        "<vertices><vertex x=\"0\" y=\"0\" z=\"0\"/></vertices>"
        "<triangles><triangle v1=\"0\" v2=\"0\" v3=\"0\"/></triangles>"
        "</mesh>"
        "</object>"
        "</resources>"
        "<build/>"
        "</model>"
    )
    path = tmp_path / "missing_id.3mf"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", model_xml)

    with pytest.raises(ValueError, match=r"missing required id"):
        core_calc.parse_geometry(str(path))
