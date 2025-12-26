import zipfile

import pytest

import core_calc


def test_3mf_rejects_build_item_missing_object_reference(tmp_path):
    model_xml = (
        "<model unit=\"millimeter\" "
        "xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">"
        "<resources/>"
        "<build><item objectid=\"999\"/></build>"
        "</model>"
    )
    path = tmp_path / "missing_object_ref.3mf"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", model_xml)

    with pytest.raises(ValueError, match=r"missing object.*999"):
        core_calc.parse_geometry(str(path))
