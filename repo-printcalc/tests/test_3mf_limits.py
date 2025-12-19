import zipfile

import pytest

import core_calc


def _write_3mf(tmp_path, model_xml: str) -> str:
    path = tmp_path / "sample.3mf"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", model_xml)
    return str(path)


def test_3mf_rejects_large_model_entry(tmp_path):
    path = tmp_path / "large_entry.3mf"
    big_payload = "a" * (core_calc.MAX_3MF_ENTRY_BYTES + 1)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", big_payload)

    with pytest.raises(ValueError, match=r"too large|exceeds limit"):
        core_calc.parse_geometry(str(path))


def test_3mf_rejects_too_many_objects_or_components(tmp_path):
    object_count = core_calc.MAX_3MF_OBJECTS + 1
    objects = "".join(
        f"<object id=\"{idx}\"/>" for idx in range(1, object_count + 1)
    )
    model_xml = (
        "<model unit=\"millimeter\" "
        "xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">"
        f"<resources>{objects}</resources>"
        "<build/>"
        "</model>"
    )
    path = _write_3mf(tmp_path, model_xml)

    with pytest.raises(ValueError, match=r"MAX_3MF_OBJECTS|objects"):
        core_calc.parse_geometry(path)
