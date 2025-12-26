import zipfile
from pathlib import Path

import numpy as np

from core_calc import parse_3mf, volume_tetra


BASE_VERTICES = [
    (1.0, 2.0, 3.0),
    (2.0, 2.0, 3.0),
    (1.0, 3.0, 3.0),
    (1.0, 2.0, 4.0),
]

TRIANGLES = np.array(
    [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ],
    dtype=np.int32,
)


def _write_model(tmp_path: Path, body: str) -> Path:
    model_path = tmp_path / "sample.3mf"
    with zipfile.ZipFile(model_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", body)
    return model_path


def _mesh_xml(object_id: str) -> str:
    verts = "\n".join(
        f'        <vertex x="{x}" y="{y}" z="{z}" />' for x, y, z in BASE_VERTICES
    )
    tris = "\n".join(
        f'        <triangle v1="{a}" v2="{b}" v3="{c}" />' for a, b, c in TRIANGLES
    )
    return (
        f"    <object id=\"{object_id}\" type=\"model\">\n"
        "      <mesh>\n"
        f"      <vertices>\n{verts}\n      </vertices>\n"
        f"      <triangles>\n{tris}\n      </triangles>\n"
        "      </mesh>\n"
        "    </object>\n"
    )


def test_transform_translation(tmp_path: Path):
    xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<model unit=\"millimeter\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n"
        "  <resources>\n"
        f"{_mesh_xml('1')}"
        "  </resources>\n"
        "  <build>\n"
        "    <item objectid=\"1\" transform=\"1 0 0 10 0 1 0 20 0 0 1 30\" />\n"
        "  </build>\n"
        "</model>\n"
    )
    path = _write_model(tmp_path, xml)

    name, V_mm, T, _, _ = parse_3mf(str(path))[0]
    assert name.endswith("item_1")
    assert np.allclose(V_mm[0], np.array([11.0, 22.0, 33.0]))
    assert np.array_equal(T, TRIANGLES)


def test_transform_singular_no_alt_order(tmp_path: Path):
    xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<model unit=\"millimeter\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n"
        "  <resources>\n"
        f"{_mesh_xml('1')}"
        "  </resources>\n"
        "  <build>\n"
        "    <item objectid=\"1\" transform=\"1 0 0 10 0 1 0 20 0 0 0 30\" />\n"
        "  </build>\n"
        "</model>\n"
    )
    path = _write_model(tmp_path, xml)

    _, V_mm, _, _, _ = parse_3mf(str(path))[0]
    assert np.allclose(V_mm[0], np.array([11.0, 22.0, 30.0]))


def test_transform_scale_volume(tmp_path: Path):
    xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<model unit=\"millimeter\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n"
        "  <resources>\n"
        f"{_mesh_xml('1')}"
        "  </resources>\n"
        "  <build>\n"
        "    <item objectid=\"1\" transform=\"2 0 0 0 0 2 0 0 0 0 2 0\" />\n"
        "  </build>\n"
        "</model>\n"
    )
    path = _write_model(tmp_path, xml)

    _, V_mm, T, _, _ = parse_3mf(str(path))[0]
    base_vol = volume_tetra(np.array(BASE_VERTICES), TRIANGLES)
    scaled_vol = volume_tetra(V_mm, T)
    assert np.isclose(scaled_vol, base_vol * 8)


def test_nested_component_order(tmp_path: Path):
    xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<model unit=\"millimeter\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n"
        "  <resources>\n"
        f"{_mesh_xml('1')}"
        "    <object id=\"2\" type=\"model\">\n"
        "      <components>\n"
        "        <component objectid=\"1\" transform=\"2 0 0 0 0 2 0 0 0 0 2 0\" />\n"
        "      </components>\n"
        "    </object>\n"
        "  </resources>\n"
        "  <build>\n"
        "    <item objectid=\"2\" transform=\"1 0 0 5 0 1 0 0 0 0 1 0\" />\n"
        "  </build>\n"
        "</model>\n"
    )
    path = _write_model(tmp_path, xml)

    _, V_mm, T, _, _ = parse_3mf(str(path))[0]
    expected_first = np.array([7.0, 4.0, 6.0])
    assert np.allclose(V_mm[0], expected_first)
    expected_vertices = np.array(
        [
            [7.0, 4.0, 6.0],
            [9.0, 4.0, 6.0],
            [7.0, 6.0, 6.0],
            [7.0, 4.0, 8.0],
        ]
    )
    assert np.allclose(np.sort(V_mm, axis=0), np.sort(expected_vertices, axis=0))
