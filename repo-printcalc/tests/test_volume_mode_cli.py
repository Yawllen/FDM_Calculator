from __future__ import annotations

import struct
import zipfile
from pathlib import Path
from typing import Any

from tests.helpers_cli import repo_root, run_cli_json


def _write_binary_stl(path: Path, triangles: list[tuple[tuple[float, float, float], ...]]) -> None:
    header = b"volume-mode-test".ljust(80, b"\0")
    data = bytearray()
    data.extend(header)
    data.extend(struct.pack("<I", len(triangles)))
    for tri in triangles:
        data.extend(struct.pack("<fff", 0.0, 0.0, 1.0))
        for v in tri:
            data.extend(struct.pack("<fff", *v))
        data.extend(struct.pack("<H", 0))
    path.write_bytes(data)


def _cube_triangles(size: float = 1.0) -> list[tuple[tuple[float, float, float], ...]]:
    s = float(size)
    v000 = (0.0, 0.0, 0.0)
    v100 = (s, 0.0, 0.0)
    v010 = (0.0, s, 0.0)
    v110 = (s, s, 0.0)
    v001 = (0.0, 0.0, s)
    v101 = (s, 0.0, s)
    v011 = (0.0, s, s)
    v111 = (s, s, s)

    return [
        (v000, v010, v110), (v000, v110, v100),
        (v001, v101, v111), (v001, v111, v011),
        (v000, v100, v101), (v000, v101, v001),
        (v010, v011, v111), (v010, v111, v110),
        (v000, v001, v011), (v000, v011, v010),
        (v100, v110, v111), (v100, v111, v101),
    ]


def _extract_volume(payload: dict[str, Any]) -> float:
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return float(summary.get("volume_model_cm3", 0.0))
    per_object = payload.get("per_object") or []
    if per_object:
        return float(per_object[0].get("volume_model_cm3", 0.0))
    raise AssertionError("No volume data found in JSON payload")


def _write_simple_3mf(path: Path) -> None:
    model = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<model unit=\"millimeter\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n"
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
        "  </resources>\n"
        "  <build>\n"
        "    <item objectid=\"1\" />\n"
        "  </build>\n"
        "</model>\n"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", model)


def test_cli_volume_modes_for_stl(tmp_path: Path) -> None:
    stl_path = tmp_path / "cube.stl"
    _write_binary_stl(stl_path, _cube_triangles())
    cwd = repo_root()

    fast_code, fast_payload, _ = run_cli_json(
        [str(stl_path), "--json", "--volume-mode", "fast"],
        cwd=cwd,
    )
    stream_code, stream_payload, _ = run_cli_json(
        [str(stl_path), "--json", "--volume-mode", "stream"],
        cwd=cwd,
    )
    bbox_code, bbox_payload, _ = run_cli_json(
        [str(stl_path), "--json", "--volume-mode", "bbox"],
        cwd=cwd,
    )

    assert fast_code == 0
    assert stream_code == 0
    assert bbox_code == 0

    fast_vol = _extract_volume(fast_payload)
    stream_vol = _extract_volume(stream_payload)
    bbox_vol = _extract_volume(bbox_payload)

    assert abs(fast_vol - stream_vol) <= 1e-6
    assert bbox_vol >= fast_vol


def test_cli_stream_rejects_3mf(tmp_path: Path) -> None:
    model_path = tmp_path / "simple.3mf"
    _write_simple_3mf(model_path)
    cwd = repo_root()

    returncode, payload, _ = run_cli_json(
        [str(model_path), "--json", "--volume-mode", "stream"],
        cwd=cwd,
    )

    assert returncode != 0
    errors = payload.get("errors") or []
    joined = " ".join(str(err.get("error", "")) for err in errors)
    assert "binary STL" in joined or "volume-mode=stream" in joined
