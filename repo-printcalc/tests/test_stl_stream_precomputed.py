import struct
from pathlib import Path

import numpy as np

import core_calc


def _write_binary_stl(path: Path, triangles: list[tuple[tuple[float, float, float], ...]]):
    header = b"".ljust(80, b"\0")
    data = bytearray()
    data.extend(header)
    data.extend(struct.pack("<I", len(triangles)))
    for tri in triangles:
        data.extend(struct.pack("<fff", 0.0, 0.0, 1.0))
        for v in tri:
            data.extend(struct.pack("<fff", *v))
        data.extend(struct.pack("<H", 0))
    path.write_bytes(data)


def test_compute_volume_cm3_stream_prefers_precomputed_stream(monkeypatch):
    def _boom(path: str) -> float:
        raise AssertionError("stl_stream_volume_cm3 should not be called")

    monkeypatch.setattr("core_calc.stl_stream_volume_cm3", _boom)

    volume = core_calc.compute_volume_cm3(
        V_mm=np.empty((0, 3), dtype=np.float64),
        T=np.empty((0, 3), dtype=np.int32),
        mode="stream",
        meta={"precomputed_stream_volume_cm3": 2.5, "type": "stl", "path": "x.stl"},
    )

    assert volume == 2.5


def test_parse_stl_adds_precomputed_stream_volume(tmp_path: Path):
    tri = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    tri2 = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    tri3 = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    tri4 = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    path = tmp_path / "tetra.stl"
    _write_binary_stl(path, [tri, tri2, tri3, tri4])

    models = core_calc.parse_stl(str(path))
    assert len(models) == 1
    _, _, _, _, meta = models[0]
    assert "precomputed_stream_volume_cm3" in meta


def test_stream_volume_matches_precomputed_from_parse(tmp_path: Path):
    tri = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    tri2 = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    tri3 = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    tri4 = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    path = tmp_path / "tetra_match.stl"
    _write_binary_stl(path, [tri, tri2, tri3, tri4])

    models = core_calc.parse_stl(str(path))
    _, _, _, _, meta = models[0]
    stream_volume = core_calc.compute_volume_cm3(
        V_mm=np.empty((0, 3), dtype=np.float64),
        T=np.empty((0, 3), dtype=np.int32),
        mode="stream",
        meta=meta,
    )
    assert stream_volume == core_calc.stl_stream_volume_cm3(str(path))
