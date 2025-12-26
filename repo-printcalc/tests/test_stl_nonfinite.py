import struct
from pathlib import Path

import pytest

from core_calc import parse_stl, stl_stream_volume_cm3


def _write_binary_stl(path: Path, triangles: list[tuple[tuple[float, float, float], ...]]):
    data = bytearray()
    data.extend(b"nonfinite".ljust(80, b"\0"))
    data.extend(struct.pack("<I", len(triangles)))
    for tri in triangles:
        data.extend(struct.pack("<fff", 0.0, 0.0, 1.0))
        for v in tri:
            data.extend(struct.pack("<fff", *v))
        data.extend(struct.pack("<H", 0))
    path.write_bytes(data)


def test_parse_stl_rejects_nonfinite_vertex(tmp_path: Path):
    nan = float("nan")
    tri = ((nan, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    stl_path = tmp_path / "nan_parse.stl"
    _write_binary_stl(stl_path, [tri])

    with pytest.raises(ValueError, match=r"Malformed binary STL: non-finite vertex coordinate"):
        parse_stl(str(stl_path))


def test_stl_stream_volume_rejects_nonfinite_vertex(tmp_path: Path):
    nan = float("nan")
    tri = ((0.0, 0.0, 0.0), (nan, 0.0, 0.0), (0.0, 1.0, 0.0))
    stl_path = tmp_path / "nan_stream.stl"
    _write_binary_stl(stl_path, [tri])

    with pytest.raises(ValueError, match=r"Malformed binary STL: non-finite vertex coordinate"):
        stl_stream_volume_cm3(str(stl_path))
