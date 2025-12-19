import os
import struct
from pathlib import Path

import numpy as np

from core_calc import _apply_transform, parse_stl, stl_stream_volume_cm3


def _write_binary_stl(path: Path, triangles: list[tuple[tuple[float, float, float], ...]], header: bytes | None = None):
    hdr = (header or b"")[:80].ljust(80, b"\0")
    data = bytearray()
    data.extend(hdr)
    data.extend(struct.pack("<I", len(triangles)))
    for tri in triangles:
        # normal (ignored)
        data.extend(struct.pack("<fff", 0.0, 0.0, 1.0))
        for v in tri:
            data.extend(struct.pack("<fff", *v))
        data.extend(struct.pack("<H", 0))
    path.write_bytes(data)


def _write_ascii_stl(path: Path):
    path.write_text(
        "\n".join(
            [
                "solid ascii",
                "facet normal 0 0 1",
                "  outer loop",
                "    vertex 0 0 0",
                "    vertex 1 0 0",
                "    vertex 0 1 0",
                "  endloop",
                "endfacet",
                "endsolid ascii",
            ]
        ),
        encoding="utf-8",
    )


def test_apply_transform_empty_array_returns_empty():
    V = np.empty((0, 3), dtype=np.float64)
    M = np.eye(4)
    out = _apply_transform(V, M)
    assert out is V
    assert out.shape == (0, 3)
    assert out.dtype == V.dtype


def test_detect_stl_binary_by_size_matches_expected(tmp_path: Path):
    tri = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    binary_path = tmp_path / "valid.stl"
    _write_binary_stl(binary_path, [tri])

    expected_size = 84 + 50 * 1
    assert os.path.getsize(binary_path) == expected_size

    models = parse_stl(str(binary_path))
    assert len(models) == 1
    _, V, T, _, meta = models[0]
    assert meta["type"] == "stl"
    assert T.shape[0] == 1
    assert V.shape[0] == 3


def test_binary_header_starts_with_solid_is_accepted(tmp_path: Path):
    tri = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    binary_path = tmp_path / "solid_header.stl"
    _write_binary_stl(binary_path, [tri], header=b"solid binary header")

    models = parse_stl(str(binary_path))
    assert len(models) == 1


def test_parse_stl_ascii_raises_user_friendly_error(tmp_path: Path):
    ascii_path = tmp_path / "ascii.stl"
    _write_ascii_stl(ascii_path)

    try:
        parse_stl(str(ascii_path))
    except ValueError as e:
        assert "ASCII STL detected" in str(e)
    else:
        raise AssertionError("ValueError expected for ASCII STL")


def test_stl_stream_rejects_ascii(tmp_path: Path):
    ascii_path = tmp_path / "ascii_stream.stl"
    _write_ascii_stl(ascii_path)

    try:
        stl_stream_volume_cm3(str(ascii_path))
    except ValueError as e:
        assert "Binary STL" in str(e)
    else:
        raise AssertionError("ValueError expected for ASCII STL")


def test_parse_stl_rejects_truncated_binary(tmp_path: Path):
    tri = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    trunc_path = tmp_path / "truncated.stl"
    # Advertise 2 triangles, but provide data only for 1
    hdr = b"truncated"
    hdr = hdr[:80].ljust(80, b"\0")
    data = bytearray()
    data.extend(hdr)
    data.extend(struct.pack("<I", 2))
    # Only one triangle worth of bytes follows.
    data.extend(struct.pack("<fff", 0.0, 0.0, 1.0))
    for v in tri:
        data.extend(struct.pack("<fff", *v))
    data.extend(struct.pack("<H", 0))
    trunc_path.write_bytes(data)

    try:
        parse_stl(str(trunc_path))
    except ValueError as e:
        assert "Malformed binary STL" in str(e)
    else:
        raise AssertionError("ValueError expected for malformed binary STL")
