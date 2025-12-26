import struct
from pathlib import Path

import pytest

from core_calc import MAX_STL_TRIANGLES, parse_stl


def test_stl_triangle_limit_exceeded(tmp_path: Path):
    invalid_path = tmp_path / "too_many_triangles.stl"
    header = b"limit-test".ljust(80, b"\0")
    data = bytearray()
    data.extend(header)
    data.extend(struct.pack("<I", MAX_STL_TRIANGLES + 1))
    invalid_path.write_bytes(data)

    with pytest.raises(ValueError, match=rf"STL limit exceeded: triangles={MAX_STL_TRIANGLES + 1} > {MAX_STL_TRIANGLES}"):
        parse_stl(str(invalid_path))
