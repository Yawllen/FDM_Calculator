from pathlib import Path
import struct

import numpy as np
import pytest

import core_calc


def _write_binary_stl(path: Path) -> None:
    header = b"Binary STL Test" + b"\0" * (80 - len("Binary STL Test"))
    triangle = struct.pack(
        "<12fH",
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0,
    )
    with path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", 1))
        f.write(triangle)


def test_parse_geometry_cache_hit_returns_same_results():
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "sub.stl"
    data_first = core_calc.parse_geometry(str(fixture_path))
    data_second = core_calc.parse_geometry(str(fixture_path))

    assert len(data_first) == len(data_second) == 1

    name1, V1, T1, vol1, meta1 = data_first[0]
    name2, V2, T2, vol2, meta2 = data_second[0]

    assert name1 == name2
    assert meta1["type"] == meta2["type"] == "stl"
    assert np.isclose(vol1, vol2)
    assert np.array_equal(V1, V2)
    assert np.array_equal(T1, T2)
    assert V1 is not V2
    assert T1 is not T2


def test_cache_invalidated_on_file_change(tmp_path):
    stl_path = tmp_path / "sample.stl"
    _write_binary_stl(stl_path)
    core_calc.parse_geometry(str(stl_path))

    with stl_path.open("ab") as f:
        f.write(b"\0")

    with pytest.raises(ValueError, match=r"Malformed binary STL"):
        core_calc.parse_geometry(str(stl_path))
