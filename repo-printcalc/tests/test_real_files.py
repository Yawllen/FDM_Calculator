from __future__ import annotations

from pathlib import Path

import numpy as np

from core_calc import parse_geometry, stl_stream_volume_cm3, volume_tetra

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
STL_FILES = [
    FIXTURES_DIR / "sub.stl",
    FIXTURES_DIR / "Untitled_v1.stl",
]
THREEMF_FILE = FIXTURES_DIR / "ksr_fdmtest_v4.3mf"


def _assert_mesh_arrays(V: np.ndarray, T: np.ndarray) -> None:
    assert isinstance(V, np.ndarray)
    assert isinstance(T, np.ndarray)
    assert V.ndim == 2
    assert T.ndim == 2
    assert V.shape[1] == 3
    assert T.shape[1] == 3
    assert V.size > 0
    assert T.size > 0


def _assert_volume_close(vol_a: float, vol_b: float) -> None:
    assert np.isfinite(vol_a)
    assert np.isfinite(vol_b)
    assert np.isclose(vol_a, vol_b, rtol=1e-5, atol=1e-6)


def test_real_stl_files_parse_and_volume_match():
    for path in STL_FILES:
        models = parse_geometry(str(path))
        assert len(models) == 1
        _, V, T, vol_fast_cm3, _ = models[0]
        _assert_mesh_arrays(V, T)
        assert np.isfinite(vol_fast_cm3)
        assert vol_fast_cm3 >= 0

        vol_tetra = volume_tetra(V, T)
        _assert_volume_close(vol_tetra, vol_fast_cm3)

        vol_stream = stl_stream_volume_cm3(str(path))
        _assert_volume_close(vol_stream, vol_fast_cm3)


def test_real_3mf_file_parse_and_volume_match():
    models = parse_geometry(str(THREEMF_FILE))
    assert len(models) >= 1

    total_volume = 0.0
    for _, V, T, vol_fast_cm3, meta in models:
        assert meta.get("type") == "3mf"
        assert np.isfinite(vol_fast_cm3)
        assert vol_fast_cm3 >= 0
        total_volume += vol_fast_cm3

        if V.size != 0 and T.size != 0:
            _assert_mesh_arrays(V, T)
            vol_tetra = volume_tetra(V, T)
            _assert_volume_close(vol_tetra, vol_fast_cm3)

    assert total_volume > 0
