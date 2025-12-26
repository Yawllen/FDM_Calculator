import numpy as np

from core_calc import compute_volume_cm3, is_stream_supported


def test_compute_volume_cm3_fast_and_bbox():
    V_mm = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    T = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )

    fast_vol = compute_volume_cm3(V_mm, T, mode="fast", meta={})
    bbox_vol = compute_volume_cm3(V_mm, T, mode="bbox", meta={})

    assert fast_vol > 0
    assert bbox_vol > 0
    assert bbox_vol >= fast_vol


def test_compute_volume_cm3_fast_prefers_precomputed():
    V_mm = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )

    precomputed = 123.456
    volume = compute_volume_cm3(
        V_mm,
        T,
        mode="fast",
        meta={"precomputed_volume_cm3": precomputed},
    )

    assert volume == precomputed


def test_compute_volume_cm3_stream_prefers_precomputed(monkeypatch):
    def _boom(path: str) -> float:
        raise AssertionError("stl_stream_volume_cm3 should not be called")

    monkeypatch.setattr("core_calc.stl_stream_volume_cm3", _boom)

    volume = compute_volume_cm3(
        V_mm=np.empty((0, 3), dtype=np.float64),
        T=np.empty((0, 3), dtype=np.int32),
        mode="stream",
        meta={"precomputed_volume_cm3": 2.5},
    )

    assert volume == 2.5


def test_is_stream_supported_requires_stl():
    assert is_stream_supported("model.stl") is True
    assert is_stream_supported("MODEL.STL") is True
    assert is_stream_supported("model.3mf") is False
    assert is_stream_supported("model.obj") is False
