import numpy as np

from core_calc import compute_volume_cm3


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
