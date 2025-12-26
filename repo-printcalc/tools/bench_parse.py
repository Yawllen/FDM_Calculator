import argparse
import os
import time

import core_calc as core


def _bench_file(path: str) -> None:
    core._reset_parser_state()
    started = time.perf_counter()
    data = core.parse_geometry(path)
    total_triangles = 0
    total_volume_cm3 = 0.0
    for _, V, T, _, srcinfo in data:
        total_triangles += int(T.shape[0])
        total_volume_cm3 += float(core.compute_volume_cm3(V, T, mode="fast", meta=srcinfo))
    elapsed_s = time.perf_counter() - started
    file_type = os.path.splitext(path)[1].lower().lstrip(".")
    print(
        f"{file_type} file={path} triangles={total_triangles} "
        f"volume_cm3={total_volume_cm3:.6f} elapsed_s={elapsed_s:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark parse + volume for STL and 3MF files.")
    parser.add_argument("--stl", required=True, help="Path to a binary STL file.")
    parser.add_argument("--mf", required=True, help="Path to a 3MF file.")
    args = parser.parse_args()

    _bench_file(args.stl)
    _bench_file(args.mf)


if __name__ == "__main__":
    main()
