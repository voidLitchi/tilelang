"""Board A/B for the FIX-to-vector handoff of non-initial PV tiles."""

import argparse
import hashlib
from pathlib import Path


def add_pv_ready(source, target):
    consumers = {"main_kernel_simd_vf_2(buf_dyn_shmem);", "main_kernel_simd_vf_2_1(buf_dyn_shmem);"}
    lines = source.splitlines(keepends=True)
    copies = [i for i, line in enumerate(lines) if "asc_copy_l0c2ub(" in line and "buf_dyn_shmem)[8192]" in line]
    readers = [i for i, line in enumerate(lines) if line.strip() in consumers]
    if len(copies) != 1 or {lines[i].strip() for i in readers} != consumers or len(readers) != 2:
        raise ValueError("Unrecognized QSA PV copy or AIV accumulator calls")
    copy = copies[0]
    ready = ["asc_sync_intra_arrive(PIPE_FIX, 12);", "asc_sync_intra_arrive(PIPE_FIX, 28);"]
    if [line.strip() for line in lines[copy + 1 : copy + 3]] == ready and all(
        lines[i - 1].strip() == "asc_sync_intra_wait(PIPE_V, 12);" for i in readers
    ):
        print("pv-ready: found native FIX 12/28 and both AIV waits; using original AscendC", flush=True)
        return source

    # Only rewrite the reviewed two-tile schedules. Dynamic addressing changes
    # the source digest even when the flag allocation and handoff are identical.
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest not in {
        "b562f36c2fb4fde2c3f6a509b6551d711111fc4833f913f3a7a0355a51a4fdac",
        "110f54facbfc6ef3a21a45e09c609c6d60b47b381f4e0fe8e95e01c6790e5033",
    }:
        raise ValueError(f"Unrecognized AscendC ({digest}); refusing to assume flag IDs or storage offsets")
    lines[copy] += "          asc_sync_intra_arrive(PIPE_FIX, 15);\n          asc_sync_intra_arrive(PIPE_FIX, 31);\n"
    for i in readers:
        lines[i] = "            asc_sync_intra_wait(PIPE_V, 15);\n" + lines[i]
    print("pv-ready: added FIX 15/31 and both AIV waits to two-tile AscendC", flush=True)
    return "".join(lines)


def enable_pv_ready():
    import tilelang
    from tilelang.ascend.callback import register_ascend_postproc

    tilelang.disable_cache()
    register_ascend_postproc(add_pv_ready)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pv-ready", action="store_true")
    parser.add_argument("--all", action="store_true", help="run the QSA pytest file and all static/dynamic run.py scenarios")
    args = parser.parse_args()
    import tilelang
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("QSA precision checks require an available Ascend NPU")
    if args.pv_ready:
        enable_pv_ready()
    else:
        tilelang.disable_cache()
    if args.all:
        import pytest
        from examples.ascend.qsa import run

        status = pytest.main(["-v", "-s", str(Path(__file__).with_name("test_qsa.py"))])
        if status != pytest.ExitCode.OK:
            raise SystemExit(status)
        for shape_args in ([], ["--dynamic"]):
            run.main(["--check-only", *shape_args])
        print(f"All QSA precision checks passed (pv_ready={args.pv_ready})")
        return
    from examples.ascend.qsa.test_qsa import test_two_tile_handoff

    test_two_tile_handoff()
    print(f"Two-tile precision check passed (pv_ready={args.pv_ready})")


if __name__ == "__main__":
    main()
