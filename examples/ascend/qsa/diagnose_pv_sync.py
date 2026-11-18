"""Single-shape board A/B: change only publication of the second PV result."""

import argparse
import hashlib


def add_pv_ready(source, target):
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != "b562f36c2fb4fde2c3f6a509b6551d711111fc4833f913f3a7a0355a51a4fdac":
        raise ValueError(f"Unrecognized AscendC ({digest}); refusing to assume flag IDs or storage offsets")
    result = []
    for line in source.splitlines(keepends=True):
        if line.strip() in ("main_kernel_simd_vf_2(buf_dyn_shmem);", "main_kernel_simd_vf_2_1(buf_dyn_shmem);"):
            result.append("            asc_sync_intra_wait(PIPE_V, 15);\n")
        result.append(line)
        if "asc_copy_l0c2ub(" in line and "buf_dyn_shmem)[8192]" in line:
            result.append("          asc_sync_intra_arrive(PIPE_FIX, 15);\n          asc_sync_intra_arrive(PIPE_FIX, 31);\n")
    return "".join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pv-ready", action="store_true")
    args = parser.parse_args()
    import tilelang
    from tilelang.ascend.callback import register_ascend_postproc
    from examples.ascend.qsa.test_qsa import test_two_tile_handoff

    tilelang.disable_cache()
    if args.pv_ready:
        register_ascend_postproc(add_pv_ready)
    test_two_tile_handoff()
    print(f"Two-tile precision check passed (pv_ready={args.pv_ready})")


if __name__ == "__main__":
    main()
