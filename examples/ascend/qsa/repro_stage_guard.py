"""Standalone Ascend stage-guard reproducer. Can be copied into a clean upstream checkout."""

import argparse
from pathlib import Path

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm


def copy_rows(width, x_stage):
    @T.prim_func
    def main(A: T.Tensor((2, width), "float32"), C: T.Tensor((2, width), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((width,), "float32")
            y = T.alloc_shared((width,), "float32")
            T.annotate_buffer_versions({x: 1, y: 1})
            # Give y a known old value so a stale read is observable.
            T.copy(A[0, :], y)
            T.copy(y, C[1, :])
            for i in T.Pipelined(2, num_stages=2, annotations={"enable_offset": True}):
                with T.Stage(x_stage):
                    if i == 1:
                        T.copy(A[0, :], x)
                        T.copy(x, C[0, :])
                with T.Stage(0):
                    if i != 0:
                        T.copy(A[1, :], y)
                        T.copy(y, C[1, :])

    return main


def dump_case(func, directory, stage):
    guards = {}

    @tvm.instrument.pass_instrument
    class Inspect:
        def run_after_pass(self, mod, info):
            if info.name != "tl.PrepareMultiBuffer":
                return
            (directory / f"stage{stage}.tir").write_text(mod.script())

            def visit(stmt):
                if isinstance(stmt, tvm.tirx.For):
                    for storage, predicate in stmt.annotations.get("tl.storage_epoch_guard_map", {}).items():
                        guards[storage.name] = predicate

            tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)

    with tvm.target.Target("ascend"), tvm.transform.PassContext(opt_level=3, instruments=[Inspect()]):
        src = tilelang.lower(func, target="ascend").kernel_source
    (directory / f"stage{stage}.asc").write_text(src)
    print(f"stage={stage}/0, storage guards: {guards}", flush=True)
    return guards


def run_case(func, width, stage, repeat):
    import torch
    import torch_npu  # noqa: F401

    kernel = tilelang.compile(func, target="ascend", out_idx=-1)
    a = torch.empty((2, width), device="npu", dtype=torch.float32)
    for trial in range(repeat):
        a[0].fill_(trial % 2)
        a[1].fill_(2 + trial % 2)
        result = kernel(a).cpu()
        expected = a.cpu()
        if not torch.equal(result, expected):
            row, col = (result != expected).nonzero()[0].tolist()
            raise AssertionError(
                f"stage={stage}/0 trial={trial} C[{row},{col}]={result[row, col].item()}, expected {expected[row, col].item()}"
            )
    print(f"stage={stage}/0: {repeat} exact device comparisons passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, help="save PrepareMultiBuffer IR and final AscendC for both schedules")
    parser.add_argument("--check-ir", action="store_true", help="fail if x and y in different stages share a guard (requires --dump-dir)")
    parser.add_argument("--run", action="store_true", help="run the same-stage control and then the shifted variant on the NPU")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    if not (args.dump_dir or args.run) or (args.check_ir and not args.dump_dir):
        parser.error("choose --dump-dir and/or --run; --check-ir requires --dump-dir")
    if args.width <= 0 or args.width % 8 or args.repeat <= 0:
        parser.error("width must be a positive multiple of 8; repeat must be positive")
    tilelang.disable_cache()
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
    for stage in (0, 1):
        func = copy_rows(args.width, stage)
        if args.dump_dir:
            guards = dump_case(func, args.dump_dir, stage)
        if args.run:
            run_case(func, args.width, stage, args.repeat)
        if args.check_ir and stage == 1:
            if not {"x", "y"} <= guards.keys():
                raise RuntimeError("Cannot identify the two UB epoch guards; inspect the dumped IR")
            assert not guards["x"].same_as(guards["y"]), "different-stage UB epochs share the same guard"


if __name__ == "__main__":
    main()
