"""Phase 0 F3: the P1 pass attaches a region to every top-level T.Parallel loop."""

import tilelang
import tilelang.ascend.language as T
import tvm
from tvm import tirx
from tilelang.ascend import transform as at
from tilelang.ascend import vsimd_ir as ir

REGION_KEY = "tl.vsimd.region"
FRAGMENT_KEY = "tl.vsimd.fragment"


def _lift(func):
    mod = tvm.IRModule({"main": func})
    return at.AscendLiftVSimd()(mod)


def _regions(mod):
    """Return the region annotations in body order."""
    regions = []

    def visit(node):
        if isinstance(node, tirx.For):
            if REGION_KEY in dict(node.annotations):
                regions.append(node.annotations[REGION_KEY])

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return regions


def _vf_fragments(mod):
    """Return the per-VF fragment tables in VF order."""
    tables = []

    def visit(node):
        if isinstance(node, tirx.SBlock) and node.name_hint == "SIMD_VF":
            if FRAGMENT_KEY in dict(node.annotations):
                tables.append(node.annotations[FRAGMENT_KEY])

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return tables


def _vf_bodies(mod):
    bodies = []

    def visit(node):
        if isinstance(node, tirx.SBlock) and node.name_hint == "SIMD_VF":
            bodies.append(node.body)

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return bodies


def _strip_for_annotations(stmt):
    """Drop For annotations so a lifted body can be compared structurally."""

    def visit(node):
        if isinstance(node, tirx.For) and dict(node.annotations):
            return tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                {},
            )
        return None

    return tirx.stmt_functor.ir_transform(stmt, None, visit)


def test_region_attached_to_parallel_and_body_unchanged():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    out = _lift(main)
    before = _vf_bodies(tvm.IRModule({"main": main}))[0]
    after = _vf_bodies(out)[0]
    assert tvm.ir.structural_equal(
        _strip_for_annotations(before), _strip_for_annotations(after)
    )

    regions = _regions(out)
    assert len(regions) == 1
    region = regions[0]
    assert region.status == "LIFTED"
    assert region.reason_code == ""
    assert int(region.vf_index) == 0
    assert int(region.region_index) == 0
    assert [str(axis.extent) for axis in region.axes] == ["64"]
    assert [op.kind for op in region.ops] == ["store"]
    assert [capture.buffer.name for capture in region.captures] == [
        "c_ub",
        "a_ub",
    ]


def test_independent_parallels_lift_independently():
    @T.prim_func
    def main(
        A: T.Tensor((96,), "float32"),
        D: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((96,), "float32")
            d_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            T.copy(D, d_ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    a_ub[i] = a_ub[i + 32]  # cross-iteration RAW: illegal
                for i in T.Parallel(64):
                    c_ub[i] = d_ub[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 2
    assert regions[0].status == "OPAQUE"
    assert regions[0].reason_code == "ILLEGAL"
    assert regions[1].status == "LIFTED"
    # No fragment connects them, so neither belongs to a group.
    assert int(regions[0].group) == -1
    assert int(regions[1].group) == -1


def test_shared_buffer_dependency_does_not_group():
    """A non-fragment dependency must not force the consumer down.

    Shared/global buffers always have the scalar fallback, so P1 keeps the
    regions independent and leaves the decision to P2.
    """

    @T.prim_func
    def main(
        A: T.Tensor((96,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((96,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    a_ub[i] = a_ub[i + 32]
                for i in T.Parallel(64):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert [region.status for region in regions] == ["OPAQUE", "LIFTED"]
    assert int(regions[1].group) == -1


def test_serial_wrapped_parallel_is_not_annotated():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            tmp = T.alloc_shared((32,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for w in T.serial(2):
                    for i in T.Parallel(32):
                        tmp[i] = 1.0 + w
                for i in T.Parallel(64):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"
    assert [int(axis.extent) for axis in regions[0].axes] == [64]


def test_no_top_level_parallel_means_no_annotation():
    @T.prim_func
    def main(
        A: T.Tensor((64,), "float32"),
        C: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for w in T.serial(2):
                    for i in T.Parallel(64):
                        a_ub[i] = 1.0 + w
            T.copy(a_ub, C)

    assert _regions(_lift(main)) == []


def test_no_parallel_no_annotation():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n, 64), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n, 64), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                T.reduce_sum(a_ub, c_ub, dim=1, clear=True)
            T.copy(c_ub, C)

    out = _lift(main)
    assert _regions(out) == []
    assert _vf_fragments(out) == []


def test_assume_wrapper_is_transparent():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                T.attr(n > 0, "tl.assume", "test assume")
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"


def test_region_annotation_survives_simplify():
    n = 32

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] * 2.0
            T.copy(c_ub, C)

    lifted = _lift(main)
    simplified = tilelang.transform.Simplify()(lifted)
    regions = _regions(simplified)
    assert len(regions) == 1
    assert regions[0].status == "LIFTED"


def test_opaque_reason_from_unsupported_parallel():
    """A row-slice reduction nested in the row loop is not modelled yet.

    This is the target-FA shape: one outer row ``T.Parallel`` whose body mixes
    nested Parallel loops with row ``T.reduce`` calls. The current nest model
    cannot represent it, so the region must be OPAQUE with the reason.
    """
    br, bc = 64, 128

    @T.prim_func
    def main(A: T.Tensor((br, bc), "float32"), M: T.Tensor((br,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((br, bc), "float32")
            m_ub = T.alloc_shared((br,), "float32")
            T.copy(A, s_ub)
            with T.SimdVF():
                for i in T.Parallel(br):
                    for j in T.Parallel(bc):
                        s_ub[i, j] = s_ub[i, j] * 0.5
                    T.reduce_max(
                        s_ub[i, 0:bc], m_ub[i : i + 1], dim=1, clear=False
                    )
            T.copy(m_ub, M)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    region = regions[0]
    assert region.status == "OPAQUE"
    assert region.reason_code == "UNSUPPORTED"
    assert "opaque Evaluate" in region.reason


def test_fragment_table_is_a_vf_annotation():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((n,), "float32")
            with T.SimdVF():
                for i in T.Parallel(n):
                    frag[i] = A[i]
            with T.SimdVF():
                for i in T.Parallel(n):
                    C[i] = frag[i] + 1.0

    tables = _vf_fragments(_lift(main))
    assert len(tables) == 2
    fragments = {
        fragment["name"]: fragment
        for table in tables
        for fragment in table["fragments"]
    }
    assert "frag" in fragments
    assert fragments["frag"]["definition_scope"] == "OUTSIDE_VF"
    assert bool(fragments["frag"]["cross_vf"]) is True
    assert bool(fragments["frag"]["escapes_vf"]) is False
    assert {int(vf) for vf in fragments["frag"]["use_vfs"]} == {0, 1}


def test_fragment_outside_vf_single_use_is_not_cross_vf():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((n,), "float32")
            with T.SimdVF():
                for i in T.Parallel(n):
                    frag[i] = A[i] + 1.0
                    C[i] = frag[i]

    tables = _vf_fragments(_lift(main))
    assert len(tables) == 1
    fragments = {fragment["name"]: fragment for fragment in tables[0]["fragments"]}
    assert "frag" in fragments
    assert fragments["frag"]["definition_scope"] == "OUTSIDE_VF"
    assert bool(fragments["frag"]["cross_vf"]) is False
    assert bool(fragments["frag"]["escapes_vf"]) is False


def test_fragment_used_only_by_top_level_copy_is_reported():
    """A fragment touched outside any Parallel still belongs to the table."""

    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            frag = T.alloc_fragment((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                # Fragment use outside P1's unit: a top-level copy.
                T.copy(frag, c_ub)
                for i in T.Parallel(n):
                    a_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    tables = _vf_fragments(_lift(main))
    assert len(tables) == 1
    fragments = {fragment["name"]: fragment for fragment in tables[0]["fragments"]}
    assert "frag" in fragments
    assert {int(vf) for vf in fragments["frag"]["use_vfs"]} == {0}


def test_fragment_touched_in_serial_loop_is_reported():
    """A fragment read inside a serial loop is outside P1, but still a use."""

    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            frag = T.alloc_fragment((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    a_ub[i] = a_ub[i] + 1.0
                for w in T.serial(n):
                    c_ub[w] = frag[w]
            T.copy(c_ub, C)

    tables = _vf_fragments(_lift(main))
    assert len(tables) == 1
    fragments = {fragment["name"]: fragment for fragment in tables[0]["fragments"]}
    assert "frag" in fragments


def test_manual_vf_region_keeps_footprint():
    """A manual VF rejects its regions, but they still record what they touch."""

    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                v = T.alloc_var("float32", init=0.0)
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + v
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    region = regions[0]
    assert region.status == "OPAQUE"
    # Manual is a VF-level verdict, so the region reports UNSUPPORTED and the
    # VF-level message stays in the reason text.
    assert region.reason_code == "UNSUPPORTED"
    assert "manual" in region.reason
    assert "VF-level rejection" in region.reason

    captures = {capture.buffer.name: capture for capture in region.captures}
    assert {"a_ub", "c_ub"} <= set(captures)
    assert all(capture.kind == "unknown" for capture in region.captures)
    op = region.ops[0]
    assert op.kind == "opaque"
    assert "c_ub" in [buffer.name for buffer in op.writes]
    assert "a_ub" in [buffer.name for buffer in op.reads]


def test_view_and_base_share_one_capture():
    """A footprint seeing a T.view and its base must not duplicate the capture.

    Capture identity is the data var, and the verifier rejects two captures for
    one data var; here the opaque region is produced from the syntactic
    footprint, which sees both buffers.
    """

    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            base = T.alloc_shared((n,), "float32")
            T.copy(A, base)
            with T.SimdVF():
                v = T.alloc_var("float32", init=0.0)  # forces the manual path
                for i in T.Parallel(n // 2):
                    view = T.view(base, shape=(n // 2, 2), dtype="float32")
                    view[i, 0] = base[2 * i] + v
            T.copy(base, C)

    regions = _regions(_lift(main))
    assert len(regions) == 1
    region = regions[0]
    assert region.status == "OPAQUE"
    data_vars = [str(capture.buffer.data) for capture in region.captures]
    assert len(data_vars) == len(set(data_vars))
    assert ir.verify_region(region) == ""


def test_fragment_group_downgrades_mates():
    """Two regions sharing a fragment are lifted or dropped together."""
    @T.prim_func
    def main(A: T.Tensor((96,), "float32"), C: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((96,), "float32")
            frag = T.alloc_fragment((64,), "float32")
            c_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    # Illegal cross-iteration RAW, and the region writes a
                    # fragment so the next region is grouped with it.
                    a_ub[i] = a_ub[i + 32]
                    frag[i] = a_ub[i]
                for i in T.Parallel(64):
                    c_ub[i] = frag[i] + 1.0
            T.copy(c_ub, C)

    regions = _regions(_lift(main))
    assert len(regions) == 2
    assert all(region.status == "OPAQUE" for region in regions)
    assert [region.reason_code for region in regions] == [
        "ILLEGAL",
        "GROUP_OPAQUE",
    ]
    assert int(regions[0].group) == int(regions[1].group)
    assert int(regions[0].group) >= 0
    assert "grouped with an OPAQUE" in regions[1].reason


def test_p1_can_be_disabled_by_config():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    with tvm.transform.PassContext(
        config={tilelang.PassConfigKey.TL_ASCEND_SIMD_LIFT: False}
    ):
        out = _lift(main)
    assert _regions(out) == []


def test_default_pipeline_keeps_legacy_lowering():
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] + 1.0
            T.copy(c_ub, C)

    with tvm.target.Target("ascend"):
        out = tilelang.lower(main)
    # The pass is metadata-only; the legacy parallel lowering still runs.
    assert out is not None
    assert "main_kernel_simd_vf_0" in out.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
