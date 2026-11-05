/*!
 * \file vsimd_ir.h
 * \brief Typed, FFI-registered P1 vsimd region objects.
 *
 * A region describes one outer ``T.Parallel`` loop (its ``axes``) and the
 * ordered operations inside it (``ops``). The value is hardware independent:
 * it keeps the logical iteration space, access maps and store values, but no
 * lane, layout or instruction choice. P2/P3 own those decisions.
 *
 * The objects are registered with FFI reflection so they can be carried by an
 * annotation and read from Python without a hand-written schema.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_IR_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_IR_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/stmt.h>

#include <string>

#include "ascend/layout/ascend_layouts.h"

namespace tvm {
namespace tl {
namespace vsimd {

/*! \brief Op kind tags. Stored as strings so an annotation is self-describing. */
namespace op_kind {
inline constexpr const char *kParallel = "parallel";
inline constexpr const char *kStore = "store";
inline constexpr const char *kReduce = "reduce";
inline constexpr const char *kCopy = "copy";
inline constexpr const char *kFill = "fill";
inline constexpr const char *kOpaque = "opaque";
} // namespace op_kind

/*! \brief Region status tags. */
namespace region_status {
inline constexpr const char *kLifted = "LIFTED";
inline constexpr const char *kOpaque = "OPAQUE";
} // namespace region_status

/*! \brief One T.Parallel loop level: induction variable and its range. */
class VSimdAxisNode : public ffi::Object {
public:
  tirx::Var var;
  PrimExpr min;
  PrimExpr extent;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO("tl.vsimd.Axis", VSimdAxisNode, ffi::Object);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
};

class VSimdAxis : public ffi::ObjectRef {
public:
  TVM_DLL VSimdAxis(tirx::Var var, PrimExpr min, PrimExpr extent);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VSimdAxis, ffi::ObjectRef,
                                                VSimdAxisNode);
};

/*!
 * \brief A rectangular buffer window ``[base, base + extents)``.
 *
 * Used by structured ops (reduce/copy/fill). ``base`` may reference enclosing
 * loop variables, which is how a row-slice reduce inside an outer Parallel is
 * expressed.
 */
class VSimdBufRegionNode : public ffi::Object {
public:
  tirx::Buffer buffer;
  ffi::Array<PrimExpr> base;
  ffi::Array<PrimExpr> extents;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO("tl.vsimd.BufRegion", VSimdBufRegionNode,
                              ffi::Object);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
};

class VSimdBufRegion : public ffi::ObjectRef {
public:
  TVM_DLL VSimdBufRegion(tirx::Buffer buffer,
                         ffi::Array<PrimExpr> base,
                         ffi::Array<PrimExpr> extents);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VSimdBufRegion, ffi::ObjectRef,
                                                VSimdBufRegionNode);
};

/*!
 * \brief One logical buffer access (load or store).
 *
 * ``value`` is set for stores only. ``dep_axes`` lists the region axes the
 * index map actually uses; a missing axis means broadcast. ``guard`` is the
 * per-access predicate (reserved: constant true until guarded control flow is
 * modelled) and refers to the same axes as the index map.
 */
class VSimdAccessNode : public ffi::Object {
public:
  tirx::Buffer buffer;
  ffi::Array<PrimExpr> indices;
  ffi::Optional<PrimExpr> value;
  bool is_write{false};
  ffi::Array<Integer> dep_axes;
  PrimExpr guard;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO("tl.vsimd.Access", VSimdAccessNode, ffi::Object);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
};

class VSimdAccess : public ffi::ObjectRef {
public:
  TVM_DLL VSimdAccess(tirx::Buffer buffer, ffi::Array<PrimExpr> indices,
                      ffi::Optional<PrimExpr> value, bool is_write,
                      ffi::Array<Integer> dep_axes, PrimExpr guard);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VSimdAccess, ffi::ObjectRef,
                                                VSimdAccessNode);
};

/*!
 * \brief One operation inside a region.
 *
 * ``kind`` selects the fields that are meaningful:
 * - parallel: ``axes`` + ``ops`` (recursive);
 * - store:    ``accesses`` (one BufferStore statement and the loads it uses);
 * - reduce:   ``src``/``dst`` + ``reduce_kind``/``reduce_dim``/``clear``;
 * - copy:     ``src``/``dst`` (+ ``transpose``);
 * - fill:     ``dst`` + ``fill_value``;
 * - opaque:   ``reason`` + conservative ``reads``/``writes``.
 *
 * ``source_op`` keeps the originating tileop name for structured ops
 * (reduce/copy/fill). It is required for those kinds: it is what the reference
 * expander re-emits and what a consumer reproduces, so dropping it would lose
 * the operation.
 * ``guard`` is the op-level predicate (ancestor axes only; reserved).
 */
class VSimdOpNode : public ffi::Object {
public:
  ffi::String kind;

  // parallel
  ffi::Array<VSimdAxis> axes;
  ffi::Array<ffi::ObjectRef> ops;

  // store
  ffi::Array<VSimdAccess> accesses;

  // reduce / copy / fill
  ffi::Optional<VSimdBufRegion> src;
  ffi::Optional<VSimdBufRegion> dst;
  ffi::Optional<ffi::String> source_op;
  ffi::Optional<ffi::String> reduce_kind;
  ffi::Optional<PrimExpr> reduce_dim;
  ffi::Optional<PrimExpr> clear;
  ffi::Optional<PrimExpr> fill_value;
  ffi::Optional<PrimExpr> transpose;

  // common
  PrimExpr guard;
  ffi::String status; // "OK" | "OPAQUE"
  ffi::String reason;
  ffi::Array<tirx::Buffer> reads;
  ffi::Array<tirx::Buffer> writes;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO("tl.vsimd.Op", VSimdOpNode, ffi::Object);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  // Built field by field while a region is assembled.
  static constexpr bool _type_mutable = true;
};

class VSimdOp : public ffi::ObjectRef {
public:
  TVM_DLL explicit VSimdOp(ffi::String kind);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VSimdOp, ffi::ObjectRef,
                                                VSimdOpNode);

  bool IsKind(const char *kind) const;
};

/*!
 * \brief Typed view of a nested op reference.
 *
 * ``VSimdOpNode::ops`` is stored as ``Array<ObjectRef>`` because it is
 * recursive; the runtime type of every element is ``VSimdOp``.
 */
TVM_DLL VSimdOp AsOp(const ffi::ObjectRef &ref);

/*! \brief Nested ops of a ``parallel`` op, typed. */
TVM_DLL ffi::Array<VSimdOp> ChildOps(const VSimdOp &op);

/*! \brief One captured buffer: capture kind plus storage class and layout. */
class VSimdCaptureNode : public ffi::Object {
public:
  tirx::Buffer buffer;
  ffi::String kind;    // param / alloc / view / local
  ffi::String storage; // GLOBAL / SHARED / FRAGMENT / UNKNOWN
  ffi::Optional<Layout> layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO("tl.vsimd.Capture", VSimdCaptureNode,
                              ffi::Object);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
};

class VSimdCapture : public ffi::ObjectRef {
public:
  TVM_DLL VSimdCapture(tirx::Buffer buffer, ffi::String kind,
                       ffi::String storage, ffi::Optional<Layout> layout);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VSimdCapture, ffi::ObjectRef,
                                                VSimdCaptureNode);
};

/*! \brief A captured non-domain variable, or a scalar definition. */
class VSimdVarBindingNode : public ffi::Object {
public:
  tirx::Var var;
  ffi::String kind;                   // thread / outer_loop / bind
  ffi::Optional<PrimExpr> value; // bindings only

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO("tl.vsimd.VarBinding", VSimdVarBindingNode,
                              ffi::Object);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
};

class VSimdVarBinding : public ffi::ObjectRef {
public:
  TVM_DLL VSimdVarBinding(tirx::Var var, ffi::String kind,
                          ffi::Optional<PrimExpr> value);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VSimdVarBinding, ffi::ObjectRef,
                                                VSimdVarBindingNode);
};

/*! \brief A local ``T.view`` alias: view buffer mapped onto a base buffer. */
class VSimdViewNode : public ffi::Object {
public:
  tirx::Buffer view;
  tirx::Buffer base;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO("tl.vsimd.View", VSimdViewNode, ffi::Object);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
};

class VSimdView : public ffi::ObjectRef {
public:
  TVM_DLL VSimdView(tirx::Buffer view, tirx::Buffer base);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VSimdView, ffi::ObjectRef,
                                                VSimdViewNode);
};

/*!
 * \brief One lifted Parallel region.
 *
 * ``axes`` is the maximal contiguous chain of single-body Parallel loops
 * (collapsed); ``ops`` is the ordered body of the innermost loop. ``group``
 * marks regions that must be lifted or fallen back together: ``-1`` means "no
 * group" (a singleton), which is what the lift pass produces today; only the
 * fragment analysis assigns a non-negative group.
 */
class VSimdRegionNode : public ffi::Object {
public:
  int vf_index{-1};
  int region_index{-1};
  ffi::Array<VSimdAxis> axes;
  ffi::Array<VSimdOp> ops;
  ffi::Array<VSimdCapture> captures;
  ffi::Array<VSimdVarBinding> outer_vars;
  ffi::Array<tirx::Var> shape_syms;
  ffi::Array<VSimdView> local_views;
  ffi::Array<PrimExpr> assumptions;
  ffi::Array<VSimdVarBinding> bindings;
  ffi::String status;
  ffi::String reason_code;
  ffi::String reason;
  int group{-1};

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO("tl.vsimd.Region", VSimdRegionNode, ffi::Object);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  // Built field by field while a region is assembled.
  static constexpr bool _type_mutable = true;
};

class VSimdRegion : public ffi::ObjectRef {
public:
  TVM_DLL VSimdRegion();
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VSimdRegion, ffi::ObjectRef,
                                                VSimdRegionNode);
};

/*! \brief Storage class of a buffer scope. */
TVM_DLL ffi::String StorageKindOf(const tirx::Buffer &buffer);

/*! \brief Constant-true guard used until guarded control flow is modelled. */
TVM_DLL PrimExpr TrueGuard();

/*!
 * \brief Structural and schema invariants of a filled region.
 *
 * Checks op kinds, per-kind required fields, index rank, dep_axes range,
 * capture coverage and local-view consistency. A failure is a compiler bug in
 * the region builder, not a property of the user program.
 */
TVM_DLL bool VerifyRegion(const VSimdRegion &region, std::string *error);

/*! \brief Human-readable region dump (stable, grep friendly). */
TVM_DLL std::string DumpRegion(const VSimdRegion &region);

} // namespace vsimd
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VSIMD_IR_H_
