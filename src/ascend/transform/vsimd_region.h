/*!
 * \file vsimd_region.h
 * \brief Build a VSimdRegion from an analyzed Parallel nest, plus the reference
 *        scalar expander used as a structural self-check.
 */
#ifndef TVM_TL_ASCEND_TRANSFORM_VSIMD_REGION_H_
#define TVM_TL_ASCEND_TRANSFORM_VSIMD_REGION_H_

#include <tvm/ffi/string.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <string>
#include <vector>

#include "vsimd_capture_scope.h"
#include "vsimd_ir.h"

namespace tvm {
namespace tl {

/*! \brief Result of the region lift decision for one Parallel nest. */
struct VSimdRegionLift {
  bool lifted{false};
  bool internal_error{false}; // builder self-check failed: compiler bug
  std::string reason;         // fallback/internal-error reason
  vsimd::VSimdRegion region;  // valid only when lifted=true
};

/*!
 * \brief Build a region for a proven, closed, rectangular nest.
 *
 * `memory_status` is the legality status of this nest (LEGAL is required).
 * All other cases return lifted=false with a diagnostic; this function never
 * mutates IR. The built region carries the collapsed Parallel axes, the
 * ordered store ops of the innermost body, and the captured context.
 */
TVM_DLL VSimdRegionLift BuildRegion(const VSimdCaptureInfo &info,
                                    const std::string &memory_status);

/*!
 * \brief Reconstruct serial scalar / structured IR from a region.
 *
 * The result is a structural oracle, not the P3 physical lowering: stores are
 * rebuilt, axes become serial loops, and structured ops are re-emitted as the
 * tileop call they came from.
 *
 * When an op cannot be represented faithfully yet (today: a copy carrying a
 * transpose), the function returns an undefined Stmt and, if `unsupported` is
 * not null, fills it with a diagnostic instead of silently dropping the
 * semantic. Callers must treat that as a compiler-bug signal, not a user
 * fallback.
 */
TVM_DLL tirx::Stmt ExpandVSimdRegionToLoops(const vsimd::VSimdRegion &region,
                                            std::string *unsupported = nullptr);

/*! \brief Result of the cheap structural expander/lift consistency check. */
struct VSimdRegionVerify {
  bool ok{true};
  std::string reason;
};

/*!
 * \brief Check that the expanded statement declares every local view, uses only
 * captured/declared buffers, and reproduces the region axes and stores in source
 * order. This is a structural guard, not a numerical oracle.
 */
TVM_DLL VSimdRegionVerify
VerifyExpandedRegion(const vsimd::VSimdRegion &region,
                     const tirx::Stmt &expanded);

/*! \brief Lift every eligible nest and render the region/expanded dump. */
TVM_DLL ffi::String AnalyzeVSimdRegionReport(const tirx::PrimFunc &func);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VSIMD_REGION_H_
