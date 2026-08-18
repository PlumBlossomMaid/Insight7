// src/ops/unary.cpp
#include "insight/core/array_iterator.h"
#include "insight/core/exception.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include "insight/ops/unary.h"
#include <string>
#include <vector>

namespace ins {

static OpSchema unary_elementwise_schema(const char *name) {
  return OpSchema(name, OpKind::UnaryElementwise,
                  {{"x", OpArgumentKind::Array, OpArgumentAccess::ReadOnly}},
                  {{"out", OpArgumentKind::Array,
                    OpArgumentAccess::WriteOnly}})
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

static DType unary_output_dtype(const char *name, DType input_dtype) {
  std::string op_name(name);
  if (op_name == "logical_not" || op_name == "isnan" || op_name == "isinf" ||
      op_name == "isfinite") {
    return DType::BOOL;
  }
  if (op_name == "abs") {
    if (input_dtype == DType::C64)
      return DType::F64;
    if (input_dtype == DType::C32)
      return DType::F32;
  }
  return input_dtype;
}

static Array schema_unary_op_with_dtype(const Array &x, const char *kernel_name,
                                         DType out_dtype) {
  INS_CHECK(x.defined(), kernel_name, ": input is undefined");

  OpSchema schema = unary_elementwise_schema(kernel_name);
  Array out(x.shape(), out_dtype, x.place());
  ArrayIterator iter = schema.make_array_iterator({x}, {out});
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      kernel_name, x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[0].layout_ptr())},
      {OpRegistry::array_arg(arrays[1].layout_ptr())});

  return out;
}

static Array schema_unary_op(const Array &x, const char *kernel_name) {
  return schema_unary_op_with_dtype(
      x, kernel_name, unary_output_dtype(kernel_name, x.dtype()));
}

// ============================================================================
// Helper macro for unary operations
// ============================================================================

#define DEFINE_UNARY_OP(op_name)                                               \
  Array op_name(const Array &x) { return schema_unary_op(x, #op_name); }

// ============================================================================
// Basic math operations
// ============================================================================

DEFINE_UNARY_OP(abs);
DEFINE_UNARY_OP(negative);
DEFINE_UNARY_OP(square);
DEFINE_UNARY_OP(reciprocal);

// ============================================================================
// Exponential and logarithmic
// ============================================================================

DEFINE_UNARY_OP(exp);
DEFINE_UNARY_OP(exp2);
DEFINE_UNARY_OP(expm1);
DEFINE_UNARY_OP(log);
DEFINE_UNARY_OP(log2);
DEFINE_UNARY_OP(log10);
DEFINE_UNARY_OP(log1p);

// ============================================================================
// Power and root
// ============================================================================

DEFINE_UNARY_OP(sqrt);
DEFINE_UNARY_OP(cbrt);

// ============================================================================
// Trigonometric
// ============================================================================

DEFINE_UNARY_OP(sin);
DEFINE_UNARY_OP(cos);
DEFINE_UNARY_OP(tan);
DEFINE_UNARY_OP(asin);
DEFINE_UNARY_OP(acos);
DEFINE_UNARY_OP(atan);

// ============================================================================
// Hyperbolic
// ============================================================================

DEFINE_UNARY_OP(sinh);
DEFINE_UNARY_OP(cosh);
DEFINE_UNARY_OP(tanh);
DEFINE_UNARY_OP(asinh);
DEFINE_UNARY_OP(acosh);
DEFINE_UNARY_OP(atanh);

// ============================================================================
// Rounding
// ============================================================================

DEFINE_UNARY_OP(floor);
DEFINE_UNARY_OP(ceil);
DEFINE_UNARY_OP(trunc);
DEFINE_UNARY_OP(rint);

// ============================================================================
// Sign
// ============================================================================

DEFINE_UNARY_OP(sign);

// ============================================================================
// Logical and bitwise
// ============================================================================

DEFINE_UNARY_OP(logical_not);
DEFINE_UNARY_OP(bitwise_not);

// ============================================================================
// Complex
// ============================================================================

Array conj(const Array &x) {
  INS_CHECK(x.defined(), "conj: input is undefined");

  // For real numbers, conjugate is identity
  if (!is_complex(x.dtype())) {
    return x.copy();
  }

  return schema_unary_op(x, "conj");
}

Array angle(const Array &x) {
  INS_CHECK(x.defined(), "angle: input is undefined");

  DType out_dtype;
  if (x.dtype() == DType::C32 || x.dtype() == DType::F32 ||
      x.dtype() == DType::F16 || x.dtype() == DType::BF16) {
    out_dtype = DType::F32;
  } else if (x.dtype() == DType::C64 || x.dtype() == DType::F64) {
    out_dtype = DType::F64;
  } else {
    // For integer types, angle is 0 for non-negative, pi for negative
    out_dtype = DType::F64;
  }

  return schema_unary_op_with_dtype(x, "angle", out_dtype);
}

// ============================================================================
// Degree/radian conversion
// ============================================================================

DEFINE_UNARY_OP(deg2rad);
DEFINE_UNARY_OP(rad2deg);

// ============================================================================
// Is finite/inf/nan
// ============================================================================

DEFINE_UNARY_OP(isnan);
DEFINE_UNARY_OP(isinf);
DEFINE_UNARY_OP(isfinite);

} // namespace ins