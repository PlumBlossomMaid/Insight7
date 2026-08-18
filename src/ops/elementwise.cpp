// src/ops/elementwise.cpp
#include "insight/ops/elementwise.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include "insight/ops/broadcast.h"
#include "insight/utils/promotion.h"
#include <vector>

namespace ins {

static OpSchema binary_elementwise_schema(const char *name,
                                          OpPromotionRule promotion_rule) {
  return OpSchema(
             name, OpKind::BinaryElementwise,
             {{"a", OpArgumentKind::Array, OpArgumentAccess::ReadOnly},
              {"b", OpArgumentKind::Array, OpArgumentAccess::ReadOnly}},
             {{"out", OpArgumentKind::Array, OpArgumentAccess::WriteOnly}})
      .promotion(promotion_rule)
      .broadcast(OpBroadcastRule::Inputs)
      .fallback(OpFallbackRule::StructuredCpu);
}

static Array cast_and_move(const Array &array, DType dtype,
                           const Place &place) {
  Array work = array.dtype() == dtype ? array : array.to(dtype);
  return work.place() == place ? work : work.to(place);
}

static Array schema_binary_op(const Array &a, const Array &b,
                              const char *kernel_name) {
  OpSchema schema =
      binary_elementwise_schema(kernel_name, OpPromotionRule::Numeric);
  DType out_dtype = schema.infer_binary_dtype(a.dtype(), b.dtype());

  Array a1 = (a.dtype() == out_dtype) ? a : a.to(out_dtype);
  Array b1 = (b.dtype() == out_dtype) ? b : b.to(out_dtype);
  Place target_place = promote_places(a1.place(), b1.place());
  a1 = cast_and_move(a1, out_dtype, target_place);
  b1 = cast_and_move(b1, out_dtype, target_place);

  Array out(schema.infer_elementwise_shape({a1, b1}), out_dtype, target_place);
  ArrayIterator iter = schema.make_array_iterator({a1, b1}, {out});
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      kernel_name, target_place, out_dtype,
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr())},
      {OpRegistry::array_arg(arrays[2].layout_ptr())});

  return out;
}

static Array schema_cmp_op(const Array &a, const Array &b,
                           const char *kernel_name) {
  OpSchema schema =
      binary_elementwise_schema(kernel_name, OpPromotionRule::Comparison);
  DType common_dtype = promote_types(a.dtype(), b.dtype());

  Array a1 = (a.dtype() == common_dtype) ? a : a.to(common_dtype);
  Array b1 = (b.dtype() == common_dtype) ? b : b.to(common_dtype);
  Place target_place = promote_places(a1.place(), b1.place());
  a1 = cast_and_move(a1, common_dtype, target_place);
  b1 = cast_and_move(b1, common_dtype, target_place);

  Array out(schema.infer_elementwise_shape({a1, b1}),
            schema.infer_binary_dtype(a1.dtype(), b1.dtype()), target_place);
  ArrayIterator iter = schema.make_array_iterator({a1, b1}, {out});
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      kernel_name, target_place, common_dtype,
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr())},
      {OpRegistry::array_arg(arrays[2].layout_ptr())});

  return out;
}

static Array schema_logical_op(const Array &a, const Array &b,
                               const char *kernel_name) {
  OpSchema schema =
      binary_elementwise_schema(kernel_name, OpPromotionRule::Identity);

  Array a1 = a.dtype() == DType::BOOL ? a : a.to(DType::BOOL);
  Array b1 = b.dtype() == DType::BOOL ? b : b.to(DType::BOOL);
  Place target_place = promote_places(a1.place(), b1.place());
  a1 = cast_and_move(a1, DType::BOOL, target_place);
  b1 = cast_and_move(b1, DType::BOOL, target_place);

  Array out(schema.infer_elementwise_shape({a1, b1}), DType::BOOL,
            target_place);
  ArrayIterator iter = schema.make_array_iterator({a1, b1}, {out});
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      kernel_name, target_place, DType::BOOL,
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr())},
      {OpRegistry::array_arg(arrays[2].layout_ptr())});

  return out;
}

// ============================================================================
// Arithmetic operations
// ============================================================================
Array add(const Array &a, const Array &b) {
  return schema_binary_op(a, b, "add");
}

Array sub(const Array &a, const Array &b) {
  return schema_binary_op(a, b, "sub");
}

Array mul(const Array &a, const Array &b) {
  return schema_binary_op(a, b, "mul");
}

Array div(const Array &a, const Array &b) {
  return schema_binary_op(a, b, "div");
}

// ============================================================================
// Power operation (special processing: converting integer exponent to floating
// point to avoid precision problems)
// ============================================================================
Array pow(const Array &a, const Array &b) {
  OpSchema schema = binary_elementwise_schema("pow", OpPromotionRule::Numeric);
  DType out_dtype = promote_types(a.dtype(), b.dtype());

  if ((is_floating_point(b.dtype()) || is_complex(b.dtype())) &&
      is_integer(out_dtype)) {
    out_dtype = DType::F64;
  }

  Array a1 = (a.dtype() == out_dtype) ? a : a.to(out_dtype);
  Array b1 = (b.dtype() == out_dtype) ? b : b.to(out_dtype);
  Place target_place = promote_places(a1.place(), b1.place());
  a1 = cast_and_move(a1, out_dtype, target_place);
  b1 = cast_and_move(b1, out_dtype, target_place);

  Array out(schema.infer_elementwise_shape({a1, b1}), out_dtype, target_place);
  ArrayIterator iter = schema.make_array_iterator({a1, b1}, {out});
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "pow", target_place, out_dtype,
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr())},
      {OpRegistry::array_arg(arrays[2].layout_ptr())});

  return out;
}

// ============================================================================
// Modulo operation
// ============================================================================
Array mod(const Array &a, const Array &b) {
  return schema_binary_op(a, b, "mod");
}

// ============================================================================
// comparison operation
// ============================================================================
Array equal(const Array &a, const Array &b) {
  return schema_cmp_op(a, b, "equal");
}

Array not_equal(const Array &a, const Array &b) {
  return schema_cmp_op(a, b, "not_equal");
}

Array greater(const Array &a, const Array &b) {
  return schema_cmp_op(a, b, "greater");
}

Array less(const Array &a, const Array &b) {
  return schema_cmp_op(a, b, "less");
}

Array greater_equal(const Array &a, const Array &b) {
  return schema_cmp_op(a, b, "greater_equal");
}

Array less_equal(const Array &a, const Array &b) {
  return schema_cmp_op(a, b, "less_equal");
}

// Alias
Array greater_than(const Array &a, const Array &b) { return greater(a, b); }

Array less_than(const Array &a, const Array &b) { return less(a, b); }

// ============================================================================
// Logical operation (convert to bool first)
// ============================================================================
Array logical_and(const Array &a, const Array &b) {
  return schema_logical_op(a, b, "logical_and");
}

Array logical_or(const Array &a, const Array &b) {
  return schema_logical_op(a, b, "logical_or");
}

Array logical_xor(const Array &a, const Array &b) {
  return schema_logical_op(a, b, "logical_xor");
}

// ============================================================================
// Bit operations (integer types)
// ============================================================================
Array bitwise_and(const Array &a, const Array &b) {
  return schema_binary_op(a, b, "bitwise_and");
}

Array bitwise_or(const Array &a, const Array &b) {
  return schema_binary_op(a, b, "bitwise_or");
}

Array bitwise_xor(const Array &a, const Array &b) {
  return schema_binary_op(a, b, "bitwise_xor");
}

Array bitwise_left_shift(const Array &a, const Array &b) {
  return schema_binary_op(a, b, "bitwise_left_shift");
}

Array bitwise_right_shift(const Array &a, const Array &b) {
  return schema_binary_op(a, b, "bitwise_right_shift");
}

// ============================================================================
// Max/Min
// ============================================================================
Array maximum(const Array &a, const Array &b) {
  return schema_binary_op(a, b, "maximum");
}

Array minimum(const Array &a, const Array &b) {
  return schema_binary_op(a, b, "minimum");
}

} // namespace ins