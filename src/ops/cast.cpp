// src/ops/cast.cpp
#include "insight/ops/cast.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include <vector>

namespace ins {

static OpSchema cast_schema() {
  return OpSchema("cast", OpKind::UnaryElementwise,
                  {{"x", OpArgumentKind::Array, OpArgumentAccess::ReadOnly}},
                  {{"out", OpArgumentKind::Array,
                    OpArgumentAccess::WriteOnly}})
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

Array cast(const Array &input, DType target_dtype, bool copy) {
  if (input.dtype() == target_dtype) {
    if (copy) {
      return input.copy();
    }
    return input;
  }

  Array output(input.shape(), target_dtype, input.place());
  OpSchema schema = cast_schema();
  ArrayIterator iter = schema.make_array_iterator({input}, {output});
  auto arrays = iter.arrays();
  int32_t target_dtype_int = static_cast<int32_t>(target_dtype);

  OpRegistry::launch_schema(
      "cast", input.place(), input.dtype(),
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::scalar_arg(&target_dtype_int)},
      {OpRegistry::array_arg(arrays[1].layout_ptr())});

  return arrays[1];
}

Array cast_like(const Array &input, const Array &other, bool copy) {
  return cast(input, other.dtype(), copy);
}

} // namespace ins