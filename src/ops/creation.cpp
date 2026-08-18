// src/ops/creation.cpp
#include "insight/ops/creation.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include <vector>

namespace ins {

static OpSchema creation_schema(const char *name) {
  return OpSchema(name, OpKind::Creation, {},
                  {{"out", OpArgumentKind::Array,
                    OpArgumentAccess::WriteOnly}})
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

static ArrayIterator creation_iterator(const char *name, const Array &output) {
  OpSchema schema = creation_schema(name);
  return schema.make_creation_iterator({output});
}

// ========== Basic Creation ==========

Array zeros(const Shape &shape, DType dtype, const Place &place) {
  return full(shape, 0.0, dtype, place);
}

Array ones(const Shape &shape, DType dtype, const Place &place) {
  return full(shape, 1.0, dtype, place);
}

Array full(const Shape &shape, double fill_value, DType dtype,
           const Place &place) {
  Array result(shape, dtype, place);
  ArrayIterator iter = creation_iterator("full", result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "full", place, dtype,
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::scalar_arg(&fill_value)},
      {OpRegistry::array_arg(arrays[0].layout_ptr())});
  return result;
}

Array eye(int64_t n, int64_t m, int64_t k, DType dtype, const Place &place) {
  if (m < 0)
    m = n;
  Shape shape({n, m});
  Array result(shape, dtype, place);
  ArrayIterator iter = creation_iterator("eye", result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "eye", place, dtype,
      {OpRegistry::array_arg(arrays[0].layout_ptr()), OpRegistry::scalar_arg(&k)},
      {OpRegistry::array_arg(arrays[0].layout_ptr())});
  return result;
}

// ========== Range Creation ==========

Array arange(double end, DType dtype, const Place &place) {
  return arange(0.0, end, 1.0, dtype, place);
}

Array arange(double start, double end, double step, DType dtype,
             const Place &place) {
  int64_t num = static_cast<int64_t>(std::ceil((end - start) / step));
  Array result(Shape({num}), dtype, place);
  ArrayIterator iter = creation_iterator("arange", result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "arange", place, dtype,
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::scalar_arg(&start), OpRegistry::scalar_arg(&step)},
      {OpRegistry::array_arg(arrays[0].layout_ptr())});
  return result;
}

Array linspace(double start, double stop, int64_t num, DType dtype,
               const Place &place) {
  Array result(Shape({num}), dtype, place);
  ArrayIterator iter = creation_iterator("linspace", result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "linspace", place, dtype,
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::scalar_arg(&start), OpRegistry::scalar_arg(&stop)},
      {OpRegistry::array_arg(arrays[0].layout_ptr())});
  return result;
}

Array logspace(double start, double stop, int64_t num, double base, DType dtype,
               const Place &place) {
  Array result(Shape({num}), dtype, place);
  ArrayIterator iter = creation_iterator("logspace", result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "logspace", place, dtype,
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::scalar_arg(&start), OpRegistry::scalar_arg(&stop),
       OpRegistry::scalar_arg(&base)},
      {OpRegistry::array_arg(arrays[0].layout_ptr())});
  return result;
}

// ========== Like Creation ==========

Array zeros_like(const Array &arr) {
  return zeros(arr.shape(), arr.dtype(), arr.place());
}

Array ones_like(const Array &arr) {
  return ones(arr.shape(), arr.dtype(), arr.place());
}

} // namespace ins