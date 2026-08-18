// src/ops/indexing.cpp
/**
 * @file indexing.cpp
 * @brief Indexing operations for array manipulation.
 *
 * Provides advanced indexing operations including take, put, where,
 * nonzero, argsort, topk, unique, and other array indexing utilities.
 */

#include "insight/ops/indexing.h"
#include "insight/core/axis.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include "insight/ops/broadcast.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/reduction.h"
#include "insight/utils/promotion.h"
#include <cmath>
#include <insight/io/print.h>
#include <iostream>
#include <string>
#include <vector>
namespace ins {

static OpSchema indexing_schema(const char *name, size_t input_count,
                                size_t output_count = 1) {
  std::vector<OpArgumentSchema> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    inputs.push_back({"x" + std::to_string(i), OpArgumentKind::Array,
                      OpArgumentAccess::ReadOnly});
  }

  std::vector<OpArgumentSchema> outputs;
  outputs.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    outputs.push_back({output_count == 1 ? "out" : "out" + std::to_string(i),
                       OpArgumentKind::Array, OpArgumentAccess::WriteOnly});
  }

  return OpSchema(name, OpKind::Indexing, inputs, outputs)
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

static ArrayIterator indexing_iterator(const char *name,
                                       const std::vector<Array> &inputs,
                                       const std::vector<Array> &outputs) {
  OpSchema schema = indexing_schema(name, inputs.size(), outputs.size());
  return schema.make_transform_iterator(inputs, outputs);
}

static ArrayIterator indexing_iterator(const char *name,
                                       const std::vector<Array> &inputs,
                                       const Array &output) {
  return indexing_iterator(name, inputs, std::vector<Array>{output});
}

static ArrayIterator indexing_creation_iterator(const char *name,
                                                const Array &output) {
  OpSchema schema(name, OpKind::Creation, {},
                  {{"out", OpArgumentKind::Array,
                    OpArgumentAccess::WriteOnly}});
  return schema.make_creation_iterator({output});
}

static OpSchema dynamic_indexing_schema(const char *name, size_t output_count) {
  return indexing_schema(name, 1, output_count);
}

static std::vector<OpRegistry::Arg> dynamic_indexing_outputs(
    const OpSchema &schema, const std::vector<Array *> &outputs) {
  INS_CHECK(outputs.size() == schema.outputs().size(), "dynamic indexing '",
            schema.name(), "': expected ", schema.outputs().size(),
            " outputs, got ", outputs.size());

  std::vector<OpRegistry::Arg> result;
  result.reserve(outputs.size());
  for (Array *output : outputs) {
    INS_CHECK(output != nullptr, "dynamic indexing '", schema.name(),
              "': output placeholder is null");
    result.push_back(OpRegistry::array_arg(output->layout_ptr()));
  }
  return result;
}

// ============================================================================
// Helper: Prepare flattened array for indexing
// ============================================================================

/**
 * @brief Prepare array for indexing by flattening or making contiguous.
 *
 * @param x Input array
 * @param axis Optional axis (if provided, only make contiguous)
 * @return Prepared array
 */
static Array prepare_flattened(const Array &x, std::optional<int> axis) {
  if (!axis.has_value()) {
    return x.reshape(Shape({x.numel()}));
  }
  return x.contiguous();
}

/**
 * @brief Broadcast shapes for take_along_axis operation.
 *
 * @param x_shape Shape of the input array
 * @param idx_shape Shape of the indices array
 * @param axis The axis along which to take
 * @return Broadcasted output shape
 */
static Shape broadcast_shapes_for_indexing(const Shape &x_shape,
                                           const Shape &idx_shape, int axis) {
  int ndim = x_shape.ndim();
  std::vector<int64_t> out_dims(ndim);
  for (int i = 0; i < ndim; ++i) {
    if (i == axis) {
      out_dims[i] = idx_shape.dim(i);
    } else {
      int64_t xdim = x_shape.dim(i);
      int64_t idxdim = (i < idx_shape.ndim()) ? idx_shape.dim(i) : 1;
      if (xdim != idxdim && xdim != 1 && idxdim != 1) {
        INS_THROW("broadcast_shapes_for_indexing: shape mismatch at axis ", i,
                  ": ", xdim, " vs ", idxdim);
      }
      out_dims[i] = std::max(xdim, idxdim);
    }
  }
  return Shape(out_dims);
}

// ============================================================================
// take
// ============================================================================

/**
 * @brief Take elements from an array along an axis.
 *
 * @param x Input array
 * @param indices Indices to take (1D array)
 * @param axis Axis along which to take (optional, if not provided, flatten)
 * @return Array of taken elements
 */
Array take(const Array &x, const Array &indices, std::optional<int> axis) {
  Array prepared = prepare_flattened(x, axis);

  Array idx = indices;
  if (idx.dtype() != DType::I64) {
    idx = idx.to(DType::I64);
  }
  if (idx.place() != x.place()) {
    idx = idx.to(x.place());
  }

  int normalized_axis = axis.has_value()
                            ? normalize_axis(axis.value(), x.shape().ndim(), "take")
                            : -1;
  Shape out_shape;
  if (axis.has_value()) {
    std::vector<int64_t> dims = x.shape().dims();
    dims[normalized_axis] = idx.numel();
    out_shape = Shape(dims);
  } else {
    out_shape = Shape({idx.numel()});
  }

  Array result(out_shape, x.dtype(), x.place());

  bool has_axis = axis.has_value();

  ArrayIterator iter = indexing_iterator("take", {prepared, idx}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "take", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[2].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::scalar_arg(&normalized_axis),
       OpRegistry::scalar_arg(&has_axis)},
      {OpRegistry::array_arg(arrays[2].layout_ptr())});

  return result;
}

// ============================================================================
// take_along_axis
// ============================================================================

/**
 * @brief Take values from the input array using indices array along an axis.
 *
 * @param x Input array
 * @param indices Indices array (same shape as output except on axis)
 * @param axis Axis along which to take
 * @return Array of taken values
 */
Array take_along_axis(const Array &x, const Array &indices, int axis) {
  int ax = normalize_axis(axis, x.shape().ndim(), "take_along_axis");

  Array idx = indices;
  if (idx.dtype() != DType::I64) {
    idx = idx.to(DType::I64);
  }
  if (idx.place() != x.place()) {
    idx = idx.to(x.place());
  }

  Shape out_shape = broadcast_shapes_for_indexing(x.shape(), idx.shape(), ax);

  Array idx_broadcasted = broadcast_to(idx, out_shape);

  Array result(out_shape, x.dtype(), x.place());

  ArrayIterator iter =
      indexing_iterator("take_along_axis", {x, idx_broadcasted}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "take_along_axis", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[2].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::scalar_arg(&ax)},
      {OpRegistry::array_arg(arrays[2].layout_ptr())});

  return result;
}

// ============================================================================
// put
// ============================================================================

/**
 * @brief Put values into an array at specified indices.
 *
 * @param x Input array (will be copied)
 * @param indices Indices where to put values
 * @param values Values to put
 * @param axis Axis along which to put (optional)
 * @return Array with values placed
 */
Array put(const Array &x, const Array &indices, const Array &values,
          std::optional<int> axis) {
  Array prepared = prepare_flattened(x, axis);

  Array idx = indices;
  if (idx.dtype() != DType::I64) {
    idx = idx.to(DType::I64);
  }
  if (idx.place() != x.place()) {
    idx = idx.to(x.place());
  }

  Array val = values;
  if (val.numel() == 1) {
    val = broadcast_to(val, Shape({idx.numel()}));
  }
  INS_CHECK(val.numel() == idx.numel(),
            "put: values must broadcast to indices shape");

  Array result = prepared.copy();

  ArrayIterator iter = indexing_iterator("put", {result, idx, val}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "put", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[2].layout_ptr())},
      {OpRegistry::array_arg(arrays[3].layout_ptr())});

  if (!axis.has_value()) {
    return result.reshape(x.shape());
  }
  return result;
}

// ============================================================================
// put_along_axis
// ============================================================================

/**
 * @brief Put values into an array along an axis using indices.
 *
 * @param x Input array (will be copied)
 * @param indices Indices where to put values
 * @param values Values to put
 * @param axis Axis along which to put
 * @return Array with values placed
 */
Array put_along_axis(const Array &x, const Array &indices, const Array &values,
                     int axis) {
  int ax = normalize_axis(axis, x.shape().ndim(), "put_along_axis");

  Array idx = indices;
  if (idx.dtype() != DType::I64) {
    idx = idx.to(DType::I64);
  }
  if (idx.place() != x.place()) {
    idx = idx.to(x.place());
  }

  Array val = values;
  if (val.dtype() != x.dtype()) {
    val = val.to(x.dtype());
  }
  if (val.place() != x.place()) {
    val = val.to(x.place());
  }

  if (val.numel() == 1) {
    val = broadcast_to(val, idx.shape());
  }

  Array result = x.copy();

  ArrayIterator iter =
      indexing_iterator("put_along_axis", {result, idx, val}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "put_along_axis", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[2].layout_ptr()),
       OpRegistry::scalar_arg(&ax)},
      {OpRegistry::array_arg(arrays[3].layout_ptr())});

  return result;
}

// ============================================================================
// gather / scatter (aliases)
// ============================================================================

/**
 * @brief Gather values along an axis using indices (alias for take_along_axis).
 */
Array gather(const Array &x, int dim, const Array &index) {
  return take_along_axis(x, index, dim);
}

/**
 * @brief Scatter values into an array (alias for scatter_reduce with replace
 * mode).
 */
Array scatter(const Array &x, int dim, const Array &index, const Array &src) {
  return scatter_reduce(x, dim, index, src, "replace");
}

/**
 * @brief Scatter and add values into an array.
 */
Array scatter_add(const Array &x, int dim, const Array &index,
                  const Array &src) {
  return scatter_reduce(x, dim, index, src, "add");
}

// ============================================================================
// scatter_reduce
// ============================================================================

/**
 * @brief Scatter values into an array with reduction.
 *
 * @param x Input array (will be copied)
 * @param dim Dimension along which to scatter
 * @param index Indices where to scatter
 * @param src Source values
 * @param reduce Reduction mode: "replace", "add", "mul", "max", "min"
 * @return Array with scattered values
 */
Array scatter_reduce(const Array &x, int dim, const Array &index,
                     const Array &src, const std::string &reduce) {
  int d = normalize_axis(dim, x.shape().ndim(), "scatter_reduce");

  Array idx = index;
  if (idx.dtype() != DType::I64) {
    idx = idx.to(DType::I64);
  }
  if (idx.place() != x.place()) {
    idx = idx.to(x.place());
  }

  Array src_broadcast = src;
  if (src_broadcast.shape() != idx.shape()) {
    src_broadcast = broadcast_to(src_broadcast, idx.shape());
  }

  Array result = x.copy();

  ArrayIterator iter =
      indexing_iterator("scatter_reduce", {result, idx, src_broadcast}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "scatter_reduce", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[2].layout_ptr()),
       OpRegistry::scalar_arg(&d),
       {const_cast<char *>(reduce.c_str()), OpRegistry::ArgKind::HostScalar}},
      {OpRegistry::array_arg(arrays[3].layout_ptr())});

  return result;
}

// ============================================================================
// masked_select
// ============================================================================

/**
 * @brief Select elements from an array where mask is true.
 *
 * @param x Input array
 * @param mask Boolean mask array
 * @return 1D array of selected elements
 */
Array masked_select(const Array &x, const Array &mask) {
  Array condition = mask;
  if (condition.dtype() != DType::BOOL) {
    condition = condition.to(DType::BOOL);
  }
  if (condition.place() != x.place()) {
    condition = condition.to(x.place());
  }

  if (condition.shape() != x.shape()) {
    condition = broadcast_to(condition, x.shape());
  }

  // First pass: count number of true elements
  Array flattened_cond = condition.reshape(Shape({condition.numel()}));
  Array cnt = count_nonzero(flattened_cond);
  int64_t count = cnt.item<int64_t>();

  Array result(Shape({count}), x.dtype(), x.place());

  ArrayIterator iter = indexing_iterator("masked_select", {x, condition}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "masked_select", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[2].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr())},
      {OpRegistry::array_arg(arrays[2].layout_ptr())});

  return result;
}

// ============================================================================
// compress
// ============================================================================

/**
 * @brief Select slices along an axis where condition is true.
 *
 * @param x Input array
 * @param condition 1D boolean array
 * @param axis Axis along which to compress
 * @return Compressed array
 */
Array compress(const Array &x, const Array &condition,
               std::optional<int> axis) {
  int ax = normalize_axis(axis.value_or(0), x.shape().ndim(), "compress");

  Array cond = condition;
  if (cond.dtype() != DType::BOOL) {
    cond = cond.to(DType::BOOL);
  }

  int64_t axis_dim = x.shape().dim(ax);
  if (cond.numel() != axis_dim) {
    INS_THROW("compress: condition length must match axis dimension");
  }

  Array cnt = count_nonzero(cond);
  int64_t keep_count = cnt.item<int64_t>();

  std::vector<int64_t> out_dims = x.shape().dims();
  out_dims[ax] = keep_count;
  Shape out_shape(out_dims);

  Array result(out_shape, x.dtype(), x.place());

  ArrayIterator iter = indexing_iterator("compress", {x, cond}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "compress", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[2].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::scalar_arg(&ax)},
      {OpRegistry::array_arg(arrays[2].layout_ptr())});

  return result;
}

// ============================================================================
// where
// ============================================================================

/**
 * @brief Return elements chosen from x or y depending on condition.
 *
 * @param condition Boolean condition array
 * @param x Values to take where condition is true
 * @param y Values to take where condition is false
 * @return Array of same shape as condition
 */
Array where(const Array &condition, const Array &x, const Array &y) {
  auto broadcasted = broadcast_arrays({condition, x, y});
  Array cond = broadcasted[0];
  if (cond.dtype() != DType::BOOL) {
    cond = cond.to(DType::BOOL);
  }
  Array X = broadcasted[1];
  Array Y = broadcasted[2];

  Array result(cond.shape(), X.dtype(), X.place());

  ArrayIterator iter = indexing_iterator("where", {cond, X, Y}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "where", condition.place(), X.dtype(),
      {OpRegistry::array_arg(arrays[3].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[2].layout_ptr())},
      {OpRegistry::array_arg(arrays[3].layout_ptr())});

  return result;
}

// ============================================================================
// nonzero
// ============================================================================

/**
 * @brief Return the indices of the non-zero elements.
 *
 * @param x Input array
 * @return 2D array of shape (ndim, nz_count) containing indices
 */
Array nonzero(const Array &x) {
  Array result;
  OpSchema schema = dynamic_indexing_schema("nonzero", 1);
  OpRegistry::launch_schema(
      "nonzero", x.place(), x.dtype(),
      {OpRegistry::array_arg(x.layout_ptr())},
      dynamic_indexing_outputs(schema, {&result}));

  return Array(result.layout_ptr());
}

// ============================================================================
// flatnonzero
// ============================================================================

/**
 * @brief Return the indices of the non-zero elements in the flattened array.
 *
 * @param x Input array
 * @return 1D array of indices
 */
Array flatnonzero(const Array &x) {
  Array flat_x = x.reshape(Shape({x.numel()}));
  Array nz = nonzero(flat_x);

  if (nz.numel() == 0) {
    return nz;
  }
  // nz shape is (1, n) for 1D input, flatten to (n,)
  return nz.reshape(Shape({nz.numel()}));
}

// ============================================================================
// argsort
// ============================================================================

/**
 * @brief Return the indices that would sort the array.
 *
 * @param x Input array
 * @param axis Axis along which to sort
 * @param descending Whether to sort in descending order
 * @return Indices array of same shape as x
 */
Array argsort(const Array &x, int axis, bool descending) {
  int ndim = x.shape().ndim();
  int ax = normalize_axis(axis, ndim, "argsort");

  // Move axis to last dimension for contiguous processing
  Array prepared = x;
  std::vector<int> perm(ndim);
  for (int i = 0; i < ndim; ++i)
    perm[i] = i;

  if (ax != ndim - 1) {
    std::swap(perm[ax], perm[ndim - 1]);
    prepared = prepared.transpose(perm);
    prepared = prepared.contiguous();
  }

  Shape out_shape = x.shape();
  Array result(out_shape, DType::I64, x.place());

  ArrayIterator iter = indexing_iterator("argsort", {prepared}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "argsort", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::scalar_arg(&descending)},
      {OpRegistry::array_arg(arrays[1].layout_ptr())});

  // Transpose back if needed
  if (ax != ndim - 1) {
    std::vector<int> inv_perm(ndim);
    for (int i = 0; i < ndim; ++i)
      inv_perm[perm[i]] = i;
    result = result.transpose(inv_perm);
  }

  return result;
}

// ============================================================================
// sort
// ============================================================================

/**
 * @brief Return a sorted copy of the array.
 *
 * @param x Input array
 * @param axis Axis along which to sort
 * @param descending Whether to sort in descending order
 * @return Sorted array
 */
Array sort(const Array &x, int axis, bool descending) {
  Array indices = argsort(x, axis, descending);
  return take_along_axis(x, indices, axis);
}

// ============================================================================
// topk
// ============================================================================

/**
 * @brief Return the top k values and their indices.
 *
 * @param x Input array
 * @param k Number of top elements to return
 * @param axis Axis along which to find top elements
 * @param largest Whether to return largest (true) or smallest (false)
 * @param sorted Whether to return sorted results
 * @return Tuple of (values, indices)
 */
std::tuple<Array, Array> topk(const Array &x, int64_t k, int axis, bool largest,
                              bool sorted) {
  int ndim = x.shape().ndim();
  int ax = normalize_axis(axis, ndim, "topk");
  INS_CHECK(k > 0, "topk: k must be positive");

  int64_t axis_size = x.shape().dim(ax);
  if (k > axis_size)
    k = axis_size;

  // Move axis to last dimension for contiguous processing
  Array prepared = x;
  std::vector<int> perm(ndim);
  for (int i = 0; i < ndim; ++i)
    perm[i] = i;

  if (ax != ndim - 1) {
    std::swap(perm[ax], perm[ndim - 1]);
    prepared = prepared.transpose(perm);
    prepared = prepared.contiguous();
  }

  std::vector<int64_t> out_dims = prepared.shape().dims();
  out_dims.back() = k;
  Shape out_shape(out_dims);

  Array values(out_shape, x.dtype(), x.place());
  Array indices(out_shape, DType::I64, x.place());

  ArrayIterator iter = indexing_iterator("topk", {prepared}, {values, indices});
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "topk", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[2].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()), OpRegistry::scalar_arg(&k),
       OpRegistry::scalar_arg(&largest), OpRegistry::scalar_arg(&sorted)},
      {OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[2].layout_ptr())});

  // Transpose back if needed
  if (ax != ndim - 1) {
    std::vector<int> inv_perm(ndim);
    for (int i = 0; i < ndim; ++i)
      inv_perm[perm[i]] = i;
    values = values.transpose(inv_perm);
    indices = indices.transpose(inv_perm);
  }

  return {values, indices};
}

// ============================================================================
// searchsorted
// ============================================================================

/**
 * @brief Find indices where elements should be inserted to maintain order.
 *
 * @param x Sorted 1D array
 * @param v Values to insert
 * @param side 'left' or 'right'
 * @param sorter Optional indices that sort x
 * @return Indices array of same shape as v
 */
Array searchsorted(const Array &x, const Array &v, const std::string &side,
                   std::optional<Array> sorter) {
  INS_CHECK(x.shape().ndim() == 1, "searchsorted: x must be 1D");

  Array sorted_x = x;
  if (sorter.has_value()) {
    sorted_x = take(sorted_x, sorter.value());
  }

  Array result(v.shape(), DType::I64, v.place());

  int side_code = (side == "left") ? 0 : 1;

  ArrayIterator iter = indexing_iterator("searchsorted", {sorted_x, v}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "searchsorted", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[2].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::scalar_arg(&side_code)},
      {OpRegistry::array_arg(arrays[2].layout_ptr())});

  return result;
}

// ============================================================================
// unique
// ============================================================================

/**
 * @brief Return the unique elements of an array.
 *
 * @param x Input array
 * @param return_indices Whether to return indices of first occurrences
 * @param return_inverse Whether to return inverse indices
 * @param return_counts Whether to return counts
 * @return UniqueResult structure containing requested arrays
 */
UniqueResult unique(const Array &x, bool return_indices, bool return_inverse,
                    bool return_counts) {
  Array flattened = x.reshape(Shape({x.numel()}));

  Array unique_arr;
  Array indices_arr;
  Array inverse_arr;
  Array counts_arr;

  size_t output_count = 1;
  if (return_indices)
    ++output_count;
  if (return_inverse)
    ++output_count;
  if (return_counts)
    ++output_count;

  OpSchema schema = dynamic_indexing_schema("unique", output_count);
  std::vector<Array *> output_arrays = {&unique_arr};
  if (return_indices)
    output_arrays.push_back(&indices_arr);
  if (return_inverse)
    output_arrays.push_back(&inverse_arr);
  if (return_counts)
    output_arrays.push_back(&counts_arr);

  OpRegistry::launch_schema(
      "unique", x.place(), x.dtype(),
      {OpRegistry::array_arg(flattened.layout_ptr()),
       OpRegistry::scalar_arg(&return_indices),
       OpRegistry::scalar_arg(&return_inverse),
       OpRegistry::scalar_arg(&return_counts)},
      dynamic_indexing_outputs(schema, output_arrays));

  UniqueResult result;
  result.unique = Array(unique_arr.layout_ptr());
  if (return_indices) {
    result.indices = Array(indices_arr.layout_ptr());
  }
  if (return_inverse) {
    result.inverse = Array(inverse_arr.layout_ptr());
  }
  if (return_counts) {
    result.counts = Array(counts_arr.layout_ptr());
  }
  return result;
}

// ============================================================================
// partition
// ============================================================================

/**
 * @brief Return a partially sorted copy of the array.
 *
 * @param x Input array
 * @param kth Element index that will be in its sorted position
 * @param axis Axis along which to partition
 * @return Partially sorted array
 */
Array partition(const Array &x, int64_t kth, int axis) {
  int ax = normalize_axis(axis, x.shape().ndim(), "partition");
  INS_CHECK(kth >= 0 && kth < x.shape().dim(ax), "partition: kth out of range");

  Array result(x.shape(), x.dtype(), x.place());

  ArrayIterator iter = indexing_iterator("partition", {x}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "partition", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::scalar_arg(&kth), OpRegistry::scalar_arg(&ax)},
      {OpRegistry::array_arg(arrays[1].layout_ptr())});

  return result;
}

// ============================================================================
// argpartition
// ============================================================================

/**
 * @brief Return the indices that would partially sort the array.
 *
 * @param x Input array
 * @param kth Element index that will be in its sorted position
 * @param axis Axis along which to partition
 * @return Indices array
 */
Array argpartition(const Array &x, int64_t kth, int axis) {
  int ax = normalize_axis(axis, x.shape().ndim(), "argpartition");
  INS_CHECK(kth >= 0 && kth < x.shape().dim(ax),
            "argpartition: kth out of range");

  Array result(x.shape(), DType::I64, x.place());

  ArrayIterator iter = indexing_iterator("argpartition", {x}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "argpartition", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::scalar_arg(&kth), OpRegistry::scalar_arg(&ax)},
      {OpRegistry::array_arg(arrays[1].layout_ptr())});

  return result;
}

// ============================================================================
// lexsort
// ============================================================================

/**
 * @brief Perform an indirect stable sort using a sequence of keys.
 *
 * @param keys Array of keys (first dimension is number of keys)
 * @param axis Axis along which to sort (default -1)
 * @return Indices that sort the keys
 */
Array lexsort(const Array &keys, int axis) {
  int ndim = keys.shape().ndim();
  INS_CHECK(ndim >= 1, "lexsort: keys must be at least 1D");
  int ax = normalize_axis(axis, ndim, "lexsort");

  // Move target axis to last position
  std::vector<int> perm(ndim);
  for (int i = 0; i < ndim; ++i)
    perm[i] = i;
  if (ax != ndim - 1) {
    std::swap(perm[ax], perm[ndim - 1]);
  }

  Array transposed = keys.transpose(perm);
  transposed = transposed.contiguous();

  const Shape &trans_shape = transposed.shape();

  // Determine batch dimensions (all except last axis)
  int64_t last_dim = trans_shape.dim(ndim - 1);
  int64_t batch_size = 1;
  for (int i = 0; i < ndim - 1; ++i) {
    batch_size *= trans_shape.dim(i);
  }

  // Number of keys = total_size / (batch_size * last_dim)
  int64_t total = transposed.numel();
  int64_t nkeys = total / (batch_size * last_dim);

  Array result(keys.shape(), DType::I64, keys.place());

  ArrayIterator iter = indexing_iterator("lexsort", {transposed}, result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "lexsort", keys.place(), keys.dtype(),
      {OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::scalar_arg(&batch_size), OpRegistry::scalar_arg(&last_dim),
       OpRegistry::scalar_arg(&nkeys)},
      {OpRegistry::array_arg(arrays[1].layout_ptr())});

  // Transpose back if needed
  if (ax != ndim - 1) {
    std::vector<int> inv_perm(ndim);
    for (int i = 0; i < ndim; ++i)
      inv_perm[perm[i]] = i;
    result = result.transpose(inv_perm);
  }

  return result;
}

// ============================================================================
// indices
// ============================================================================

/**
 * @brief Return an array representing the indices of a grid.
 *
 * @param shape Shape of the grid
 * @param sparse If true, return sparse representation (not implemented)
 * @return Dense grid indices array of shape (ndim, dim0, dim1, ...)
 */
Array indices(const Shape &shape, bool sparse) {
  int ndim = shape.ndim();
  if (sparse) {
    INS_THROW("indices: sparse mode not yet implemented - use dense mode");
  }

  // Dense mode: output shape is (ndim, dim0, dim1, ...)
  std::vector<int64_t> out_dims;
  out_dims.push_back(ndim);
  for (int i = 0; i < ndim; ++i) {
    out_dims.push_back(shape.dim(i));
  }
  Shape out_shape(out_dims);

  Array result(out_shape, DType::I64, CPUPlace());

  // Convert shape to int64_t array for kernel
  std::vector<int64_t> shape_dims = shape.dims();
  int64_t *shape_ptr = shape_dims.data();
  int ndim_val = ndim;

  ArrayIterator iter = indexing_creation_iterator("indices", result);
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "indices", CPUPlace(), DType::I64,
      {OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::scalar_arg(&ndim_val),
       {shape_ptr, OpRegistry::ArgKind::HostScalar}},
      {OpRegistry::array_arg(arrays[0].layout_ptr())});

  return result;
}

// ============================================================================
// ix_
// ============================================================================

/**
 * @brief Construct an open mesh from multiple sequences.
 *
 * @param arrays 1D arrays to form the mesh
 * @return Vector of broadcasted arrays for each dimension
 */
std::vector<Array> ix_(const std::vector<Array> &arrays) {
  int n = static_cast<int>(arrays.size());
  if (n == 0)
    return {};

  for (const auto &arr : arrays) {
    INS_CHECK(arr.shape().ndim() == 1, "ix_: all inputs must be 1D");
  }

  // Build broadcast shapes: each array gets shape with 1s except its own
  // dimension
  std::vector<Shape> target_shapes(n);
  for (int i = 0; i < n; ++i) {
    std::vector<int64_t> dims(n, 1);
    dims[i] = arrays[i].numel();
    target_shapes[i] = Shape(dims);
  }

  std::vector<Array> result;
  for (int i = 0; i < n; ++i) {
    Array reshaped = arrays[i].reshape(target_shapes[i]);
    result.push_back(reshaped);
  }
  return result;
}

// ============================================================================
// interp (linear interpolation)
// ============================================================================

/**
 * @brief Extract a scalar value from an array at a given index.
 *
 * @param arr Input array
 * @param idx Index
 * @return Double value
 */
static double extract_scalar(const Array &arr, int64_t idx) {
  Array scalar = arr.at(idx);
  if (scalar.dtype() == DType::F32) {
    return static_cast<double>(scalar.item<float>());
  } else if (scalar.dtype() == DType::F64) {
    return scalar.item<double>();
  } else {
    INS_THROW("extract_scalar: unsupported dtype ", dtype_name(scalar.dtype()));
  }
}

/**
 * @brief One-dimensional linear interpolation.
 *
 * @param x The x-coordinates at which to evaluate the interpolated values
 * @param xp The x-coordinates of the data points
 * @param fp The y-coordinates of the data points
 * @param left Value to return for x < xp[0] (default: fp[0])
 * @param right Value to return for x > xp[-1] (default: fp[-1])
 * @return Interpolated values
 */
Array interp(const Array &x, const Array &xp, const Array &fp,
             std::optional<double> left, std::optional<double> right) {
  INS_CHECK(x.defined() && xp.defined() && fp.defined(),
            "interp: inputs are undefined");
  INS_CHECK(xp.shape().ndim() == 1, "interp: xp must be 1D");
  INS_CHECK(fp.shape().ndim() == 1, "interp: fp must be 1D");
  INS_CHECK(xp.numel() == fp.numel(),
            "interp: xp and fp must have same length");

  // Ensure xp is sorted
  Array sort_idx = argsort(xp);
  Array sorted_xp = take(xp, sort_idx);
  Array sorted_fp = take(fp, sort_idx);

  double left_val =
      left.has_value() ? left.value() : extract_scalar(sorted_fp, 0);
  double right_val =
      right.has_value() ? right.value() : extract_scalar(sorted_fp, -1);
  double xp_min = extract_scalar(sorted_xp, 0);
  double xp_max = extract_scalar(sorted_xp, -1);

  // Find insertion indices
  Array right_idxs = searchsorted(sorted_xp, x, "right");
  Array left_idxs = sub(right_idxs, Array(1));

  // Clamp indices
  Array safe_left = maximum(left_idxs, zeros_like(left_idxs));
  Array safe_right = minimum(
      right_idxs, full(right_idxs.shape(), sorted_xp.numel() - 1, DType::I64));

  // Get values at indices
  Array x_left = take(sorted_xp, safe_left);
  Array x_right = take(sorted_xp, safe_right);
  Array y_left = take(sorted_fp, safe_left);
  Array y_right = take(sorted_fp, safe_right);

  // Linear interpolation with division by zero handling
  Array denominator = sub(x_right, x_left);
  Array numerator = sub(x, x_left);

  // Use isnan to handle division by zero
  // When denominator == 0, t will be NaN, then we replace with 0
  Array t = div(numerator, denominator);
  t = where(isnan(t), zeros_like(t), t);

  Array interp_vals = add(y_left, mul(t, sub(y_right, y_left)));

  // Handle boundaries
  Array result =
      where(less_equal(x, full(x.shape(), xp_min, interp_vals.dtype())),
            full(x.shape(), left_val, interp_vals.dtype()), interp_vals);
  result = where(greater_equal(x, full(x.shape(), xp_max, interp_vals.dtype())),
                 full(x.shape(), right_val, interp_vals.dtype()), result);

  // Replace any remaining NaN values with left_val or right_val
  // (safety net for edge cases)
  result = where(isnan(result),
                 where(less_equal(x, full(x.shape(), xp_min, result.dtype())),
                       full(x.shape(), left_val, result.dtype()),
                       full(x.shape(), right_val, result.dtype())),
                 result);

  return result;
}

} // namespace ins