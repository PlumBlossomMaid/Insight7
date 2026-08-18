// src/ops/manipulation.cpp
/**
 * @file manipulation.cpp
 * @brief Array manipulation operations.
 *
 * Provides shape manipulation, transposition, joining, splitting,
 * tiling, padding, rolling, and other array transformation operations.
 */

#include "insight/ops/manipulation.h"
#include "insight/core/axis.h"
#include "insight/core/exception.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"

namespace ins {

// ============================================================================
// Helper: launch manipulation kernel
// ============================================================================

static void add_kernel_arg(std::vector<OpRegistry::Arg> &inputs,
                           std::nullptr_t) {
  inputs.push_back({nullptr, OpRegistry::ArgKind::HostScalar});
}

template <typename T>
static void add_kernel_arg(std::vector<OpRegistry::Arg> &inputs, T *arg) {
  inputs.push_back(OpRegistry::scalar_arg(arg));
}

template <typename T>
static void add_kernel_arg(std::vector<OpRegistry::Arg> &inputs,
                           const T &arg) {
  inputs.push_back({const_cast<void *>(static_cast<const void *>(&arg)),
                    OpRegistry::ArgKind::HostScalar});
}

static OpSchema manipulation_schema(const char *name, size_t input_count = 1) {
  std::vector<OpArgumentSchema> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    inputs.push_back({input_count == 1 ? "x" : "x" + std::to_string(i),
                      OpArgumentKind::Array, OpArgumentAccess::ReadOnly});
  }
  return OpSchema(name, OpKind::Manipulation, inputs,
                  {{"out", OpArgumentKind::Array,
                    OpArgumentAccess::WriteOnly}})
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

template <typename... Args>
static void launch_manipulation_kernel(const char *kernel_name, const Array &x,
                                       Array &result, Args &&...extra_args) {
  OpSchema schema = manipulation_schema(kernel_name);
  ArrayIterator iter = schema.make_transform_iterator({x}, {result});
  auto arrays = iter.arrays();

  std::vector<OpRegistry::Arg> inputs;
  inputs.push_back(OpRegistry::array_arg(arrays[1].layout_ptr()));
  inputs.push_back(OpRegistry::array_arg(arrays[0].layout_ptr()));

  (add_kernel_arg(inputs, extra_args), ...);

  OpRegistry::launch_schema(
      kernel_name, x.place(), x.dtype(), inputs,
      {OpRegistry::array_arg(arrays[1].layout_ptr())});
}

// ============================================================================
// Shape manipulation (view operations, no kernel needed)
// ============================================================================

Array reshape(const Array &x, const Shape &new_shape) {
  return x.reshape(new_shape);
}

Array flatten(const Array &x) { return x.reshape(Shape({x.numel()})); }

Array ravel(const Array &x) {
  if (x.is_contiguous()) {
    return flatten(x);
  } else {
    return x.copy().reshape(Shape({x.numel()}));
  }
}

Array squeeze(const Array &x) { return x.squeeze(); }

Array squeeze(const Array &x, int axis) { return x.squeeze(axis); }

Array unsqueeze(const Array &x, int axis) { return x.unsqueeze(axis); }

// ============================================================================
// Transpose (view operations, no kernel needed)
// ============================================================================

Array transpose(const Array &x) { return x.transpose(); }

Array permute(const Array &x, const std::vector<int> &axes) {
  return x.transpose(axes);
}

Array swapaxes(const Array &x, int axis1, int axis2) {
  Shape shape = x.shape();
  int ndim = shape.ndim();
  axis1 = normalize_axis(axis1, ndim, "swapaxes(axis1)");
  axis2 = normalize_axis(axis2, ndim, "swapaxes(axis2)");

  std::vector<int> perm(ndim);
  for (int i = 0; i < ndim; ++i)
    perm[i] = i;
  std::swap(perm[axis1], perm[axis2]);
  return x.transpose(perm);
}

Array moveaxis(const Array &x, int source, int destination) {
  Shape shape = x.shape();
  int ndim = shape.ndim();
  source = normalize_axis(source, ndim, "moveaxis(source)");
  destination = normalize_axis(destination, ndim, "moveaxis(destination)");

  if (source == destination)
    return x;

  std::vector<int> perm;
  for (int i = 0; i < ndim; ++i) {
    if (i != source)
      perm.push_back(i);
  }
  perm.insert(perm.begin() + destination, source);
  return x.transpose(perm);
}

// ============================================================================
// Flipping and rotating
// ============================================================================

Array flip(const Array &x, std::optional<int> axis) {
  if (!axis.has_value()) {
    // Flip all axes
    Array result = x;
    for (int i = 0; i < x.shape().ndim(); ++i) {
      result = flip(result, i);
    }
    return result;
  }

  int ax = normalize_axis(axis.value(), x.shape().ndim(), "flip");

  Array result(x.shape(), x.dtype(), x.place());

  launch_manipulation_kernel("flip", x, result, ax);

  return result;
}

Array rot90(const Array &x, int k, const std::vector<int> &axes) {
  INS_CHECK(x.shape().ndim() >= 2, "rot90: requires at least 2 dimensions");
  INS_CHECK(axes.size() == 2, "rot90: axes must have exactly 2 elements");

  int ndim = x.shape().ndim();
  int axis1 = normalize_axis(axes[0], ndim, "rot90(axis1)");
  int axis2 = normalize_axis(axes[1], ndim, "rot90(axis2)");
  INS_CHECK(axis1 != axis2, "rot90: axes must be different");

  // Normalize k to [0, 3]
  int k_mod = ((k % 4) + 4) % 4;
  if (k_mod == 0)
    return x.copy();

  // Build permutation that swaps axis1 and axis2
  std::vector<int> perm(ndim);
  for (int i = 0; i < ndim; ++i)
    perm[i] = i;
  perm[axis1] = axis2;
  perm[axis2] = axis1;

  Array result = x;
  for (int i = 0; i < k_mod; ++i) {
    // One 90-degree counter-clockwise rotation: transpose then flip
    result = result.transpose(perm);
    result = flip(result, axis1); // axis1 is the new position of axis2
  }

  return result;
}

// ============================================================================
// Joining
// ============================================================================

Array concat(const std::vector<Array> &input_arrays, int axis) {
  if (input_arrays.empty())
    INS_THROW("concat: no arrays provided");
  if (input_arrays.size() == 1)
    return input_arrays[0];

  const Shape &first_shape = input_arrays[0].shape();
  int ndim = first_shape.ndim();
  int ax = normalize_axis(axis, ndim, "concat");

  // Check compatibility and compute output shape
  DType dtype = input_arrays[0].dtype();
  Place place = input_arrays[0].place();
  std::vector<int64_t> out_dims = first_shape.dims();
  int64_t concat_size = 0;

  for (const auto &array : input_arrays) {
    INS_CHECK(array.dtype() == dtype, "concat: dtype mismatch");
    INS_CHECK(array.place() == place, "concat: device mismatch");
    INS_CHECK(array.shape().ndim() == ndim, "concat: dimension mismatch");
    for (int i = 0; i < ndim; ++i) {
      if (i != ax) {
        INS_CHECK(array.shape().dim(i) == out_dims[i],
                  "concat: shape mismatch at dimension ", i);
      }
    }
    concat_size += array.shape().dim(ax);
  }
  out_dims[ax] = concat_size;
  Shape out_shape(out_dims);

  Array result(out_shape, dtype, place);
  OpSchema schema = manipulation_schema("concat", input_arrays.size());
  ArrayIterator iter = schema.make_transform_iterator(input_arrays, {result});
  auto iter_arrays = iter.arrays();

  std::vector<OpRegistry::Arg> inputs;
  inputs.push_back(
      OpRegistry::array_arg(iter_arrays[input_arrays.size()].layout_ptr()));

  int num_arrays = static_cast<int>(input_arrays.size());
  inputs.push_back(OpRegistry::scalar_arg(&num_arrays));

  for (size_t i = 0; i < input_arrays.size(); ++i) {
    inputs.push_back(OpRegistry::array_arg(iter_arrays[i].layout_ptr()));
  }

  inputs.push_back(OpRegistry::scalar_arg(&ax));

  OpRegistry::launch_schema(
      "concat", place, dtype, inputs,
      {OpRegistry::array_arg(iter_arrays[input_arrays.size()].layout_ptr())});

  return result;
}

Array stack(const std::vector<Array> &input_arrays, int axis) {
  if (input_arrays.empty())
    INS_THROW("stack: no arrays provided");

  const Shape &first_shape = input_arrays[0].shape();
  int ndim = first_shape.ndim();
  int ax = normalize_axis_insert(axis, ndim, "stack");

  // Check all arrays have same shape and dtype
  DType dtype = input_arrays[0].dtype();
  Place place = input_arrays[0].place();
  for (const auto &array : input_arrays) {
    INS_CHECK(array.shape() == first_shape, "stack: shape mismatch");
    INS_CHECK(array.dtype() == dtype, "stack: dtype mismatch");
    INS_CHECK(array.place() == place, "stack: device mismatch");
  }

  // First unsqueeze each array, then concat
  std::vector<Array> expanded;
  expanded.reserve(input_arrays.size());
  for (const auto &array : input_arrays) {
    expanded.push_back(unsqueeze(array, ax));
  }
  return concat(expanded, ax);
}

// ============================================================================
// Splitting (view operations, no kernel needed)
// ============================================================================

std::vector<Array> split(const Array &x, int indices_or_sections, int axis) {
  Shape shape = x.shape();
  int ndim = shape.ndim();
  int ax = normalize_axis(axis, ndim, "split");

  int64_t dim_size = shape.dim(ax);
  if (dim_size % indices_or_sections != 0) {
    INS_THROW("split: axis dimension must be divisible by number of splits");
  }

  int64_t split_size = dim_size / indices_or_sections;
  std::vector<int64_t> indices;
  for (int64_t i = split_size; i < dim_size; i += split_size) {
    indices.push_back(i);
  }
  return split(x, indices, ax);
}

std::vector<Array> split(const Array &x, const std::vector<int64_t> &indices,
                         int axis) {
  Shape shape = x.shape();
  int ndim = shape.ndim();
  int ax = normalize_axis(axis, ndim, "split");

  std::vector<Array> result;
  int64_t start = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    int64_t end = indices[i];
    std::vector<Slice> slices(ndim, Slice::all());
    slices[ax] = Slice(start, end);
    result.push_back(x.slice(slices));
    start = end;
  }

  std::vector<Slice> slices(ndim, Slice::all());
  slices[ax] = Slice(start, shape.dim(ax));
  result.push_back(x.slice(slices));

  return result;
}

// ============================================================================
// Tiling and Repeating
// ============================================================================

Array repeat(const Array &x, int repeats, std::optional<int> axis) {
  if (!axis.has_value()) {
    Array flat = ravel(x);
    return repeat(flat, repeats, 0);
  }

  int ax = normalize_axis(axis.value(), x.shape().ndim(), "repeat");
  INS_CHECK(repeats >= 0, "repeat: repeats must be non-negative");

  // Compute output shape
  std::vector<int64_t> out_dims = x.shape().dims();
  out_dims[ax] *= repeats;
  Shape out_shape(out_dims);

  Array result(out_shape, x.dtype(), x.place());

  launch_manipulation_kernel("repeat", x, result, repeats, ax);

  return result;
}

Array tile(const Array &x, const Shape &reps) {
  // Compute output shape
  Shape in_shape = x.shape();
  int in_ndim = in_shape.ndim();
  int out_ndim = std::max(in_ndim, reps.ndim());

  std::vector<int64_t> out_dims(out_ndim, 1);
  for (int i = 0; i < out_ndim; ++i) {
    int in_idx = i - (out_ndim - in_ndim);
    int64_t in_dim = (in_idx >= 0) ? in_shape.dim(in_idx) : 1;
    int64_t rep = (i < reps.ndim()) ? reps.dim(i) : 1;
    out_dims[i] = in_dim * rep;
  }
  Shape out_shape(out_dims);

  Array result(out_shape, x.dtype(), x.place());

  launch_manipulation_kernel("tile", x, result, reps);

  return result;
}

// ============================================================================
// Padding
// ============================================================================

Array pad(const Array &x, const std::vector<int64_t> &pad_width,
          double constant_value) {
  Shape in_shape = x.shape();
  int ndim = in_shape.ndim();
  INS_CHECK(pad_width.size() == static_cast<size_t>(2 * ndim),
            "pad: pad_width size mismatch");

  // Compute output shape
  std::vector<int64_t> out_dims(ndim);
  for (int i = 0; i < ndim; ++i) {
    out_dims[i] = in_shape.dim(i) + pad_width[2 * i] + pad_width[2 * i + 1];
  }
  Shape out_shape(out_dims);

  Array result(out_shape, x.dtype(), x.place());

  int64_t *pad_width_ptr = const_cast<int64_t *>(pad_width.data());
  launch_manipulation_kernel("pad", x, result, pad_width_ptr, constant_value);

  return result;
}

// ============================================================================
// Rolling
// ============================================================================

Array roll(const Array &x, int shift, std::optional<int> axis) {
  int ax = axis.has_value() ? normalize_axis(axis.value(), x.shape().ndim(), "roll") : -1;
  Array result(x.shape(), x.dtype(), x.place());

  launch_manipulation_kernel("roll", x, result, shift, ax);

  return result;
}

// ============================================================================
// Diagonal
// ============================================================================

Array diag(const Array &x, int k) {
  const Shape &shape = x.shape();
  Array result;

  if (shape.ndim() == 1) {
    // Construct diagonal matrix from 1D array
    int64_t n = x.numel();
    int64_t size = n + std::abs(k);
    Shape out_shape({size, size});
    result = Array(out_shape, x.dtype(), x.place());
  } else if (shape.ndim() == 2) {
    // Extract diagonal from 2D array
    int64_t rows = shape.dim(0);
    int64_t cols = shape.dim(1);
    int64_t diag_len;
    if (k >= 0) {
      diag_len = std::min(rows, cols - k);
    } else {
      diag_len = std::min(rows + k, cols);
    }
    Shape out_shape({diag_len});
    result = Array(out_shape, x.dtype(), x.place());
  } else {
    INS_THROW("diag: input must be 1D or 2D");
  }

  launch_manipulation_kernel("diag", x, result, k);

  return result;
}

Array diagonal(const Array &x, int offset, int axis1, int axis2) {
  // For 2D arrays, this is similar to diag
  return diag(x, offset);
}

// ============================================================================
// Triangular
// ============================================================================

Array tril(const Array &x, int k) {
  Array result(x.shape(), x.dtype(), x.place());
  launch_manipulation_kernel("tril", x, result, k);
  return result;
}

Array triu(const Array &x, int k) {
  Array result(x.shape(), x.dtype(), x.place());
  launch_manipulation_kernel("triu", x, result, k);
  return result;
}

// ============================================================================
// Slicing (Views) - Direct Array methods
// ============================================================================

Array slice(Array &x, int dim, int64_t start, int64_t stop, int64_t step) {
  return x.slice(dim, start, stop, step);
}

Array slice(Array &x, const std::vector<Slice> &slices) {
  return x.slice(slices);
}

// ============================================================================
// diff
// ============================================================================

Array diff(const Array &x, int n, int axis) {
  INS_CHECK(x.defined(), "diff: input is undefined");
  INS_CHECK(n >= 0, "diff: n must be non-negative");

  if (n == 0) {
    return x.copy();
  }

  int ndim = x.shape().ndim();
  int ax = normalize_axis(axis, ndim, "diff");

  Array result = x;
  for (int i = 0; i < n; ++i) {
    int64_t axis_size = result.shape().dim(ax);
    INS_CHECK(axis_size > 1, "diff: axis size must be at least 2");

    std::vector<Slice> slices_front(ndim, Slice::all());
    std::vector<Slice> slices_back(ndim, Slice::all());
    slices_front[ax] = Slice(0, axis_size - 1);
    slices_back[ax] = Slice(1, axis_size);

    Array front = result.slice(slices_front);
    Array back = result.slice(slices_back);

    result = sub(back, front);
  }

  return result;
}

// ============================================================================
// contiguous
// ============================================================================

Array contiguous(const Array &x) {
  if (x.is_contiguous())
    return x;

  Array result(x.shape(), x.dtype(), x.place());
  OpSchema schema = manipulation_schema("contiguous_copy");
  ArrayIterator iter = schema.make_transform_iterator({x}, {result});
  auto arrays = iter.arrays();
  OpRegistry::launch_schema(
      "contiguous_copy", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[0].layout_ptr())},
      {OpRegistry::array_arg(arrays[1].layout_ptr())});

  return result;
}

} // namespace ins