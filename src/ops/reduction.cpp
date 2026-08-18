// src/ops/reduction.cpp
/**
 * @file reduction.cpp
 * @brief Reduction operations for array processing.
 *
 * Provides various reduction functions like sum, max, min, mean, etc.
 * All reductions support axis specification, keepdim semantics, and
 * proper handling of NaN values where applicable.
 */

#include "insight/ops/reduction.h"
#include "insight/core/axis.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/manipulation.h"
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ins {

static OpSchema reduction_schema(const char *name, size_t output_count = 1) {
  std::vector<OpArgumentSchema> inputs = {
      {"x", OpArgumentKind::Array, OpArgumentAccess::ReadOnly}};
  std::vector<OpArgumentSchema> outputs;
  outputs.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    outputs.push_back({output_count == 1 ? "out" : "out" + std::to_string(i),
                       OpArgumentKind::Array, OpArgumentAccess::WriteOnly});
  }
  return OpSchema(name, OpKind::Reduction, inputs, outputs)
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

static OpSchema weighted_reduction_schema(const char *name) {
  return OpSchema(name, OpKind::Reduction,
                  {{"x", OpArgumentKind::Array, OpArgumentAccess::ReadOnly},
                   {"weights", OpArgumentKind::Array,
                    OpArgumentAccess::ReadOnly}},
                  {{"out", OpArgumentKind::Array,
                    OpArgumentAccess::WriteOnly}})
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

// ============================================================================
// ReductionInfo: Prepare reduction by moving reduction axis to last dimension
// ============================================================================

struct ReductionInfo {
  bool flatten_all = false;  ///< Reduce over all dimensions
  int64_t batch_size = 1;    ///< Product of non-reduced dimensions
  int64_t reduce_size = 1;   ///< Size of the reduction dimension
  Shape out_shape;           ///< Shape of the output array
  std::vector<int> perm;     ///< Permutation to move reduction axis to last
  std::vector<int> inv_perm; ///< Inverse permutation to restore order
  int reduced_axis = -1;     ///< Original axis index being reduced
  int ndim = 0;              ///< Number of dimensions
};

/**
 * @brief Prepare reduction information for a given input shape and axis.
 *
 * @param in_shape Input shape
 * @param axis Reduction axis (nullopt means reduce all)
 * @param keepdim Whether to keep reduced dimensions as size 1
 * @return ReductionInfo Prepared reduction information
 */
static ReductionInfo prepare_reduction(const Shape &in_shape,
                                       std::optional<int> axis, bool keepdim) {
  ReductionInfo info;
  info.ndim = in_shape.ndim();

  if (!axis.has_value()) {
    info.flatten_all = true;
    info.batch_size = 1;
    info.reduce_size = in_shape.numel();
    if (keepdim) {
      std::vector<int64_t> dims(info.ndim, 1);
      info.out_shape = Shape(dims);
    } else {
      info.out_shape = Shape({});
    }
    return info;
  }

  int ax = normalize_axis(axis.value(), info.ndim, "reduction");
  info.reduced_axis = ax;

  // Build permutation: move reduction axis to last position
  info.perm.reserve(info.ndim);
  for (int i = 0; i < info.ndim; ++i) {
    if (i != ax)
      info.perm.push_back(i);
  }
  info.perm.push_back(ax);

  // Build inverse permutation
  info.inv_perm.resize(info.ndim);
  for (int i = 0; i < info.ndim; ++i) {
    info.inv_perm[info.perm[i]] = i;
  }

  // Compute output shape
  std::vector<int64_t> out_dims;
  for (int i = 0; i < info.ndim; ++i) {
    if (i == ax) {
      if (keepdim)
        out_dims.push_back(1);
    } else {
      out_dims.push_back(in_shape.dim(i));
    }
  }
  info.out_shape = Shape(out_dims);

  // Compute batch size (product of non-reduced dimensions)
  info.batch_size = 1;
  for (int i = 0; i < info.ndim - 1; ++i) {
    info.batch_size *= in_shape.dim(info.perm[i]);
  }
  info.reduce_size = in_shape.dim(ax);

  return info;
}

/**
 * @brief Prepare input array for reduction by making it contiguous and
 * rearranging axes.
 *
 * @param x Input array
 * @param info Reduction information
 * @return Prepared array (contiguous, batch_size × reduce_size layout)
 */
static Array prepare_input(const Array &x, const ReductionInfo &info) {
  if (info.flatten_all) {
    return x.reshape(Shape({x.numel()}));
  }
  if (info.perm.empty()) {
    return x.contiguous();
  }

  // Check if permutation is identity
  bool is_identity = true;
  for (size_t i = 0; i < info.perm.size(); ++i) {
    if (info.perm[i] != static_cast<int>(i)) {
      is_identity = false;
      break;
    }
  }
  if (is_identity) {
    return x.contiguous();
  }

  Array transposed = x.transpose(info.perm);
  return transposed.contiguous();
}

/**
 * @brief Restore original axis order after reduction.
 *
 * @param result Reduced array
 * @param info Reduction information
 * @return Array with original axis order
 */
static Array transpose_output(const Array &result, const ReductionInfo &info) {
  if (info.flatten_all || info.perm.empty() || info.inv_perm.empty()) {
    return result;
  }

  int result_ndim = result.shape().ndim();
  if (result_ndim != static_cast<int>(info.inv_perm.size())) {
    return result;
  }

  // Check if inverse permutation is identity
  bool is_identity = true;
  for (int i = 0; i < result_ndim; ++i) {
    if (info.inv_perm[i] != i) {
      is_identity = false;
      break;
    }
  }
  if (is_identity) {
    return result;
  }

  Array transposed = result.transpose(info.inv_perm);
  return transposed.contiguous();
}

/**
 * @brief Apply keepdim semantics (squeeze reduced dimension if not keepdim).
 *
 * @param result Reduced array
 * @param info Reduction information
 * @param keepdim Whether to keep reduced dimensions
 * @param axis Original reduction axis
 * @return Array with proper shape
 */
static Array post_process_keepdim(const Array &result,
                                  const ReductionInfo &info, bool keepdim,
                                  std::optional<int> axis) {
  if (!keepdim && axis.has_value()) {
    int ax = info.reduced_axis;
    if (ax < result.shape().ndim() && result.shape().dim(ax) == 1) {
      return result.squeeze(ax);
    }
  }
  return result;
}

static Shape reduction_output_shape(const ReductionInfo &info, bool keepdim) {
  if (info.flatten_all && !keepdim) {
    return Shape({});
  }
  return info.out_shape;
}

static Array post_process_reduction(const Array &result,
                                    const ReductionInfo &info, bool keepdim,
                                    std::optional<int> axis) {
  Array transposed = transpose_output(result, info);
  return post_process_keepdim(transposed, info, keepdim, axis);
}

// ============================================================================
// Helper: Launch reduction kernel
// ============================================================================

/**
 * @brief Launch a reduction kernel with common parameters.
 *
 * @tparam Args Parameter pack types
 * @param kernel_name Kernel name to launch
 * @param x Input array
 * @param info Reduction information
 * @param keepdim Whether to keep reduced dimensions
 * @param axis Reduction axis
 * @param dtype Output data type (for kernels that support type conversion)
 * @param extra_args Additional kernel arguments
 * @return Reduced array
 */
template <typename... Args>
static Array launch_reduction(const char *kernel_name, const Array &x,
                              const ReductionInfo &info, bool keepdim,
                              std::optional<int> axis, DType dtype,
                              Args &&...extra_args) {

  Array prepared = prepare_input(x, info);
  Array result(reduction_output_shape(info, keepdim), dtype, x.place());
  OpSchema schema = reduction_schema(kernel_name);
  ArrayIterator iter = schema.make_reduction_iterator(
      prepared, result, info.batch_size, info.reduce_size);
  auto arrays = iter.arrays();
  int64_t batch_size = iter.reduction_batch_size();
  int64_t reduce_size = iter.reduction_reduce_size();

  std::vector<OpRegistry::Arg> inputs;
  inputs.push_back(OpRegistry::array_arg(arrays[1].layout_ptr()));
  inputs.push_back(OpRegistry::array_arg(arrays[0].layout_ptr()));
  inputs.push_back(OpRegistry::scalar_arg(&batch_size));
  inputs.push_back(OpRegistry::scalar_arg(&reduce_size));
  (inputs.push_back(OpRegistry::scalar_arg(&extra_args)), ...);

  OpRegistry::launch_schema(
      kernel_name, x.place(), x.dtype(), inputs,
      {OpRegistry::array_arg(arrays[1].layout_ptr())});

  return result;
}

template <typename... Args>
static Array launch_reduction_with_postprocess(const char *kernel_name,
                                               const Array &x,
                                               const ReductionInfo &info,
                                               bool keepdim,
                                               std::optional<int> axis,
                                               DType dtype,
                                               Args &&...extra_args) {
  Array result = launch_reduction(kernel_name, x, info, keepdim, axis, dtype,
                                  std::forward<Args>(extra_args)...);
  return post_process_reduction(result, info, keepdim, axis);
}

static std::pair<Array, Array> launch_nansum_pair(const Array &x,
                                                  const ReductionInfo &info,
                                                  bool keepdim,
                                                  std::optional<int> axis) {
  Array prepared = prepare_input(x, info);
  Shape out_shape = reduction_output_shape(info, keepdim);
  Array sum_out(out_shape, x.dtype(), x.place());
  Array count_out(out_shape, DType::I64, x.place());
  OpSchema schema = reduction_schema("nansum", 2);
  ArrayIterator iter = schema.make_reduction_iterator(
      prepared, {sum_out, count_out}, info.batch_size, info.reduce_size);
  auto arrays = iter.arrays();
  int64_t batch_size = iter.reduction_batch_size();
  int64_t reduce_size = iter.reduction_reduce_size();

  OpRegistry::launch_schema(
      "nansum", x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[2].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::scalar_arg(&batch_size),
       OpRegistry::scalar_arg(&reduce_size)},
      {OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[2].layout_ptr())});

  return {post_process_reduction(sum_out, info, keepdim, axis),
          post_process_reduction(count_out, info, keepdim, axis)};
}

// ============================================================================
// sum: Sum of array elements
// ============================================================================

Array sum(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction("sum", x, info, keepdim, axis, x.dtype());
}

// ============================================================================
// mean: Arithmetic mean
// ============================================================================

Array mean(const Array &x, std::optional<int> axis, bool keepdim) {
  Array s = sum(x, axis, true);

  int ax = axis.has_value() ? normalize_axis(axis.value(), x.shape().ndim(), "mean") : -1;
  int64_t n = axis.has_value() ? x.shape().dim(ax) : x.numel();
  double inv_n = 1.0 / static_cast<double>(n);

  DType out_dtype = is_integer(x.dtype()) ? DType::F64 : x.dtype();
  Array s_cast = (s.dtype() != out_dtype) ? s.to(out_dtype) : s;
  Array divisor = full(s_cast.shape(), inv_n, out_dtype, s_cast.place());
  Array result = mul(s_cast, divisor);

  if (!keepdim && axis.has_value() && result.shape().ndim() > 0 &&
      result.shape().dim(result.shape().ndim() - 1) == 1) {
    result = result.squeeze(result.shape().ndim() - 1);
  }

  return result;
}

// ============================================================================
// max: Maximum value
// ============================================================================

Array max(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction("max", x, info, keepdim, axis, x.dtype());
}

// ============================================================================
// min: Minimum value
// ============================================================================

Array min(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction("min", x, info, keepdim, axis, x.dtype());
}

// ============================================================================
// prod: Product of elements
// ============================================================================

Array prod(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction("prod", x, info, keepdim, axis, x.dtype());
}

// ============================================================================
// any: Logical OR (True if any element is non-zero)
// ============================================================================

Array any(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction("any", x, info, keepdim, axis, DType::BOOL);
}

// ============================================================================
// all: Logical AND (True if all elements are non-zero)
// ============================================================================

Array all(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction("all", x, info, keepdim, axis, DType::BOOL);
}

// ============================================================================
// argmax: Index of maximum value
// ============================================================================

Array argmax(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction("argmax", x, info, keepdim, axis, DType::I64);
}

// ============================================================================
// argmin: Index of minimum value
// ============================================================================

Array argmin(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction("argmin", x, info, keepdim, axis, DType::I64);
}

// ============================================================================
// count_nonzero: Count of non-zero elements
// ============================================================================

Array count_nonzero(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction("count_nonzero", x, info, keepdim, axis, DType::I64);
}

// ============================================================================
// var: Variance
// ============================================================================

Array var(const Array &x, std::optional<int> axis, bool keepdim, int ddof) {
  if (!axis.has_value()) {
    Array flat = x.reshape(Shape({x.numel()}));
    return var(flat, 0, keepdim, ddof);
  }

  int ax = normalize_axis(axis.value(), x.shape().ndim(), "var");
  int64_t n = x.shape().dim(ax);

  Array m = mean(x, ax, true);
  Array centered = sub(x, m);
  Array squared = square(centered);
  Array sum_sq = sum(squared, ax, keepdim);

  double divisor = static_cast<double>(n - ddof);
  if (divisor <= 0.0)
    divisor = 1.0;
  Array div_arr = full(sum_sq.shape(), divisor, sum_sq.dtype(), sum_sq.place());

  return div(sum_sq, div_arr);
}

// ============================================================================
// std: Standard deviation
// ============================================================================

Array std(const Array &x, std::optional<int> axis, bool keepdim, int ddof) {
  Array v = var(x, axis, keepdim, ddof);
  return sqrt(v);
}

// ============================================================================
// sem: Standard error of the mean
// ============================================================================

Array sem(const Array &x, std::optional<int> axis, bool keepdim, int ddof) {
  Array s = std(x, axis, true, ddof);
  int ax = axis.has_value() ? normalize_axis(axis.value(), x.shape().ndim(), "sem") : -1;
  int64_t n = axis.has_value() ? x.shape().dim(ax) : x.numel();
  double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n));
  Array factor = full(s.shape(), inv_sqrt_n, s.dtype(), s.place());
  Array result = mul(s, factor);

  if (!keepdim && axis.has_value() && ax < result.shape().ndim() &&
      result.shape().dim(ax) == 1) {
    result = result.squeeze(ax);
  }
  return result;
}

static Array launch_cumulative(const char *kernel_name, const Array &x,
                               int axis) {
  int ax = normalize_axis(axis, x.shape().ndim(), kernel_name);

  Array result(x.shape(), x.dtype(), x.place());
  OpSchema schema = reduction_schema(kernel_name);
  ArrayIterator iter = schema.make_reduction_iterator(x, result, x.numel(), 1);
  auto arrays = iter.arrays();

  OpRegistry::launch_schema(
      kernel_name, x.place(), x.dtype(),
      {OpRegistry::array_arg(arrays[1].layout_ptr()),
       OpRegistry::array_arg(arrays[0].layout_ptr()),
       OpRegistry::scalar_arg(&ax)},
      {OpRegistry::array_arg(arrays[1].layout_ptr())});

  return result;
}

// ============================================================================
// cumsum: Cumulative sum
// ============================================================================

Array cumsum(const Array &x, int axis) { return launch_cumulative("cumsum", x, axis); }

// ============================================================================
// cumprod: Cumulative product
// ============================================================================

Array cumprod(const Array &x, int axis) {
  return launch_cumulative("cumprod", x, axis);
}

// ============================================================================
// cummax: Cumulative maximum
// ============================================================================

Array cummax(const Array &x, int axis) { return launch_cumulative("cummax", x, axis); }

// ============================================================================
// cummin: Cumulative minimum
// ============================================================================

Array cummin(const Array &x, int axis) { return launch_cumulative("cummin", x, axis); }

// ============================================================================
// median: Median (50th percentile)
// ============================================================================

Array median(const Array &x, std::optional<int> axis, bool keepdim) {
  return quantile(x, 0.5, axis, keepdim);
}

// ============================================================================
// quantile: Quantile (single value)
// ============================================================================

Array quantile(const Array &x, double q, std::optional<int> axis,
               bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction_with_postprocess("quantile", x, info, keepdim, axis,
                                           DType::F64, q);
}

// ============================================================================
// quantile: Quantile (array of quantiles)
// ============================================================================

Array quantile(const Array &x, const Array &q, std::optional<int> axis,
               bool keepdim) {
  int64_t n_q = q.numel();
  if (n_q == 1) {
    double q_val = (q.dtype() == DType::F32)
                       ? static_cast<double>(q.item<float>())
                       : q.item<double>();
    return quantile(x, q_val, axis, keepdim);
  }

  std::vector<Array> results;
  for (int64_t i = 0; i < n_q; ++i) {
    Array q_i = q.slice(0, i, i + 1);
    double q_val = (q.dtype() == DType::F32)
                       ? static_cast<double>(q_i.item<float>())
                       : q_i.item<double>();
    results.push_back(quantile(x, q_val, axis, keepdim));
  }
  return stack(results, 0);
}

// ============================================================================
// nansum: Sum ignoring NaN
// ============================================================================

Array nansum(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_nansum_pair(x, info, keepdim, axis).first;
}

// ============================================================================
// nanmean: Mean ignoring NaN
// ============================================================================

Array nanmean(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  auto [sum_out, count_out] = launch_nansum_pair(x, info, keepdim, axis);
  Array count_float = count_out.to(x.dtype());
  return div(sum_out, count_float);
}

// ============================================================================
// nanmax: Maximum ignoring NaN
// ============================================================================

Array nanmax(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction("nanmax", x, info, keepdim, axis, x.dtype());
}

// ============================================================================
// nanmin: Minimum ignoring NaN
// ============================================================================

Array nanmin(const Array &x, std::optional<int> axis, bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction("nanmin", x, info, keepdim, axis, x.dtype());
}

// ============================================================================
// nanvar: Variance ignoring NaN
// ============================================================================

Array nanvar(const Array &x, std::optional<int> axis, bool keepdim, int ddof) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction_with_postprocess("nanvar", x, info, keepdim, axis,
                                           x.dtype(), ddof);
}

// ============================================================================
// nanstd: Standard deviation ignoring NaN
// ============================================================================

Array nanstd(const Array &x, std::optional<int> axis, bool keepdim, int ddof) {
  Array v = nanvar(x, axis, keepdim, ddof);
  return sqrt(v);
}

// ============================================================================
// nanmedian: Median ignoring NaN (50th percentile)
// ============================================================================

Array nanmedian(const Array &x, std::optional<int> axis, bool keepdim) {
  return nanquantile(x, 0.5, axis, keepdim);
}

// ============================================================================
// nanquantile: Quantile ignoring NaN (single value)
// ============================================================================

Array nanquantile(const Array &x, double q, std::optional<int> axis,
                  bool keepdim) {
  ReductionInfo info = prepare_reduction(x.shape(), axis, keepdim);
  return launch_reduction_with_postprocess("nanquantile", x, info, keepdim,
                                           axis, x.dtype(), q);
}

// ============================================================================
// nanquantile: Quantile ignoring NaN (array of quantiles)
// ============================================================================

Array nanquantile(const Array &x, const Array &q, std::optional<int> axis,
                  bool keepdim) {
  int64_t n_q = q.numel();
  if (n_q == 1) {
    double q_val = (q.dtype() == DType::F32)
                       ? static_cast<double>(q.item<float>())
                       : q.item<double>();
    return nanquantile(x, q_val, axis, keepdim);
  }

  std::vector<Array> results;
  for (int64_t i = 0; i < n_q; ++i) {
    Array q_i = q.slice(0, i, i + 1);
    double q_val = (q.dtype() == DType::F32)
                       ? static_cast<double>(q_i.item<float>())
                       : q_i.item<double>();
    results.push_back(nanquantile(x, q_val, axis, keepdim));
  }
  return stack(results, 0);
}

// ============================================================================
// percentile: Percentile (quantile * 100)
// ============================================================================

Array percentile(const Array &x, double q, std::optional<int> axis,
                 bool keepdim) {
  return quantile(x, q / 100.0, axis, keepdim);
}

// ============================================================================
// bincount: Count occurrences of each integer value
// ============================================================================

Array bincount(const Array &x, std::optional<Array> weights,
               int64_t minlength) {
  INS_CHECK(x.shape().ndim() == 1, "bincount: x must be 1D, got shape ",
            x.shape());
  INS_CHECK(x.dtype() == DType::I32 || x.dtype() == DType::I64,
            "bincount: x must be int32 or int64, got ", dtype_name(x.dtype()));
  INS_CHECK(minlength >= 0, "bincount: minlength must be >= 0, got ",
            minlength);

  // Compute output length
  int64_t max_val = 0;
  if (x.numel() > 0) {
    Array m = max(x);
    max_val = (x.dtype() == DType::I32)
                  ? static_cast<int64_t>(m.item<int32_t>())
                  : m.item<int64_t>();
  }
  int64_t out_len = std::max(max_val + 1, minlength);

  if (weights.has_value()) {
    const Array &w = *weights;
    INS_CHECK(w.numel() == x.numel(), "bincount: weights length ", w.numel(),
              " must match x length ", x.numel());
    INS_CHECK(w.shape().ndim() == x.shape().ndim(),
              "bincount: weights must have same ndim as x");

    DType out_dtype = w.dtype();
    INS_CHECK(
        out_dtype == DType::F32 || out_dtype == DType::F64 ||
            out_dtype == DType::I32 || out_dtype == DType::I64,
        "bincount: weights must be int32, int64, float32, or float64, got ",
        dtype_name(out_dtype));

    Array result(Shape({out_len}), out_dtype, x.place());
    result = zeros_like(result);

    OpSchema schema = weighted_reduction_schema("bincount_weighted");
    ArrayIterator iter = schema.make_reduction_iterator(
        {x, w}, {result}, x.numel(), 1);
    auto arrays = iter.arrays();
    OpRegistry::launch_schema(
        "bincount_weighted", x.place(), x.dtype(),
        {OpRegistry::array_arg(arrays[2].layout_ptr()),
         OpRegistry::array_arg(arrays[0].layout_ptr()),
         OpRegistry::array_arg(arrays[1].layout_ptr())},
        {OpRegistry::array_arg(arrays[2].layout_ptr())});

    return result;
  } else {
    Array result(Shape({out_len}), DType::I64, x.place());
    result = zeros_like(result);

    OpSchema schema = reduction_schema("bincount");
    ArrayIterator iter = schema.make_reduction_iterator(x, result, x.numel(), 1);
    auto arrays = iter.arrays();
    OpRegistry::launch_schema(
        "bincount", x.place(), x.dtype(),
        {OpRegistry::array_arg(arrays[1].layout_ptr()),
         OpRegistry::array_arg(arrays[0].layout_ptr())},
        {OpRegistry::array_arg(arrays[1].layout_ptr())});

    return result;
  }
}

} // namespace ins