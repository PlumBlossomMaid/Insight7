// src/ops/signal/peak_finding.cpp
#include "insight/ops/signal/peak_finding.h"
#include "insight/core/axis.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/indexing.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/operator.h"
#include "insight/ops/reduction.h"
#include "insight/ops/unary.h"

namespace ins {
namespace signal {

namespace {

// Build shifted arrays with clip-mode boundary handling
// Returns arrays of "right neighbor" and "left neighbor" values
// For clip mode: boundary values repeat the edge element
void build_neighbors(const Array &data, int axis, int64_t shift,
                     Array &right_out, Array &left_out) {
  int ndim = data.shape().ndim();
  int64_t datalen = data.shape().dim(axis);

  // Right neighbor: data[min(i+shift, n-1)]
  // Left neighbor: data[max(i-shift, 0)]
  // Use roll + edge padding for clip mode

  // For right: shift data right by `shift`, pad left edge with first element
  // For left: shift data left by `shift`, pad right edge with last element

  // Simpler approach: build using index gathering
  // Create index arrays for right and left neighbors
  Place cpu = CPUPlace();
  Array data_cpu =
      (data.place().kind() == DeviceKind::CPU) ? data : data.to(cpu);

  // Build right indices: min(i+shift, n-1) for each i
  std::vector<int64_t> right_idx(datalen), left_idx(datalen);
  for (int64_t i = 0; i < datalen; ++i) {
    right_idx[i] = std::min(i + shift, datalen - 1);
    left_idx[i] = std::max(i - shift, (int64_t)0);
  }

  // Flatten data for gathering
  Array flat = reshape(data_cpu, {data_cpu.numel()});

  // Compute axis stride
  int64_t axis_stride = 1;
  for (int d = axis + 1; d < ndim; ++d) {
    axis_stride *= data.shape().dim(d);
  }
  int64_t outer_count = data_cpu.numel() / datalen;

  // For each element, compute the flat index of its right and left neighbors
  int64_t total = data_cpu.numel();
  std::vector<int64_t> right_flat_idx(total), left_flat_idx(total);

  for (int64_t idx = 0; idx < total; ++idx) {
    int64_t axis_pos = (idx / axis_stride) % datalen;
    int64_t outer_base = idx - (axis_pos * axis_stride);

    int64_t r_pos = std::min(axis_pos + shift, datalen - 1);
    int64_t l_pos = std::max(axis_pos - shift, (int64_t)0);

    right_flat_idx[idx] = outer_base + r_pos * axis_stride;
    left_flat_idx[idx] = outer_base + l_pos * axis_stride;
  }

  Array right_idx_arr = to_array(right_flat_idx, DType::I64, cpu);
  Array left_idx_arr = to_array(left_flat_idx, DType::I64, cpu);

  right_out = take(flat, right_idx_arr);
  right_out = reshape(right_out, data.shape());

  left_out = take(flat, left_idx_arr);
  left_out = reshape(left_out, data.shape());

  // Transfer back to original device
  if (data.place().kind() != DeviceKind::CPU) {
    right_out = right_out.to(data.place());
    left_out = left_out.to(data.place());
  }
}

Array boolrelextrema(const Array &data, const std::string &comparator, int axis,
                     int order, const std::string &mode) {
  int axis_norm = normalize_axis(axis, data.shape().ndim(), "boolrelextrema");

  // Start with all true
  Array results = ones(data.shape(), DType::BOOL, data.place());

  for (int64_t shift = 1; shift <= order; ++shift) {
    // Build neighbor arrays with clip-mode boundary handling
    Array right, left;
    build_neighbors(data, axis_norm, shift, right, left);

    // Compare center with both neighbors
    Array cmp_right, cmp_left;
    if (comparator == "greater") {
      cmp_right = greater(data, right);
      cmp_left = greater(data, left);
    } else {
      cmp_right = less(data, right);
      cmp_left = less(data, left);
    }

    // Both must be true
    Array both = logical_and(cmp_right, cmp_left);

    // Update results: element is extremum only if it was extremum for all
    // previous shifts AND this one
    results = logical_and(results, both);
  }

  return results;
}

} // anonymous namespace

std::vector<Array> argrelextrema(const Array &data,
                                 const std::string &comparator, int axis,
                                 int order, const std::string &mode) {
  INS_CHECK(data.defined(), "argrelextrema: input is undefined");
  INS_CHECK(order >= 1, "argrelextrema: order must be >= 1");
  INS_CHECK(comparator == "greater" || comparator == "less",
            "argrelextrema: comparator must be 'greater' or 'less'");

  Array mask = boolrelextrema(data, comparator, axis, order, mode);

  // Use nonzero() to find flat indices where mask is true
  Place cpu = CPUPlace();
  Array mask_cpu =
      (mask.place().kind() == DeviceKind::CPU) ? mask : mask.to(cpu);

  Array flat_idx = nonzero(mask_cpu);

  // Convert flat indices to multi-dimensional indices
  int ndim = data.shape().ndim();
  axis = normalize_axis(axis, ndim, "argrelextrema");

  int64_t axis_stride = 1;
  for (int d = axis + 1; d < ndim; ++d) {
    axis_stride *= data.shape().dim(d);
  }

  // axis_idx = (flat_idx / axis_stride) % datalen
  Array axis_stride_arr = full(flat_idx.shape(), axis_stride, DType::I64, cpu);
  Array datalen_arr =
      full(flat_idx.shape(), data.shape().dim(axis), DType::I64, cpu);
  Array axis_idx = mod(div(flat_idx, axis_stride_arr), datalen_arr);

  std::vector<Array> result;
  result.push_back(axis_idx);

  // For multi-dimensional, also compute indices for other dimensions
  for (int d = 0; d < ndim; ++d) {
    if (d == axis)
      continue;
    int64_t d_stride = 1;
    for (int dd = d + 1; dd < ndim; ++dd) {
      d_stride *= data.shape().dim(dd);
    }
    Array d_stride_arr = full(flat_idx.shape(), d_stride, DType::I64, cpu);
    Array d_dim_arr =
        full(flat_idx.shape(), data.shape().dim(d), DType::I64, cpu);
    Array d_idx = mod(div(flat_idx, d_stride_arr), d_dim_arr);
    result.push_back(d_idx);
  }

  // Transfer back to original place if needed
  if (data.place().kind() != DeviceKind::CPU) {
    for (auto &arr : result) {
      arr = arr.to(data.place());
    }
  }

  return result;
}

std::vector<Array> argrelmax(const Array &data, int axis, int order,
                             const std::string &mode) {
  return argrelextrema(data, "greater", axis, order, mode);
}

std::vector<Array> argrelmin(const Array &data, int axis, int order,
                             const std::string &mode) {
  return argrelextrema(data, "less", axis, order, mode);
}

} // namespace signal
} // namespace ins
