// backends/cuda/kernels/indexing/unique.cu
/**
 * @file unique.cu
 * @brief CUDA kernel for unique operation.
 *
 * Returns unique elements and their indices/counts.
 * Uses CPU fallback with proper output allocation.
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* flattened input
 *   inputs[1] = bool* return_indices
 *   inputs[2] = bool* return_inverse
 *   inputs[3] = bool* return_counts
 *
 * Output layout:
 *   outputs[0] = InsightArray* unique values
 *   outputs[1..N] = optional: indices, inverse, counts
 */

#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include "insight/c_api/place.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

static void allocate_gpu_output(InsightArray *out, int64_t ndim,
                                const int64_t *dims, int64_t numel,
                                int32_t dtype) {
  int32_t elem_size = 4; // default
  switch (dtype) {
  case INSIGHT_DTYPE_BOOL:
    elem_size = 1;
    break;
  case INSIGHT_DTYPE_U8:
  case INSIGHT_DTYPE_I8:
    elem_size = 1;
    break;
  case INSIGHT_DTYPE_I16:
  case INSIGHT_DTYPE_U16:
    elem_size = 2;
    break;
  case INSIGHT_DTYPE_I32:
  case INSIGHT_DTYPE_U32:
  case INSIGHT_DTYPE_F32:
    elem_size = 4;
    break;
  case INSIGHT_DTYPE_I64:
  case INSIGHT_DTYPE_U64:
  case INSIGHT_DTYPE_F64:
    elem_size = 8;
    break;
  case INSIGHT_DTYPE_C32:
    elem_size = 8;
    break;
  case INSIGHT_DTYPE_C64:
    elem_size = 16;
    break;
  default:
    elem_size = 4;
    break;
  }

  // Save existing ref_count before memset
  int32_t *saved_ref = out->ref_count;

  size_t bytes = numel * elem_size;
  void *data = nullptr;
  if (bytes > 0) {
    cudaGetLastError(); // clear pending errors
    cudaError_t err = cudaMalloc(&data, bytes);
    if (err != cudaSuccess) {
      gpu_set_last_error(cudaGetErrorString(err));
      data = nullptr;
    }
  }

  // Preserve ref_count, update everything else
  out->data = data;
  out->ndim = ndim;
  for (int i = 0; i < ndim; ++i) {
    out->dims[i] = dims[i];
  }
  out->numel = numel;
  out->dtype = dtype;
  if (ndim > 0) {
    out->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
      out->strides[i] = out->strides[i + 1] * out->dims[i + 1];
    }
  }
  out->offset = 0;
  out->is_view = 0;
  out->device_type = INSIGHT_DEVICE_GPU;
  out->device_id = 0;
  out->ref_count = saved_ref;
  if (!out->ref_count) {
    out->ref_count = new int32_t;
    if (out->ref_count)
      *out->ref_count = 1;
  }
}

template <typename T>
static C_Status unique_impl(InsightArray *x, void **outputs,
                            bool return_indices, bool return_inverse,
                            bool return_counts) {
  int64_t n = x->numel;

  // Copy input to CPU
  T *x_data = new T[n];
  cudaMemcpy(x_data, x->data, n * sizeof(T), cudaMemcpyDeviceToHost);

  // Sort pairs (value, original_index)
  std::vector<std::pair<T, int64_t>> pairs(n);
  for (int64_t i = 0; i < n; ++i) {
    pairs[i] = {x_data[i], i};
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  // Count unique values
  int64_t unique_count = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (i == 0 || pairs[i].first != pairs[i - 1].first) {
      unique_count++;
    }
  }

  // Build inverse indices (mapping from original position to unique index)
  std::vector<int64_t> inverse(n);
  int64_t unique_idx = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (i > 0 && pairs[i].first != pairs[i - 1].first)
      unique_idx++;
    inverse[pairs[i].second] = unique_idx;
  }

  // Build first occurrence indices
  std::vector<int64_t> first_occ(unique_count);
  unique_idx = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (i == 0 || pairs[i].first != pairs[i - 1].first) {
      first_occ[unique_idx] = pairs[i].second;
      unique_idx++;
    }
  }

  // Build counts
  std::vector<int64_t> counts(unique_count, 0);
  for (int64_t i = 0; i < n; ++i) {
    counts[inverse[i]]++;
  }

  // Build sorted unique values
  T *sorted_values = new T[unique_count];
  unique_idx = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (i == 0 || pairs[i].first != pairs[i - 1].first) {
      sorted_values[unique_idx] = pairs[i].first;
      unique_idx++;
    }
  }

  // Allocate and copy outputs
  // outputs[0] = unique values
  InsightArray *out_values = static_cast<InsightArray *>(outputs[0]);
  int64_t out_dims[1] = {unique_count};
  allocate_gpu_output(out_values, 1, out_dims, unique_count, x->dtype);
  cudaMemcpy(out_values->data, sorted_values, unique_count * sizeof(T),
             cudaMemcpyHostToDevice);

  int out_idx = 1;
  if (return_indices) {
    InsightArray *out_indices = static_cast<InsightArray *>(outputs[out_idx++]);
    allocate_gpu_output(out_indices, 1, out_dims, unique_count,
                        INSIGHT_DTYPE_I64);
    cudaMemcpy(out_indices->data, first_occ.data(),
               unique_count * sizeof(int64_t), cudaMemcpyHostToDevice);
  }
  if (return_inverse) {
    InsightArray *out_inverse = static_cast<InsightArray *>(outputs[out_idx++]);
    int64_t inv_dims[1] = {n};
    allocate_gpu_output(out_inverse, 1, inv_dims, n, INSIGHT_DTYPE_I64);
    cudaMemcpy(out_inverse->data, inverse.data(), n * sizeof(int64_t),
               cudaMemcpyHostToDevice);
  }
  if (return_counts) {
    InsightArray *out_counts = static_cast<InsightArray *>(outputs[out_idx++]);
    allocate_gpu_output(out_counts, 1, out_dims, unique_count,
                        INSIGHT_DTYPE_I64);
    cudaMemcpy(out_counts->data, counts.data(), unique_count * sizeof(int64_t),
               cudaMemcpyHostToDevice);
  }

  delete[] x_data;
  delete[] sorted_values;

  return C_SUCCESS;
}

extern "C" {

C_Status unique_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = static_cast<InsightArray *>(inputs[0]);
  bool return_indices = *static_cast<bool *>(inputs[1]);
  bool return_inverse = *static_cast<bool *>(inputs[2]);
  bool return_counts = *static_cast<bool *>(inputs[3]);

  if (!x) {
    gpu_set_last_error("unique: null array pointer");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32:
    return unique_impl<float>(x, outputs, return_indices, return_inverse,
                              return_counts);
  case INSIGHT_DTYPE_F64:
    return unique_impl<double>(x, outputs, return_indices, return_inverse,
                               return_counts);
  case INSIGHT_DTYPE_I32:
    return unique_impl<int32_t>(x, outputs, return_indices, return_inverse,
                                return_counts);
  case INSIGHT_DTYPE_I64:
    return unique_impl<int64_t>(x, outputs, return_indices, return_inverse,
                                return_counts);
  default:
    gpu_set_last_error("unique: unsupported dtype");
    return C_FAILED;
  }
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(unique, INSIGHT_DTYPE_F32, unique_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(unique, INSIGHT_DTYPE_F64, unique_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(unique, INSIGHT_DTYPE_I32, unique_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(unique, INSIGHT_DTYPE_I64, unique_kernel_gpu);
