// backends/cuda/kernels/indexing/argsort.cu
/**
 * @file argsort.cu
 * @brief CUDA kernel for argsort operation.
 *
 * Returns indices that would sort the array.
 * Uses CPU fallback for sorting.
 *
 * Input layout:
 *   inputs[0] = InsightArray* output (indices, int64)
 *   inputs[1] = InsightArray* input (already transposed so axis is last)
 *   inputs[2] = bool* descending
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <utility>

template <typename T>
static void argsort_impl(const T *host_data, int64_t *result,
                         int64_t outer_size, int64_t axis_size,
                         bool descending) {
  std::pair<T, int64_t> *pairs = new std::pair<T, int64_t>[axis_size];
  for (int64_t outer = 0; outer < outer_size; ++outer) {
    int64_t base = outer * axis_size;
    for (int64_t i = 0; i < axis_size; ++i) {
      pairs[i] = {host_data[base + i], i};
    }
    if (descending) {
      std::sort(pairs, pairs + axis_size,
                [](const auto &a, const auto &b) { return a.first > b.first; });
    } else {
      std::sort(pairs, pairs + axis_size,
                [](const auto &a, const auto &b) { return a.first < b.first; });
    }
    for (int64_t i = 0; i < axis_size; ++i) {
      result[base + i] = pairs[i].second;
    }
  }
  delete[] pairs;
}

extern "C" {

C_Status argsort_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);

  if (!out || !x) {
    gpu_set_last_error("argsort: null array pointer");
    return C_FAILED;
  }

  bool descending = *static_cast<bool *>(inputs[2]);

  int64_t ndim = x->ndim;
  int64_t n = x->numel;
  int64_t axis_size = x->dims[ndim - 1];
  int64_t outer_size = n / axis_size;

  int64_t *result = new int64_t[n];

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32: {
    float *host_data = new float[n];
    cudaMemcpy(host_data, x->data, n * sizeof(float), cudaMemcpyDeviceToHost);
    argsort_impl(host_data, result, outer_size, axis_size, descending);
    delete[] host_data;
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *host_data = new double[n];
    cudaMemcpy(host_data, x->data, n * sizeof(double), cudaMemcpyDeviceToHost);
    argsort_impl(host_data, result, outer_size, axis_size, descending);
    delete[] host_data;
    break;
  }
  case INSIGHT_DTYPE_I32: {
    int32_t *host_data = new int32_t[n];
    cudaMemcpy(host_data, x->data, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    argsort_impl(host_data, result, outer_size, axis_size, descending);
    delete[] host_data;
    break;
  }
  case INSIGHT_DTYPE_I64: {
    int64_t *host_data = new int64_t[n];
    cudaMemcpy(host_data, x->data, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
    argsort_impl(host_data, result, outer_size, axis_size, descending);
    delete[] host_data;
    break;
  }
  default:
    delete[] result;
    gpu_set_last_error("argsort: unsupported dtype");
    return C_FAILED;
  }

  cudaMemcpy(out->data, result, n * sizeof(int64_t), cudaMemcpyHostToDevice);
  delete[] result;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(argsort, INSIGHT_DTYPE_F32, argsort_kernel_gpu);
REGISTER_IXUCA_KERNEL(argsort, INSIGHT_DTYPE_F64, argsort_kernel_gpu);
REGISTER_IXUCA_KERNEL(argsort, INSIGHT_DTYPE_I32, argsort_kernel_gpu);
REGISTER_IXUCA_KERNEL(argsort, INSIGHT_DTYPE_I64, argsort_kernel_gpu);
