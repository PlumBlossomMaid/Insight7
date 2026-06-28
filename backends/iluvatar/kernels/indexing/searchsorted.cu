// backends/cuda/kernels/indexing/searchsorted.cu
/**
 * @file searchsorted.cu
 * @brief CUDA kernel for searchsorted operation.
 *
 * Finds indices where elements should be inserted to maintain order.
 * Uses CPU fallback.
 */

#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

template <typename T>
static void searchsorted_impl(const T *sorted_data, int64_t sorted_size,
                              const T *values_data, int64_t *result, int64_t n,
                              bool right) {
  for (int64_t i = 0; i < n; ++i) {
    if (right) {
      result[i] = std::upper_bound(sorted_data, sorted_data + sorted_size,
                                   values_data[i]) -
                  sorted_data;
    } else {
      result[i] = std::lower_bound(sorted_data, sorted_data + sorted_size,
                                   values_data[i]) -
                  sorted_data;
    }
  }
}

extern "C" {

C_Status searchsorted_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *sorted = static_cast<InsightArray *>(inputs[1]);
  InsightArray *values = static_cast<InsightArray *>(inputs[2]);

  if (!out || !sorted || !values) {
    gpu_set_last_error("searchsorted: null array pointer");
    return C_FAILED;
  }

  int side_code = *static_cast<int *>(inputs[3]);
  bool right = (side_code != 0);

  int64_t n = values->numel;
  int64_t sorted_size = sorted->numel;

  int64_t *result = new int64_t[n];

  switch (sorted->dtype) {
  case INSIGHT_DTYPE_F32: {
    float *sorted_data = new float[sorted_size];
    float *values_data = new float[n];
    cudaMemcpy(sorted_data, sorted->data, sorted_size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(values_data, values->data, n * sizeof(float),
               cudaMemcpyDeviceToHost);
    searchsorted_impl(sorted_data, sorted_size, values_data, result, n, right);
    delete[] sorted_data;
    delete[] values_data;
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *sorted_data = new double[sorted_size];
    double *values_data = new double[n];
    cudaMemcpy(sorted_data, sorted->data, sorted_size * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(values_data, values->data, n * sizeof(double),
               cudaMemcpyDeviceToHost);
    searchsorted_impl(sorted_data, sorted_size, values_data, result, n, right);
    delete[] sorted_data;
    delete[] values_data;
    break;
  }
  case INSIGHT_DTYPE_I32: {
    int32_t *sorted_data = new int32_t[sorted_size];
    int32_t *values_data = new int32_t[n];
    cudaMemcpy(sorted_data, sorted->data, sorted_size * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(values_data, values->data, n * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    searchsorted_impl(sorted_data, sorted_size, values_data, result, n, right);
    delete[] sorted_data;
    delete[] values_data;
    break;
  }
  case INSIGHT_DTYPE_I64: {
    int64_t *sorted_data = new int64_t[sorted_size];
    int64_t *values_data = new int64_t[n];
    cudaMemcpy(sorted_data, sorted->data, sorted_size * sizeof(int64_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(values_data, values->data, n * sizeof(int64_t),
               cudaMemcpyDeviceToHost);
    searchsorted_impl(sorted_data, sorted_size, values_data, result, n, right);
    delete[] sorted_data;
    delete[] values_data;
    break;
  }
  default:
    delete[] result;
    gpu_set_last_error("searchsorted: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(searchsorted, INSIGHT_DTYPE_F32, searchsorted_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(searchsorted, INSIGHT_DTYPE_F64, searchsorted_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(searchsorted, INSIGHT_DTYPE_I32, searchsorted_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(searchsorted, INSIGHT_DTYPE_I64, searchsorted_kernel_gpu);
