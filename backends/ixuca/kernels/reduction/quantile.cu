// backends/cuda/kernels/reduction/quantile.cu
/**
 * @file quantile.cu
 * @brief CUDA kernel for quantile reduction.
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* output (dtype F64)
 *   inputs[1] = InsightArray* prepared (dtype F32 or F64)
 *   inputs[2] = int64_t* batch_size
 *   inputs[3] = int64_t* reduce_size
 *   inputs[4] = double* q
 *
 * Output is always F64 (matching CPU kernel behavior).
 *
 * Note: Uses CPU fallback with sorting since full GPU sort is complex.
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

template <typename T>
static void compute_quantile(const T *host_data, double *result,
                             int64_t batch_size, int64_t reduce_size,
                             double q) {
  T *sorted_slice = new T[reduce_size];
  for (int64_t i = 0; i < batch_size; ++i) {
    const T *slice = host_data + i * reduce_size;
    for (int64_t j = 0; j < reduce_size; ++j) {
      sorted_slice[j] = slice[j];
    }
    std::sort(sorted_slice, sorted_slice + reduce_size);

    if (q == 0.5) {
      if (reduce_size % 2 == 1) {
        result[i] = static_cast<double>(sorted_slice[reduce_size / 2]);
      } else {
        result[i] = (static_cast<double>(sorted_slice[reduce_size / 2 - 1]) +
                     static_cast<double>(sorted_slice[reduce_size / 2])) *
                    0.5;
      }
    } else {
      double idx = q * (reduce_size - 1);
      int64_t lo = static_cast<int64_t>(idx);
      int64_t hi = lo + 1;
      if (hi >= reduce_size)
        hi = reduce_size - 1;
      double frac = idx - lo;
      result[i] = static_cast<double>(sorted_slice[lo]) * (1 - frac) +
                  static_cast<double>(sorted_slice[hi]) * frac;
    }
  }
  delete[] sorted_slice;
}

extern "C" {

C_Status quantile_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *prepared = static_cast<InsightArray *>(inputs[1]);

  if (!out || !prepared) {
    gpu_set_last_error("quantile: null array pointer");
    return C_FAILED;
  }

  int64_t batch_size = *static_cast<int64_t *>(inputs[2]);
  int64_t reduce_size = *static_cast<int64_t *>(inputs[3]);
  double q = *static_cast<double *>(inputs[4]);

  double *result = new double[batch_size];

  switch (prepared->dtype) {
  case INSIGHT_DTYPE_F32: {
    float *host_data = new float[prepared->numel];
    cudaMemcpy(host_data, prepared->data, prepared->numel * sizeof(float),
               cudaMemcpyDeviceToHost);
    compute_quantile(host_data, result, batch_size, reduce_size, q);
    delete[] host_data;
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *host_data = new double[prepared->numel];
    cudaMemcpy(host_data, prepared->data, prepared->numel * sizeof(double),
               cudaMemcpyDeviceToHost);
    compute_quantile(host_data, result, batch_size, reduce_size, q);
    delete[] host_data;
    break;
  }
  default:
    delete[] result;
    gpu_set_last_error("quantile: unsupported dtype");
    return C_FAILED;
  }

  // Copy result back to GPU as double (F64)
  cudaMemcpy(out->data, result, batch_size * sizeof(double),
             cudaMemcpyHostToDevice);

  delete[] result;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(quantile, INSIGHT_DTYPE_F32, quantile_kernel_gpu);
REGISTER_IXUCA_KERNEL(quantile, INSIGHT_DTYPE_F64, quantile_kernel_gpu);
