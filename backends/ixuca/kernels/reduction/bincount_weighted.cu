// backends/cuda/kernels/reduction/bincount_weighted.cu
/**
 * @file bincount_weighted.cu
 * @brief CUDA kernel for weighted bincount reduction.
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* output (pre-zeroed)
 *   inputs[1] = InsightArray* indices (1D integer array)
 *   inputs[2] = InsightArray* weights (1D array)
 *
 * Output layout:
 *   outputs[0] = InsightArray* result (same as inputs[0])
 */

#include "../../registry/ixuca_registry.h"
#include "../common/atomic_helpers.cuh"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T, typename W>
__global__ void bincount_weighted_kernel(W *dst, const T *src, const W *weights,
                                         int64_t n, int64_t num_bins) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t bin = static_cast<int64_t>(src[idx]);
    if (bin >= 0 && bin < num_bins) {
      if constexpr (std::is_same_v<W, double>) {
        atomicAddDouble(&dst[bin], weights[idx]);
      } else if constexpr (std::is_same_v<W, int64_t>) {
        atomicAdd(reinterpret_cast<unsigned long long *>(&dst[bin]),
                  static_cast<unsigned long long>(weights[idx]));
      } else if constexpr (std::is_same_v<W, int32_t>) {
        atomicAdd(reinterpret_cast<int *>(&dst[bin]),
                  static_cast<int>(weights[idx]));
      } else {
        atomicAdd(&dst[bin], weights[idx]);
      }
    }
  }
}

extern "C" {

C_Status bincount_weighted_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *prepared = static_cast<InsightArray *>(inputs[1]);
  InsightArray *weights_arr = static_cast<InsightArray *>(inputs[2]);

  if (!out || !prepared || !weights_arr) {
    gpu_set_last_error("bincount_weighted: null array pointer");
    return C_FAILED;
  }

  int64_t n = prepared->numel;
  int64_t num_bins = out->numel;

  int threads = reduction_threads();
  int blocks = reduction_blocks(n);

  // Dispatch based on weights dtype (which determines output dtype)
  if (prepared->dtype == INSIGHT_DTYPE_I32) {
    switch (weights_arr->dtype) {
    case INSIGHT_DTYPE_F32:
      cudaMemset(out->data, 0, num_bins * sizeof(float));
      bincount_weighted_kernel<int32_t, float><<<blocks, threads>>>(
          static_cast<float *>(out->data),
          static_cast<const int32_t *>(prepared->data),
          static_cast<const float *>(weights_arr->data), n, num_bins);
      break;
    case INSIGHT_DTYPE_F64:
      cudaMemset(out->data, 0, num_bins * sizeof(double));
      bincount_weighted_kernel<int32_t, double><<<blocks, threads>>>(
          static_cast<double *>(out->data),
          static_cast<const int32_t *>(prepared->data),
          static_cast<const double *>(weights_arr->data), n, num_bins);
      break;
    case INSIGHT_DTYPE_I32:
      cudaMemset(out->data, 0, num_bins * sizeof(int32_t));
      bincount_weighted_kernel<int32_t, int32_t><<<blocks, threads>>>(
          static_cast<int32_t *>(out->data),
          static_cast<const int32_t *>(prepared->data),
          static_cast<const int32_t *>(weights_arr->data), n, num_bins);
      break;
    case INSIGHT_DTYPE_I64:
      cudaMemset(out->data, 0, num_bins * sizeof(int64_t));
      bincount_weighted_kernel<int32_t, int64_t><<<blocks, threads>>>(
          static_cast<int64_t *>(out->data),
          static_cast<const int32_t *>(prepared->data),
          static_cast<const int64_t *>(weights_arr->data), n, num_bins);
      break;
    default:
      gpu_set_last_error("bincount_weighted: unsupported weights dtype");
      return C_FAILED;
    }
  } else if (prepared->dtype == INSIGHT_DTYPE_I64) {
    switch (weights_arr->dtype) {
    case INSIGHT_DTYPE_F32:
      cudaMemset(out->data, 0, num_bins * sizeof(float));
      bincount_weighted_kernel<int64_t, float><<<blocks, threads>>>(
          static_cast<float *>(out->data),
          static_cast<const int64_t *>(prepared->data),
          static_cast<const float *>(weights_arr->data), n, num_bins);
      break;
    case INSIGHT_DTYPE_F64:
      cudaMemset(out->data, 0, num_bins * sizeof(double));
      bincount_weighted_kernel<int64_t, double><<<blocks, threads>>>(
          static_cast<double *>(out->data),
          static_cast<const int64_t *>(prepared->data),
          static_cast<const double *>(weights_arr->data), n, num_bins);
      break;
    case INSIGHT_DTYPE_I32:
      cudaMemset(out->data, 0, num_bins * sizeof(int32_t));
      bincount_weighted_kernel<int64_t, int32_t><<<blocks, threads>>>(
          static_cast<int32_t *>(out->data),
          static_cast<const int64_t *>(prepared->data),
          static_cast<const int32_t *>(weights_arr->data), n, num_bins);
      break;
    case INSIGHT_DTYPE_I64:
      cudaMemset(out->data, 0, num_bins * sizeof(int64_t));
      bincount_weighted_kernel<int64_t, int64_t><<<blocks, threads>>>(
          static_cast<int64_t *>(out->data),
          static_cast<const int64_t *>(prepared->data),
          static_cast<const int64_t *>(weights_arr->data), n, num_bins);
      break;
    default:
      gpu_set_last_error("bincount_weighted: unsupported weights dtype");
      return C_FAILED;
    }
  } else {
    gpu_set_last_error("bincount_weighted: unsupported input dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(bincount_weighted, INSIGHT_DTYPE_I32,
                         bincount_weighted_kernel_gpu);
REGISTER_IXUCA_KERNEL(bincount_weighted, INSIGHT_DTYPE_I64,
                         bincount_weighted_kernel_gpu);
