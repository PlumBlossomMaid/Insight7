// backends/cuda/kernels/reduction/bincount.cu
/**
 * @file bincount.cu
 * @brief CUDA kernel for bincount reduction.
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void bincount_kernel(int64_t *dst, const T *src, int64_t n,
                                int64_t num_bins) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t bin = static_cast<int64_t>(src[idx]);
    if (bin >= 0 && bin < num_bins) {
      atomicAdd(reinterpret_cast<unsigned long long *>(&dst[bin]),
                static_cast<unsigned long long>(1));
    }
  }
}

extern "C" {

C_Status bincount_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *prepared = static_cast<InsightArray *>(inputs[1]);

  if (!out || !prepared) {
    gpu_set_last_error("bincount: null array pointer");
    return C_FAILED;
  }

  int64_t n = prepared->numel;
  int64_t num_bins = out->numel;

  // Zero output first
  cudaMemset(out->data, 0, num_bins * sizeof(int64_t));

  int threads = reduction_threads();
  int blocks = reduction_blocks(n);

  switch (prepared->dtype) {
  case INSIGHT_DTYPE_I32:
    bincount_kernel<int32_t><<<blocks, threads>>>(
        static_cast<int64_t *>(out->data),
        static_cast<const int32_t *>(prepared->data), n, num_bins);
    break;
  case INSIGHT_DTYPE_I64:
    bincount_kernel<int64_t><<<blocks, threads>>>(
        static_cast<int64_t *>(out->data),
        static_cast<const int64_t *>(prepared->data), n, num_bins);
    break;
  default:
    gpu_set_last_error("bincount: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

/**
 * @brief Disabled entry point: defers bincount to the CPU implementation.
 *
 * DISABLED PENDING A COREX NUMERICAL FIX. bincount_kernel_gpu above produces
 * wrong results on CoreX (see tests/cuda/cpp/test_reduction.cpp, BincountBasic
 * and BincountMinlength) 鈥?the atomic histogram accumulation does not match
 * the CPU/CUDA reference.
 *
 * Returning C_FALLBACK makes the framework transfer the inputs to the host and
 * run the CPU kernel, which is correct but slower. Silently returning wrong
 * counts is not an option, and an unregistered op would hard-fail
 * (src/core/kernel.cpp: insight_kernel_launch returns C_FAILED when no kernel
 * is found 鈥?the fallback path only runs for a *registered* kernel that
 * returns C_FALLBACK).
 *
 * Re-register bincount_kernel_gpu once the CoreX issue is fixed.
 */
C_Status bincount_kernel_gpu_fallback(void **inputs, void **outputs) {
  (void)inputs;
  (void)outputs;
  return C_FALLBACK;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(bincount, INSIGHT_DTYPE_I32,
                      bincount_kernel_gpu_fallback);
REGISTER_IXUCA_KERNEL(bincount, INSIGHT_DTYPE_I64,
                      bincount_kernel_gpu_fallback);
