// backends/cuda/kernels/reduction/bincount.cu
/**
 * @file bincount.cu
 * @brief CUDA kernel for bincount reduction.
 */

#include "../../registry/iluvatar_registry.h"
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

} // extern "C"

REGISTER_ILUVATAR_KERNEL(bincount, INSIGHT_DTYPE_I32, bincount_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(bincount, INSIGHT_DTYPE_I64, bincount_kernel_gpu);
