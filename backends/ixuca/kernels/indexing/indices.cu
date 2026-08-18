// backends/cuda/kernels/indexing/indices.cu
/**
 * @file indices.cu
 * @brief CUDA kernel for indices operation.
 *
 * Returns an array of indices.
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

__global__ void indices_kernel(int64_t *out, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = idx;
  }
}

extern "C" {

C_Status indices_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("indices: null array pointer");
    return C_FAILED;
  }

  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  indices_kernel<<<blocks, threads>>>(static_cast<int64_t *>(out->data), n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(indices, INSIGHT_DTYPE_I64, indices_kernel_gpu);
