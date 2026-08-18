// backends/cuda/kernels/random/randperm.cu
/**
 * @file randperm.cu
 * @brief CUDA kernel for random permutation.
 *
 * Generates a random permutation of [0, n).
 * Uses CPU implementation (Fisher-Yates shuffle) as GPU fallback.
 * Supports I32 and I64 types.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = uint64_t* device seed
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <random>

extern "C" {

C_Status randperm_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("randperm: output array is null");
    return C_FAILED;
  }

  uint64_t seed = *static_cast<uint64_t *>(inputs[1]);
  int64_t n = out->numel;

  if (n == 0)
    return C_SUCCESS;

  // Generate permutation on CPU (Fisher-Yates shuffle)
  int64_t *perm = new int64_t[n];
  for (int64_t i = 0; i < n; ++i) {
    perm[i] = i;
  }

  std::mt19937 rng(static_cast<uint32_t>(seed));
  for (int64_t i = n - 1; i > 0; --i) {
    std::uniform_int_distribution<int64_t> dist(0, i);
    int64_t j = dist(rng);
    std::swap(perm[i], perm[j]);
  }

  // Copy to device
  switch (out->dtype) {
  case INSIGHT_DTYPE_I32: {
    int32_t *perm32 = new int32_t[n];
    for (int64_t i = 0; i < n; ++i) {
      perm32[i] = static_cast<int32_t>(perm[i]);
    }
    cudaMemcpy(out->data, perm32, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    delete[] perm32;
    break;
  }
  case INSIGHT_DTYPE_I64:
    cudaMemcpy(out->data, perm, n * sizeof(int64_t), cudaMemcpyHostToDevice);
    break;
  default:
    delete[] perm;
    gpu_set_last_error("randperm: unsupported dtype");
    return C_FAILED;
  }

  delete[] perm;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(randperm, INSIGHT_DTYPE_I32, randperm_kernel_gpu);
REGISTER_IXUCA_KERNEL(randperm, INSIGHT_DTYPE_I64, randperm_kernel_gpu);
