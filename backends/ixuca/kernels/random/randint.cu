// backends/cuda/kernels/random/randint.cu
/**
 * @file randint.cu
 * @brief CUDA kernel for random integer distribution [low, high).
 *
 * Generates random integers from uniform distribution.
 * Supports I32 and I64 types.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = int64_t* low
 *                [2] = int64_t* high
 *                [3] = uint64_t* device seed
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void randint_kernel(curandState *states, int64_t n, T *out,
                               int64_t low, int64_t high) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T range = static_cast<T>(high - low);
    out[idx] = static_cast<T>(low) +
               static_cast<T>(curand_uniform(states + idx) * range);
  }
}

extern "C" {

C_Status randint_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("randint: output array is null");
    return C_FAILED;
  }

  int64_t low = *static_cast<int64_t *>(inputs[1]);
  int64_t high = *static_cast<int64_t *>(inputs[2]);
  uint64_t seed = *static_cast<uint64_t *>(inputs[3]);

  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = random_threads();
  int blocks = random_blocks(n);

  curandState *states;
  cudaMalloc(&states, n * sizeof(curandState));
  init_curand_states<<<blocks, threads>>>(
      states, static_cast<unsigned long long>(seed), n);

  switch (out->dtype) {
  case INSIGHT_DTYPE_I32:
    randint_kernel<int32_t><<<blocks, threads>>>(
        states, n, static_cast<int32_t *>(out->data), low, high);
    break;
  case INSIGHT_DTYPE_I64:
    randint_kernel<int64_t><<<blocks, threads>>>(
        states, n, static_cast<int64_t *>(out->data), low, high);
    break;
  default:
    cudaFree(states);
    gpu_set_last_error("randint: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  cudaFree(states);

  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(randint, INSIGHT_DTYPE_I32, randint_kernel_gpu);
REGISTER_IXUCA_KERNEL(randint, INSIGHT_DTYPE_I64, randint_kernel_gpu);
