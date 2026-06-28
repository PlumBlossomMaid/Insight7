// backends/cuda/kernels/random/poisson.cu
/**
 * @file poisson.cu
 * @brief CUDA kernel for Poisson distribution.
 *
 * Generates random numbers from Poisson(lambda) distribution.
 * Uses inverse transform sampling.
 * Supports I32 and I64 types.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = double* lambda
 *                [2] = uint64_t* device seed
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__global__ void poisson_kernel(curandState *states, int64_t n, T *out,
                               double lambda) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    double L = exp(-lambda);
    T k = 0;
    double p = 1.0;
    do {
      k++;
      p *= uniform_f64(states + idx);
    } while (p > L);
    out[idx] = k - 1;
  }
}

extern "C" {

C_Status poisson_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("poisson: output array is null");
    return C_FAILED;
  }

  double lambda = *static_cast<double *>(inputs[1]);
  uint64_t seed = *static_cast<uint64_t *>(inputs[2]);

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
    poisson_kernel<int32_t><<<blocks, threads>>>(
        states, n, static_cast<int32_t *>(out->data), lambda);
    break;
  case INSIGHT_DTYPE_I64:
    poisson_kernel<int64_t><<<blocks, threads>>>(
        states, n, static_cast<int64_t *>(out->data), lambda);
    break;
  default:
    cudaFree(states);
    gpu_set_last_error("poisson: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(poisson, INSIGHT_DTYPE_I32, poisson_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(poisson, INSIGHT_DTYPE_I64, poisson_kernel_gpu);
