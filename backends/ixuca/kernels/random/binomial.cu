// backends/cuda/kernels/random/binomial.cu
/**
 * @file binomial.cu
 * @brief CUDA kernel for binomial distribution.
 *
 * Generates random numbers from binomial(n_trials, p) distribution.
 * Supports I32 and I64 types.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = int64_t* n_trials
 *                [2] = double* p
 *                [3] = uint64_t* device seed
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void binomial_kernel(curandState *states, int64_t n, T *out,
                                int64_t n_trials, double p) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T count = 0;
    for (int64_t i = 0; i < n_trials; ++i) {
      if (uniform_f64(states + idx) < p) {
        count++;
      }
    }
    out[idx] = count;
  }
}

extern "C" {

C_Status binomial_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("binomial: output array is null");
    return C_FAILED;
  }

  int64_t n_trials = *static_cast<int64_t *>(inputs[1]);
  double p = *static_cast<double *>(inputs[2]);
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
    binomial_kernel<int32_t><<<blocks, threads>>>(
        states, n, static_cast<int32_t *>(out->data), n_trials, p);
    break;
  case INSIGHT_DTYPE_I64:
    binomial_kernel<int64_t><<<blocks, threads>>>(
        states, n, static_cast<int64_t *>(out->data), n_trials, p);
    break;
  default:
    cudaFree(states);
    gpu_set_last_error("binomial: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(binomial, INSIGHT_DTYPE_I32, binomial_kernel_gpu);
REGISTER_IXUCA_KERNEL(binomial, INSIGHT_DTYPE_I64, binomial_kernel_gpu);
