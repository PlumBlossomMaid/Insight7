// backends/cuda/kernels/random/chisquare.cu
/**
 * @file chisquare.cu
 * @brief CUDA kernel for chi-square distribution.
 *
 * Generates random numbers from chi-square(df) distribution.
 * chi-square(df) = 2 * gamma(df/2, 1)
 * Supports F32 and F64 types.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = double* df (degrees of freedom)
 *                [2] = uint64_t* device seed
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void chisquare_kernel(curandState *states, int64_t n, T *out, T df) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = 2.0f * gamma_f32(states + idx, df / 2.0f);
  }
}

template <>
__global__ void chisquare_kernel<double>(curandState *states, int64_t n,
                                         double *out, double df) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = 2.0 * gamma_f64(states + idx, df / 2.0);
  }
}

extern "C" {

C_Status chisquare_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("chisquare: output array is null");
    return C_FAILED;
  }

  double df = *static_cast<double *>(inputs[1]);
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
  case INSIGHT_DTYPE_F32:
    chisquare_kernel<float><<<blocks, threads>>>(
        states, n, static_cast<float *>(out->data), static_cast<float>(df));
    break;
  case INSIGHT_DTYPE_F64:
    chisquare_kernel<double>
        <<<blocks, threads>>>(states, n, static_cast<double *>(out->data), df);
    break;
  default:
    cudaFree(states);
    gpu_set_last_error("chisquare: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(chisquare, INSIGHT_DTYPE_F32, chisquare_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(chisquare, INSIGHT_DTYPE_F64, chisquare_kernel_gpu);
