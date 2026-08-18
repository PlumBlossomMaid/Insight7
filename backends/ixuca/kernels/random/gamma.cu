// backends/cuda/kernels/random/gamma.cu
/**
 * @file gamma.cu
 * @brief CUDA kernel for gamma distribution.
 *
 * Generates random numbers from gamma(shape, rate) distribution.
 * Uses Marsaglia-Tsang method for generation.
 * Supports F32 and F64 types.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = double* shape
 *                [2] = double* rate
 *                [3] = uint64_t* device seed
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void gamma_kernel(curandState *states, int64_t n, T *out, T shape,
                             T rate) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = gamma_f32(states + idx, shape) * rate;
  }
}

template <>
__global__ void gamma_kernel<double>(curandState *states, int64_t n,
                                     double *out, double shape, double rate) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = gamma_f64(states + idx, shape) * rate;
  }
}

extern "C" {

C_Status gamma_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("gamma: output array is null");
    return C_FAILED;
  }

  double shape_param = *static_cast<double *>(inputs[1]);
  double rate = *static_cast<double *>(inputs[2]);
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
  case INSIGHT_DTYPE_F32:
    gamma_kernel<float><<<blocks, threads>>>(
        states, n, static_cast<float *>(out->data),
        static_cast<float>(shape_param), static_cast<float>(1.0 / rate));
    break;
  case INSIGHT_DTYPE_F64:
    gamma_kernel<double><<<blocks, threads>>>(
        states, n, static_cast<double *>(out->data), shape_param, 1.0 / rate);
    break;
  default:
    cudaFree(states);
    gpu_set_last_error("gamma: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(gamma, INSIGHT_DTYPE_F32, gamma_kernel_gpu);
REGISTER_IXUCA_KERNEL(gamma, INSIGHT_DTYPE_F64, gamma_kernel_gpu);
