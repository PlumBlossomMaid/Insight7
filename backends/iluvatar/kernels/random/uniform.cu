// backends/cuda/kernels/random/uniform.cu
/**
 * @file uniform.cu
 * @brief CUDA kernel for uniform distribution [low, high).
 *
 * Generates random numbers from uniform(low, high) distribution.
 * Supports F32 and F64 types.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = double* low
 *                [2] = double* high
 *                [3] = uint64_t* device seed
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void uniform_kernel(curandState *states, int64_t n, T *out, T low,
                               T high) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = low + (high - low) * uniform_f32(states + idx);
  }
}

template <>
__global__ void uniform_kernel<double>(curandState *states, int64_t n,
                                       double *out, double low, double high) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = low + (high - low) * uniform_f64(states + idx);
  }
}

extern "C" {

C_Status uniform_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("uniform: output array is null");
    return C_FAILED;
  }

  double low = *static_cast<double *>(inputs[1]);
  double high = *static_cast<double *>(inputs[2]);
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
    uniform_kernel<float><<<blocks, threads>>>(
        states, n, static_cast<float *>(out->data), static_cast<float>(low),
        static_cast<float>(high));
    break;
  case INSIGHT_DTYPE_F64:
    uniform_kernel<double><<<blocks, threads>>>(
        states, n, static_cast<double *>(out->data), low, high);
    break;
  default:
    cudaFree(states);
    gpu_set_last_error("uniform: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(uniform, INSIGHT_DTYPE_F32, uniform_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(uniform, INSIGHT_DTYPE_F64, uniform_kernel_gpu);
