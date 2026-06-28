// backends/cuda/kernels/random/normal.cu
/**
 * @file normal.cu
 * @brief CUDA kernel for normal distribution with mean and std.
 *
 * Generates random numbers from normal(mean, std) distribution.
 * Supports F32 and F64 types.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = double* mean
 *                [2] = double* std
 *                [3] = uint64_t* device seed
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void normal_kernel(curandState *states, int64_t n, T *out, T mean,
                              T std) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = mean + std * normal_f32(states + idx);
  }
}

template <>
__global__ void normal_kernel<double>(curandState *states, int64_t n,
                                      double *out, double mean, double std) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = mean + std * normal_f64(states + idx);
  }
}

extern "C" {

C_Status normal_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("normal: output array is null");
    return C_FAILED;
  }

  double mean = *static_cast<double *>(inputs[1]);
  double std = *static_cast<double *>(inputs[2]);
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
    normal_kernel<float><<<blocks, threads>>>(
        states, n, static_cast<float *>(out->data), static_cast<float>(mean),
        static_cast<float>(std));
    break;
  case INSIGHT_DTYPE_F64:
    normal_kernel<double><<<blocks, threads>>>(
        states, n, static_cast<double *>(out->data), mean, std);
    break;
  default:
    cudaFree(states);
    gpu_set_last_error("normal: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(normal, INSIGHT_DTYPE_F32, normal_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(normal, INSIGHT_DTYPE_F64, normal_kernel_gpu);
