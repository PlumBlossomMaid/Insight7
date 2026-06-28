// backends/cuda/kernels/random/exponential.cu
/**
 * @file exponential.cu
 * @brief CUDA kernel for exponential distribution.
 *
 * Generates random numbers from exponential(scale) distribution.
 * Supports F32 and F64 types.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = double* scale
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
__global__ void exponential_kernel(curandState *states, int64_t n, T *out,
                                   T inv_scale) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T u = uniform_f32(states + idx);
    out[idx] = -logf(1.0f - u) / inv_scale;
  }
}

template <>
__global__ void exponential_kernel<double>(curandState *states, int64_t n,
                                           double *out, double inv_scale) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    double u = uniform_f64(states + idx);
    out[idx] = -log(1.0 - u) / inv_scale;
  }
}

extern "C" {

C_Status exponential_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("exponential: output array is null");
    return C_FAILED;
  }

  double scale = *static_cast<double *>(inputs[1]);
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
    exponential_kernel<float>
        <<<blocks, threads>>>(states, n, static_cast<float *>(out->data),
                              static_cast<float>(1.0 / scale));
    break;
  case INSIGHT_DTYPE_F64:
    exponential_kernel<double><<<blocks, threads>>>(
        states, n, static_cast<double *>(out->data), 1.0 / scale);
    break;
  default:
    cudaFree(states);
    gpu_set_last_error("exponential: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(exponential, INSIGHT_DTYPE_F32,
                         exponential_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(exponential, INSIGHT_DTYPE_F64,
                         exponential_kernel_gpu);
