// backends/cuda/kernels/random/beta.cu
/**
 * @file beta.cu
 * @brief CUDA kernel for beta distribution.
 *
 * Generates random numbers from beta(a, b) distribution.
 * beta(a, b) = Gamma(a) / (Gamma(a) + Gamma(b))
 * Supports F32 and F64 types.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = double* a
 *                [2] = double* b
 *                [3] = uint64_t* device seed
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void beta_kernel(curandState *states, int64_t n, T *out, T a, T b) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T x = gamma_f32(states + idx, a);
    T y = gamma_f32(states + idx, b);
    out[idx] = x / (x + y);
  }
}

template <>
__global__ void beta_kernel<double>(curandState *states, int64_t n, double *out,
                                    double a, double b) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    double x = gamma_f64(states + idx, a);
    double y = gamma_f64(states + idx, b);
    out[idx] = x / (x + y);
  }
}

extern "C" {

C_Status beta_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("beta: output array is null");
    return C_FAILED;
  }

  double a = *static_cast<double *>(inputs[1]);
  double b = *static_cast<double *>(inputs[2]);
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
    beta_kernel<float>
        <<<blocks, threads>>>(states, n, static_cast<float *>(out->data),
                              static_cast<float>(a), static_cast<float>(b));
    break;
  case INSIGHT_DTYPE_F64:
    beta_kernel<double><<<blocks, threads>>>(
        states, n, static_cast<double *>(out->data), a, b);
    break;
  default:
    cudaFree(states);
    gpu_set_last_error("beta: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(beta, INSIGHT_DTYPE_F32, beta_kernel_gpu);
REGISTER_IXUCA_KERNEL(beta, INSIGHT_DTYPE_F64, beta_kernel_gpu);
