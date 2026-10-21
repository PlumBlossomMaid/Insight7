// backends/cuda/kernels/random/randn.cu
/**
 * @file randn.cu
 * @brief CUDA kernel for normal distribution (0, 1).
 *
 * Generates random numbers from standard normal distribution.
 * Supports F32 and F64 types.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = uint64_t* device seed
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void randn_kernel(curandState *states, int64_t n, T *out) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = normal_f32(states + idx);
  }
}

template <>
__global__ void randn_kernel<double>(curandState *states, int64_t n,
                                     double *out) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = normal_f64(states + idx);
  }
}

extern "C" {

C_Status randn_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("randn: output array is null");
    return C_FAILED;
  }

  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  uint64_t seed = *static_cast<uint64_t *>(inputs[1]);

  int threads = random_threads();
  int blocks = random_blocks(n);

  curandState *states;
  cudaMalloc(&states, n * sizeof(curandState));
  init_curand_states<<<blocks, threads>>>(
      states, static_cast<unsigned long long>(seed), n);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    randn_kernel<float>
        <<<blocks, threads>>>(states, n, static_cast<float *>(out->data));
    break;
  case INSIGHT_DTYPE_F64:
    randn_kernel<double>
        <<<blocks, threads>>>(states, n, static_cast<double *>(out->data));
    break;
  default:
    cudaFree(states);
    gpu_set_last_error("randn: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(randn, INSIGHT_DTYPE_F32, randn_kernel_gpu);
REGISTER_IXUCA_KERNEL(randn, INSIGHT_DTYPE_F64, randn_kernel_gpu);
