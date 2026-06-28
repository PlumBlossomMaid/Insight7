// backends/cuda/kernels/reduction/max.cu
/**
 * @file max.cu
 * @brief CUDA kernel for max reduction.
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <limits>

template <typename T>
__global__ void max_kernel(T *dst, const T *src, int64_t total_out,
                           int64_t reduce_size) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total_out) {
    T max_val = src[i * reduce_size];
    for (int64_t j = 1; j < reduce_size; ++j) {
      T val = src[i * reduce_size + j];
      if (val > max_val)
        max_val = val;
    }
    dst[i] = max_val;
  }
}

extern "C" {

C_Status max_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *prepared = static_cast<InsightArray *>(inputs[1]);

  if (!out || !prepared) {
    gpu_set_last_error("max: null array pointer");
    return C_FAILED;
  }

  int64_t total_out = out->numel;
  int64_t reduce_size = prepared->numel / total_out;

  int threads = reduction_threads();
  int blocks = reduction_blocks(total_out);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    max_kernel<float><<<blocks, threads>>>(
        static_cast<float *>(out->data),
        static_cast<const float *>(prepared->data), total_out, reduce_size);
    break;
  case INSIGHT_DTYPE_F64:
    max_kernel<double><<<blocks, threads>>>(
        static_cast<double *>(out->data),
        static_cast<const double *>(prepared->data), total_out, reduce_size);
    break;
  case INSIGHT_DTYPE_I32:
    max_kernel<int32_t><<<blocks, threads>>>(
        static_cast<int32_t *>(out->data),
        static_cast<const int32_t *>(prepared->data), total_out, reduce_size);
    break;
  case INSIGHT_DTYPE_I64:
    max_kernel<int64_t><<<blocks, threads>>>(
        static_cast<int64_t *>(out->data),
        static_cast<const int64_t *>(prepared->data), total_out, reduce_size);
    break;
  default:
    gpu_set_last_error("max: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(max, INSIGHT_DTYPE_F32, max_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(max, INSIGHT_DTYPE_F64, max_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(max, INSIGHT_DTYPE_I32, max_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(max, INSIGHT_DTYPE_I64, max_kernel_gpu);
