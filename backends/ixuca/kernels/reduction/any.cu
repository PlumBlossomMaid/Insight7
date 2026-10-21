// backends/cuda/kernels/reduction/any.cu
/**
 * @file any.cu
 * @brief CUDA kernel for any reduction.
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void any_kernel(bool *dst, const T *src, int64_t batch_size,
                           int64_t reduce_size) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) {
    bool any = false;
    for (int64_t j = 0; j < reduce_size; ++j) {
      if (src[i * reduce_size + j] != 0) {
        any = true;
        break;
      }
    }
    dst[i] = any;
  }
}

extern "C" {

C_Status any_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *prepared = static_cast<InsightArray *>(inputs[1]);

  if (!out || !prepared) {
    gpu_set_last_error("any: null array pointer");
    return C_FAILED;
  }

  int64_t batch_size = *static_cast<int64_t *>(inputs[2]);
  int64_t reduce_size = *static_cast<int64_t *>(inputs[3]);

  int threads = reduction_threads();
  int blocks = reduction_blocks(batch_size);

  switch (prepared->dtype) {
  case INSIGHT_DTYPE_BOOL:
    any_kernel<bool><<<blocks, threads>>>(
        static_cast<bool *>(out->data),
        static_cast<const bool *>(prepared->data), batch_size, reduce_size);
    break;
  case INSIGHT_DTYPE_F32:
    any_kernel<float><<<blocks, threads>>>(
        static_cast<bool *>(out->data),
        static_cast<const float *>(prepared->data), batch_size, reduce_size);
    break;
  case INSIGHT_DTYPE_F64:
    any_kernel<double><<<blocks, threads>>>(
        static_cast<bool *>(out->data),
        static_cast<const double *>(prepared->data), batch_size, reduce_size);
    break;
  case INSIGHT_DTYPE_I32:
    any_kernel<int32_t><<<blocks, threads>>>(
        static_cast<bool *>(out->data),
        static_cast<const int32_t *>(prepared->data), batch_size, reduce_size);
    break;
  case INSIGHT_DTYPE_I64:
    any_kernel<int64_t><<<blocks, threads>>>(
        static_cast<bool *>(out->data),
        static_cast<const int64_t *>(prepared->data), batch_size, reduce_size);
    break;
  default:
    gpu_set_last_error("any: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(any, INSIGHT_DTYPE_BOOL, any_kernel_gpu);
REGISTER_IXUCA_KERNEL(any, INSIGHT_DTYPE_F32, any_kernel_gpu);
REGISTER_IXUCA_KERNEL(any, INSIGHT_DTYPE_F64, any_kernel_gpu);
REGISTER_IXUCA_KERNEL(any, INSIGHT_DTYPE_I32, any_kernel_gpu);
REGISTER_IXUCA_KERNEL(any, INSIGHT_DTYPE_I64, any_kernel_gpu);
