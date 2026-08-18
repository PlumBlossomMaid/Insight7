// backends/cuda/kernels/indexing/put.cu
/**
 * @file put.cu
 * @brief CUDA kernel for put operation.
 *
 * Puts values into an array at specified indices.
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void put_kernel(T *x, const int64_t *indices, const T *values,
                           int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[indices[idx]] = values[idx];
  }
}

extern "C" {

C_Status put_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = static_cast<InsightArray *>(inputs[0]);
  InsightArray *indices = static_cast<InsightArray *>(inputs[1]);
  InsightArray *values = static_cast<InsightArray *>(inputs[2]);

  if (!x || !indices || !values) {
    gpu_set_last_error("put: null array pointer");
    return C_FAILED;
  }

  int64_t n = indices->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32:
    put_kernel<float>
        <<<blocks, threads>>>(static_cast<float *>(x->data),
                              static_cast<const int64_t *>(indices->data),
                              static_cast<const float *>(values->data), n);
    break;
  case INSIGHT_DTYPE_F64:
    put_kernel<double>
        <<<blocks, threads>>>(static_cast<double *>(x->data),
                              static_cast<const int64_t *>(indices->data),
                              static_cast<const double *>(values->data), n);
    break;
  case INSIGHT_DTYPE_I32:
    put_kernel<int32_t>
        <<<blocks, threads>>>(static_cast<int32_t *>(x->data),
                              static_cast<const int64_t *>(indices->data),
                              static_cast<const int32_t *>(values->data), n);
    break;
  case INSIGHT_DTYPE_I64:
    put_kernel<int64_t>
        <<<blocks, threads>>>(static_cast<int64_t *>(x->data),
                              static_cast<const int64_t *>(indices->data),
                              static_cast<const int64_t *>(values->data), n);
    break;
  default:
    gpu_set_last_error("put: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(put, INSIGHT_DTYPE_F32, put_kernel_gpu);
REGISTER_IXUCA_KERNEL(put, INSIGHT_DTYPE_F64, put_kernel_gpu);
REGISTER_IXUCA_KERNEL(put, INSIGHT_DTYPE_I32, put_kernel_gpu);
REGISTER_IXUCA_KERNEL(put, INSIGHT_DTYPE_I64, put_kernel_gpu);
