// backends/cuda/kernels/indexing/where.cu
/**
 * @file where.cu
 * @brief CUDA kernel for where operation.
 *
 * Returns elements chosen from x or y depending on condition.
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void where_kernel(const bool *cond, const T *x, const T *y, T *out,
                             int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = cond[idx] ? x[idx] : y[idx];
  }
}

extern "C" {

C_Status where_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *cond = static_cast<InsightArray *>(inputs[1]);
  InsightArray *x = static_cast<InsightArray *>(inputs[2]);
  InsightArray *y = static_cast<InsightArray *>(inputs[3]);

  if (!out || !cond || !x || !y) {
    gpu_set_last_error("where: null array pointer");
    return C_FAILED;
  }

  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    where_kernel<float>
        <<<blocks, threads>>>(static_cast<const bool *>(cond->data),
                              static_cast<const float *>(x->data),
                              static_cast<const float *>(y->data),
                              static_cast<float *>(out->data), n);
    break;
  case INSIGHT_DTYPE_F64:
    where_kernel<double>
        <<<blocks, threads>>>(static_cast<const bool *>(cond->data),
                              static_cast<const double *>(x->data),
                              static_cast<const double *>(y->data),
                              static_cast<double *>(out->data), n);
    break;
  case INSIGHT_DTYPE_I32:
    where_kernel<int32_t>
        <<<blocks, threads>>>(static_cast<const bool *>(cond->data),
                              static_cast<const int32_t *>(x->data),
                              static_cast<const int32_t *>(y->data),
                              static_cast<int32_t *>(out->data), n);
    break;
  case INSIGHT_DTYPE_I64:
    where_kernel<int64_t>
        <<<blocks, threads>>>(static_cast<const bool *>(cond->data),
                              static_cast<const int64_t *>(x->data),
                              static_cast<const int64_t *>(y->data),
                              static_cast<int64_t *>(out->data), n);
    break;
  default:
    gpu_set_last_error("where: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(where, INSIGHT_DTYPE_F32, where_kernel_gpu);
REGISTER_IXUCA_KERNEL(where, INSIGHT_DTYPE_F64, where_kernel_gpu);
REGISTER_IXUCA_KERNEL(where, INSIGHT_DTYPE_I32, where_kernel_gpu);
REGISTER_IXUCA_KERNEL(where, INSIGHT_DTYPE_I64, where_kernel_gpu);
