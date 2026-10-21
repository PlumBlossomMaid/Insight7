// backends/cuda/kernels/reduction/cummin.cu
/**
 * @file cummin.cu
 * @brief CUDA kernel for cummin reduction.
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* output
 *   inputs[1] = InsightArray* input
 *   inputs[2] = int* axis
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void cummin_kernel(const T *src, T *dst, int64_t n,
                              int64_t axis_size, int64_t inner_size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t inner = idx % inner_size;
    int64_t i = (idx / inner_size) % axis_size;
    int64_t outer = idx / (axis_size * inner_size);

    int64_t base = outer * axis_size * inner_size + inner;
    T min_val = src[base];
    for (int64_t j = 0; j <= i; ++j) {
      T val = src[base + j * inner_size];
      if (val < min_val)
        min_val = val;
    }
    dst[base + i * inner_size] = min_val;
  }
}

extern "C" {

C_Status cummin_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);

  if (!out || !x) {
    gpu_set_last_error("cummin: null array pointer");
    return C_FAILED;
  }

  int axis = *static_cast<int *>(inputs[2]);
  int64_t ndim = x->ndim;
  if (axis < 0)
    axis += ndim;

  int64_t n = x->numel;
  if (n == 0)
    return C_SUCCESS;

  int64_t axis_size = x->dims[axis];
  int64_t inner_size = 1;
  for (int d = axis + 1; d < ndim; ++d) {
    inner_size *= x->dims[d];
  }

  int threads = reduction_threads();
  int blocks = reduction_blocks(n);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    cummin_kernel<float><<<blocks, threads>>>(
        static_cast<const float *>(x->data), static_cast<float *>(out->data), n,
        axis_size, inner_size);
    break;
  case INSIGHT_DTYPE_F64:
    cummin_kernel<double><<<blocks, threads>>>(
        static_cast<const double *>(x->data), static_cast<double *>(out->data),
        n, axis_size, inner_size);
    break;
  case INSIGHT_DTYPE_I32:
    cummin_kernel<int32_t><<<blocks, threads>>>(
        static_cast<const int32_t *>(x->data),
        static_cast<int32_t *>(out->data), n, axis_size, inner_size);
    break;
  case INSIGHT_DTYPE_I64:
    cummin_kernel<int64_t><<<blocks, threads>>>(
        static_cast<const int64_t *>(x->data),
        static_cast<int64_t *>(out->data), n, axis_size, inner_size);
    break;
  default:
    gpu_set_last_error("cummin: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(cummin, INSIGHT_DTYPE_F32, cummin_kernel_gpu);
REGISTER_IXUCA_KERNEL(cummin, INSIGHT_DTYPE_F64, cummin_kernel_gpu);
REGISTER_IXUCA_KERNEL(cummin, INSIGHT_DTYPE_I32, cummin_kernel_gpu);
REGISTER_IXUCA_KERNEL(cummin, INSIGHT_DTYPE_I64, cummin_kernel_gpu);
