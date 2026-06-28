// backends/cuda/kernels/indexing/put_along_axis.cu
/**
 * @file put_along_axis.cu
 * @brief CUDA kernel for put_along_axis operation.
 */

#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void
put_along_axis_kernel(T *x, const int64_t *indices, const T *values, int64_t n,
                      int64_t ndim, const int64_t *dims,
                      const int64_t *x_strides, const int64_t *idx_strides,
                      int64_t axis) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Convert linear index to multi-dimensional indices
    int64_t out_indices[10];
    int64_t remaining = idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
      out_indices[d] = remaining % dims[d];
      remaining /= dims[d];
    }

    // Get index value
    int64_t index_offset = 0;
    for (int64_t d = 0; d < ndim; ++d) {
      index_offset += out_indices[d] * idx_strides[d];
    }
    int64_t src_idx = indices[index_offset];

    // Compute x offset
    int64_t x_offset = 0;
    for (int64_t d = 0; d < ndim; ++d) {
      int64_t idx_d = (d == axis) ? src_idx : out_indices[d];
      x_offset += idx_d * x_strides[d];
    }

    x[x_offset] = values[idx];
  }
}

extern "C" {

C_Status put_along_axis_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = static_cast<InsightArray *>(inputs[0]);
  InsightArray *indices = static_cast<InsightArray *>(inputs[1]);
  InsightArray *values = static_cast<InsightArray *>(inputs[2]);

  if (!x || !indices || !values) {
    gpu_set_last_error("put_along_axis: null array pointer");
    return C_FAILED;
  }

  int axis = *static_cast<int *>(inputs[3]);
  int64_t ndim = x->ndim;
  if (axis < 0)
    axis += ndim;

  int64_t n = values->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  int64_t *d_dims, *d_x_strides, *d_idx_strides;
  cudaMalloc(&d_dims, ndim * sizeof(int64_t));
  cudaMalloc(&d_x_strides, ndim * sizeof(int64_t));
  cudaMalloc(&d_idx_strides, ndim * sizeof(int64_t));
  cudaMemcpy(d_dims, indices->dims, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_x_strides, x->strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx_strides, indices->strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32:
    put_along_axis_kernel<float>
        <<<blocks, threads>>>(static_cast<float *>(x->data),
                              static_cast<const int64_t *>(indices->data),
                              static_cast<const float *>(values->data), n, ndim,
                              d_dims, d_x_strides, d_idx_strides, axis);
    break;
  case INSIGHT_DTYPE_F64:
    put_along_axis_kernel<double>
        <<<blocks, threads>>>(static_cast<double *>(x->data),
                              static_cast<const int64_t *>(indices->data),
                              static_cast<const double *>(values->data), n,
                              ndim, d_dims, d_x_strides, d_idx_strides, axis);
    break;
  case INSIGHT_DTYPE_I32:
    put_along_axis_kernel<int32_t>
        <<<blocks, threads>>>(static_cast<int32_t *>(x->data),
                              static_cast<const int64_t *>(indices->data),
                              static_cast<const int32_t *>(values->data), n,
                              ndim, d_dims, d_x_strides, d_idx_strides, axis);
    break;
  case INSIGHT_DTYPE_I64:
    put_along_axis_kernel<int64_t>
        <<<blocks, threads>>>(static_cast<int64_t *>(x->data),
                              static_cast<const int64_t *>(indices->data),
                              static_cast<const int64_t *>(values->data), n,
                              ndim, d_dims, d_x_strides, d_idx_strides, axis);
    break;
  default:
    cudaFree(d_dims);
    cudaFree(d_x_strides);
    cudaFree(d_idx_strides);
    gpu_set_last_error("put_along_axis: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  cudaFree(d_dims);
  cudaFree(d_x_strides);
  cudaFree(d_idx_strides);

  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(put_along_axis, INSIGHT_DTYPE_F32,
                    put_along_axis_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(put_along_axis, INSIGHT_DTYPE_F64,
                    put_along_axis_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(put_along_axis, INSIGHT_DTYPE_I32,
                    put_along_axis_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(put_along_axis, INSIGHT_DTYPE_I64,
                    put_along_axis_kernel_gpu);
