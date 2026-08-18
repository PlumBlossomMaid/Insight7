// backends/cuda/kernels/manipulation/roll.cu
/**
 * @file roll.cu
 * @brief CUDA kernel for roll operation.
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* output
 *   inputs[1] = InsightArray* input
 *   inputs[2] = int* shift
 *   inputs[3] = int* axis (-1 for flatten)
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void roll_kernel(const T *src, T *dst, int64_t n, int64_t ndim,
                            const int64_t *dims, const int64_t *src_strides,
                            int64_t shift, int64_t axis_size,
                            int64_t axis_stride) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Decompose linear index into outer, axis_index, inner
    int64_t inner = idx % axis_stride;
    int64_t outer = idx / (axis_size * axis_stride);
    int64_t axis_idx = (idx / axis_stride) % axis_size;

    // Apply roll: source index = (axis_idx - shift + axis_size) % axis_size
    int64_t src_axis_idx = (axis_idx - shift + axis_size) % axis_size;
    int64_t src_idx =
        outer * axis_size * axis_stride + src_axis_idx * axis_stride + inner;

    dst[idx] = src[src_idx];
  }
}

// 1D flatten roll kernel
template <typename T>
__global__ void roll_1d_kernel(const T *src, T *dst, int64_t n, int64_t shift) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t src_idx = (idx - shift + n) % n;
    dst[idx] = src[src_idx];
  }
}

extern "C" {

C_Status roll_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);

  if (!out || !x) {
    gpu_set_last_error("roll: null array pointer");
    return C_FAILED;
  }

  // inputs[2] = int* shift, inputs[3] = int* axis — match CPU kernel
  int shift = *static_cast<int *>(inputs[2]);
  int axis = *static_cast<int *>(inputs[3]);

  int64_t ndim = x->ndim;
  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32: {
    const float *src = static_cast<const float *>(x->data);
    float *dst = static_cast<float *>(out->data);
    if (axis == -1) {
      // Flatten roll
      int64_t s = ((shift % n) + n) % n;
      roll_1d_kernel<float><<<blocks, threads>>>(src, dst, n, s);
    } else {
      int ax = axis;
      if (ax < 0)
        ax += ndim;
      int64_t axis_size = x->dims[ax];
      int64_t axis_stride = 1;
      for (int d = ax + 1; d < ndim; ++d) {
        axis_stride *= x->dims[d];
      }
      int64_t s = ((shift % axis_size) + axis_size) % axis_size;
      roll_kernel<float><<<blocks, threads>>>(
          src, dst, n, ndim, x->dims, x->strides, s, axis_size, axis_stride);
    }
    break;
  }
  case INSIGHT_DTYPE_F64: {
    const double *src = static_cast<const double *>(x->data);
    double *dst = static_cast<double *>(out->data);
    if (axis == -1) {
      int64_t s = ((shift % n) + n) % n;
      roll_1d_kernel<double><<<blocks, threads>>>(src, dst, n, s);
    } else {
      int ax = axis;
      if (ax < 0)
        ax += ndim;
      int64_t axis_size = x->dims[ax];
      int64_t axis_stride = 1;
      for (int d = ax + 1; d < ndim; ++d) {
        axis_stride *= x->dims[d];
      }
      int64_t s = ((shift % axis_size) + axis_size) % axis_size;
      roll_kernel<double><<<blocks, threads>>>(
          src, dst, n, ndim, x->dims, x->strides, s, axis_size, axis_stride);
    }
    break;
  }
  case INSIGHT_DTYPE_I32: {
    const int32_t *src = static_cast<const int32_t *>(x->data);
    int32_t *dst = static_cast<int32_t *>(out->data);
    if (axis == -1) {
      int64_t s = ((shift % n) + n) % n;
      roll_1d_kernel<int32_t><<<blocks, threads>>>(src, dst, n, s);
    } else {
      int ax = axis;
      if (ax < 0)
        ax += ndim;
      int64_t axis_size = x->dims[ax];
      int64_t axis_stride = 1;
      for (int d = ax + 1; d < ndim; ++d) {
        axis_stride *= x->dims[d];
      }
      int64_t s = ((shift % axis_size) + axis_size) % axis_size;
      roll_kernel<int32_t><<<blocks, threads>>>(
          src, dst, n, ndim, x->dims, x->strides, s, axis_size, axis_stride);
    }
    break;
  }
  case INSIGHT_DTYPE_I64: {
    const int64_t *src = static_cast<const int64_t *>(x->data);
    int64_t *dst = static_cast<int64_t *>(out->data);
    if (axis == -1) {
      int64_t s = ((shift % n) + n) % n;
      roll_1d_kernel<int64_t><<<blocks, threads>>>(src, dst, n, s);
    } else {
      int ax = axis;
      if (ax < 0)
        ax += ndim;
      int64_t axis_size = x->dims[ax];
      int64_t axis_stride = 1;
      for (int d = ax + 1; d < ndim; ++d) {
        axis_stride *= x->dims[d];
      }
      int64_t s = ((shift % axis_size) + axis_size) % axis_size;
      roll_kernel<int64_t><<<blocks, threads>>>(
          src, dst, n, ndim, x->dims, x->strides, s, axis_size, axis_stride);
    }
    break;
  }
  default:
    gpu_set_last_error("roll: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(roll, INSIGHT_DTYPE_F32, roll_kernel_gpu);
REGISTER_IXUCA_KERNEL(roll, INSIGHT_DTYPE_F64, roll_kernel_gpu);
REGISTER_IXUCA_KERNEL(roll, INSIGHT_DTYPE_I32, roll_kernel_gpu);
REGISTER_IXUCA_KERNEL(roll, INSIGHT_DTYPE_I64, roll_kernel_gpu);
