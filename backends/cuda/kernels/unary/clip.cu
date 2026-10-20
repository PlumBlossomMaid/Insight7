// backends/cuda/kernels/unary/clip.cu
/**
 * @file clip.cu
 * @brief CUDA kernel for value clipping.
 *
 * Computes clip(x) = min(max(x, lo), hi) elementwise.
 * Bounds are passed as double* in inputs[1] (lo) and inputs[2] (hi).
 * Supports integer and floating point types.
 */

#include "../../registry/cuda_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cuda_runtime.h>

template <typename T>
__global__ void clip_kernel(const T *x, T *out, int64_t n, int ndim,
                            const int64_t *dims, const int64_t *x_strides,
                            const int64_t *out_strides, int64_t x_offset,
                            int64_t out_offset, T lo, T hi) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t off_x = x_offset + unary_offset(idx, ndim, dims, x_strides);
    int64_t off_out = out_offset + unary_offset(idx, ndim, dims, out_strides);
    T v = x[off_x];
    out[off_out] = v < lo ? lo : (v > hi ? hi : v);
  }
}

extern "C" {

C_Status clip_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = static_cast<InsightArray *>(inputs[0]);
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!x || !out) {
    gpu_set_last_error("clip: null array pointer");
    return C_FAILED;
  }
  if (!inputs[1] || !inputs[2]) {
    gpu_set_last_error("clip: bounds are null");
    return C_FAILED;
  }

  double lo_host = *static_cast<double *>(inputs[1]);
  double hi_host = *static_cast<double *>(inputs[2]);

  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int ndim = out->ndim;
  int threads = unary_threads();
  int blocks = unary_blocks(n);

  int64_t *d_dims, *d_x_strides, *d_out_strides;
  cudaMalloc(&d_dims, ndim * sizeof(int64_t));
  cudaMalloc(&d_x_strides, ndim * sizeof(int64_t));
  cudaMalloc(&d_out_strides, ndim * sizeof(int64_t));
  cudaMemcpy(d_dims, out->dims, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x_strides, x->strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_strides, out->strides, ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  int64_t x_offset = x->offset;
  int64_t out_offset = out->offset;

  switch (out->dtype) {
  case INSIGHT_DTYPE_I8:
    clip_kernel<int8_t><<<blocks, threads>>>(
        static_cast<const int8_t *>(x->data), static_cast<int8_t *>(out->data),
        n, ndim, d_dims, d_x_strides, d_out_strides, x_offset, out_offset,
        static_cast<int8_t>(lo_host), static_cast<int8_t>(hi_host));
    break;
  case INSIGHT_DTYPE_I16:
    clip_kernel<int16_t><<<blocks, threads>>>(
        static_cast<const int16_t *>(x->data),
        static_cast<int16_t *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides, x_offset, out_offset, static_cast<int16_t>(lo_host),
        static_cast<int16_t>(hi_host));
    break;
  case INSIGHT_DTYPE_I32:
    clip_kernel<int32_t><<<blocks, threads>>>(
        static_cast<const int32_t *>(x->data),
        static_cast<int32_t *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides, x_offset, out_offset, static_cast<int32_t>(lo_host),
        static_cast<int32_t>(hi_host));
    break;
  case INSIGHT_DTYPE_I64:
    clip_kernel<int64_t><<<blocks, threads>>>(
        static_cast<const int64_t *>(x->data),
        static_cast<int64_t *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides, x_offset, out_offset, static_cast<int64_t>(lo_host),
        static_cast<int64_t>(hi_host));
    break;
  case INSIGHT_DTYPE_U8:
    clip_kernel<uint8_t><<<blocks, threads>>>(
        static_cast<const uint8_t *>(x->data),
        static_cast<uint8_t *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides, x_offset, out_offset, static_cast<uint8_t>(lo_host),
        static_cast<uint8_t>(hi_host));
    break;
  case INSIGHT_DTYPE_U16:
    clip_kernel<uint16_t><<<blocks, threads>>>(
        static_cast<const uint16_t *>(x->data),
        static_cast<uint16_t *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides, x_offset, out_offset, static_cast<uint16_t>(lo_host),
        static_cast<uint16_t>(hi_host));
    break;
  case INSIGHT_DTYPE_U32:
    clip_kernel<uint32_t><<<blocks, threads>>>(
        static_cast<const uint32_t *>(x->data),
        static_cast<uint32_t *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides, x_offset, out_offset, static_cast<uint32_t>(lo_host),
        static_cast<uint32_t>(hi_host));
    break;
  case INSIGHT_DTYPE_U64:
    clip_kernel<uint64_t><<<blocks, threads>>>(
        static_cast<const uint64_t *>(x->data),
        static_cast<uint64_t *>(out->data), n, ndim, d_dims, d_x_strides,
        d_out_strides, x_offset, out_offset, static_cast<uint64_t>(lo_host),
        static_cast<uint64_t>(hi_host));
    break;
  case INSIGHT_DTYPE_F32:
    clip_kernel<float><<<blocks, threads>>>(
        static_cast<const float *>(x->data), static_cast<float *>(out->data), n,
        ndim, d_dims, d_x_strides, d_out_strides, x_offset, out_offset,
        static_cast<float>(lo_host), static_cast<float>(hi_host));
    break;
  case INSIGHT_DTYPE_F64:
    clip_kernel<double><<<blocks, threads>>>(
        static_cast<const double *>(x->data), static_cast<double *>(out->data),
        n, ndim, d_dims, d_x_strides, d_out_strides, x_offset, out_offset,
        lo_host, hi_host);
    break;
  default:
    cudaFree(d_dims);
    cudaFree(d_x_strides);
    cudaFree(d_out_strides);
    gpu_set_last_error("clip: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  cudaFree(d_dims);
  cudaFree(d_x_strides);
  cudaFree(d_out_strides);

  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_GPU_KERNEL(clip, INSIGHT_DTYPE_I8, clip_kernel_gpu);
REGISTER_GPU_KERNEL(clip, INSIGHT_DTYPE_I16, clip_kernel_gpu);
REGISTER_GPU_KERNEL(clip, INSIGHT_DTYPE_I32, clip_kernel_gpu);
REGISTER_GPU_KERNEL(clip, INSIGHT_DTYPE_I64, clip_kernel_gpu);
REGISTER_GPU_KERNEL(clip, INSIGHT_DTYPE_U8, clip_kernel_gpu);
REGISTER_GPU_KERNEL(clip, INSIGHT_DTYPE_U16, clip_kernel_gpu);
REGISTER_GPU_KERNEL(clip, INSIGHT_DTYPE_U32, clip_kernel_gpu);
REGISTER_GPU_KERNEL(clip, INSIGHT_DTYPE_U64, clip_kernel_gpu);
REGISTER_GPU_KERNEL(clip, INSIGHT_DTYPE_F32, clip_kernel_gpu);
REGISTER_GPU_KERNEL(clip, INSIGHT_DTYPE_F64, clip_kernel_gpu);