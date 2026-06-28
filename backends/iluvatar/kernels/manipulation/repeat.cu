// backends/cuda/kernels/manipulation/repeat.cu
/**
 * @file repeat.cu
 * @brief CUDA kernel for repeat operation.
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* output
 *   inputs[1] = InsightArray* input
 *   inputs[2] = int* repeats
 *   inputs[3] = int* axis
 */

#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void repeat_kernel(const T *src, T *dst, int64_t total,
                              int64_t axis_size, int64_t repeat_count,
                              int64_t inner_stride) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    // Decompose output linear index into outer, rep, axis_idx, inner
    int64_t inner = idx % inner_stride;
    int64_t tmp = idx / inner_stride;
    int64_t axis_idx = tmp % (axis_size * repeat_count);
    int64_t outer = tmp / (axis_size * repeat_count);

    // Map output axis index to input axis index
    int64_t src_axis_idx = axis_idx / repeat_count;

    // Compute source offset
    int64_t src_idx =
        outer * axis_size * inner_stride + src_axis_idx * inner_stride + inner;

    dst[idx] = src[src_idx];
  }
}

extern "C" {

C_Status repeat_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);

  if (!out || !x) {
    gpu_set_last_error("repeat: null array pointer");
    return C_FAILED;
  }

  // inputs[2] = repeats (int*), inputs[3] = axis (int*) — match CPU kernel
  int repeats = *static_cast<int *>(inputs[2]);
  int axis = *static_cast<int *>(inputs[3]);
  int64_t ndim = x->ndim;
  if (axis < 0)
    axis += ndim;

  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int64_t axis_size = x->dims[axis];
  int64_t inner_stride = 1;
  for (int d = axis + 1; d < ndim; ++d) {
    inner_stride *= x->dims[d];
  }

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    repeat_kernel<float><<<blocks, threads>>>(
        static_cast<const float *>(x->data), static_cast<float *>(out->data), n,
        axis_size, repeats, inner_stride);
    break;
  case INSIGHT_DTYPE_F64:
    repeat_kernel<double><<<blocks, threads>>>(
        static_cast<const double *>(x->data), static_cast<double *>(out->data),
        n, axis_size, repeats, inner_stride);
    break;
  case INSIGHT_DTYPE_I32:
    repeat_kernel<int32_t><<<blocks, threads>>>(
        static_cast<const int32_t *>(x->data),
        static_cast<int32_t *>(out->data), n, axis_size, repeats, inner_stride);
    break;
  case INSIGHT_DTYPE_I64:
    repeat_kernel<int64_t><<<blocks, threads>>>(
        static_cast<const int64_t *>(x->data),
        static_cast<int64_t *>(out->data), n, axis_size, repeats, inner_stride);
    break;
  default:
    gpu_set_last_error("repeat: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(repeat, INSIGHT_DTYPE_F32, repeat_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(repeat, INSIGHT_DTYPE_F64, repeat_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(repeat, INSIGHT_DTYPE_I32, repeat_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(repeat, INSIGHT_DTYPE_I64, repeat_kernel_gpu);
