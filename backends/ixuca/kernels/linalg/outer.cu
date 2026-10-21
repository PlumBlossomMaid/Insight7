// backends/cuda/kernels/linalg/outer.cu
/**
 * @file outer.cu
 * @brief CUDA kernel for outer product of two vectors.
 */

#include "common.cuh"

template <typename T>
__global__ void outer_kernel(T *dst, const T *a, const T *b, int m, int n) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m && j < n) {
    dst[i * n + j] = a[i] * b[j];
  }
}

static C_Status outer_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *a = (InsightArray *)inputs[1];
  InsightArray *b = (InsightArray *)inputs[2];

  if (!out || !a || !b) {
    gpu_set_last_error("outer: null array pointer");
    return C_FAILED;
  }

  int m = (int)a->numel;
  int n = (int)b->numel;

  dim3 threads(16, 16);
  dim3 blocks((n + 15) / 16, (m + 15) / 16);

  if (a->dtype == INSIGHT_DTYPE_F64) {
    outer_kernel<double><<<blocks, threads>>>((double *)out->data,
                                              (const double *)a->data,
                                              (const double *)b->data, m, n);
  } else {
    outer_kernel<float><<<blocks, threads>>>((float *)out->data,
                                             (const float *)a->data,
                                             (const float *)b->data, m, n);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_IXUCA_KERNEL(outer, INSIGHT_DTYPE_F32, outer_kernel_gpu);
REGISTER_IXUCA_KERNEL(outer, INSIGHT_DTYPE_F64, outer_kernel_gpu);
