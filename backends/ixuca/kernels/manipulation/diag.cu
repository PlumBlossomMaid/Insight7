// backends/cuda/kernels/manipulation/diag.cu
/**
 * @file diag.cu
 * @brief CUDA kernel for diag operation.
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* output
 *   inputs[1] = InsightArray* input
 *   inputs[2] = int* k (diagonal offset)
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cstring>
#include <cuda_runtime.h>

// Extract diagonal from 2D matrix
template <typename T>
__global__ void diag_extract_kernel(const T *src, T *dst, int64_t diag_len,
                                    int64_t rows, int64_t cols, int64_t k) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < diag_len) {
    int64_t src_idx;
    if (k >= 0) {
      src_idx = idx * cols + (idx + k);
    } else {
      src_idx = (idx - k) * cols + idx;
    }
    dst[idx] = src[src_idx];
  }
}

// Construct diagonal matrix from 1D array
template <typename T>
__global__ void diag_construct_kernel(const T *src, T *dst, int64_t n,
                                      int64_t size, int64_t k) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t row = idx / size;
    int64_t col = idx % size;
    if (col == row + k) {
      int64_t diag_idx = (k >= 0) ? row : col;
      dst[idx] = src[diag_idx];
    } else {
      dst[idx] = T(0);
    }
  }
}

extern "C" {

C_Status diag_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);

  if (!out || !x) {
    gpu_set_last_error("diag: null array pointer");
    return C_FAILED;
  }

  int k = *static_cast<int *>(inputs[2]);

  if (x->ndim == 2) {
    // Extract diagonal from 2D array
    int64_t rows = x->dims[0];
    int64_t cols = x->dims[1];
    int64_t diag_len;
    if (k >= 0) {
      diag_len = (rows < cols - k) ? rows : cols - k;
    } else {
      diag_len = (rows + k < cols) ? rows + k : cols;
    }
    if (diag_len < 0)
      diag_len = 0;

    int threads = 256;
    int blocks = (diag_len + threads - 1) / threads;

    switch (out->dtype) {
    case INSIGHT_DTYPE_F32:
      diag_extract_kernel<float><<<blocks, threads>>>(
          static_cast<const float *>(x->data), static_cast<float *>(out->data),
          diag_len, rows, cols, k);
      break;
    case INSIGHT_DTYPE_F64:
      diag_extract_kernel<double><<<blocks, threads>>>(
          static_cast<const double *>(x->data),
          static_cast<double *>(out->data), diag_len, rows, cols, k);
      break;
    case INSIGHT_DTYPE_I32:
      diag_extract_kernel<int32_t><<<blocks, threads>>>(
          static_cast<const int32_t *>(x->data),
          static_cast<int32_t *>(out->data), diag_len, rows, cols, k);
      break;
    case INSIGHT_DTYPE_I64:
      diag_extract_kernel<int64_t><<<blocks, threads>>>(
          static_cast<const int64_t *>(x->data),
          static_cast<int64_t *>(out->data), diag_len, rows, cols, k);
      break;
    default:
      gpu_set_last_error("diag: unsupported dtype");
      return C_FAILED;
    }
  } else if (x->ndim == 1) {
    // Construct diagonal matrix from 1D array
    int64_t n_elem = x->numel;
    int64_t size = n_elem + (k >= 0 ? k : -k);
    int64_t total = out->numel; // size * size

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    switch (out->dtype) {
    case INSIGHT_DTYPE_F32:
      diag_construct_kernel<float><<<blocks, threads>>>(
          static_cast<const float *>(x->data), static_cast<float *>(out->data),
          total, size, k);
      break;
    case INSIGHT_DTYPE_F64:
      diag_construct_kernel<double><<<blocks, threads>>>(
          static_cast<const double *>(x->data),
          static_cast<double *>(out->data), total, size, k);
      break;
    case INSIGHT_DTYPE_I32:
      diag_construct_kernel<int32_t><<<blocks, threads>>>(
          static_cast<const int32_t *>(x->data),
          static_cast<int32_t *>(out->data), total, size, k);
      break;
    case INSIGHT_DTYPE_I64:
      diag_construct_kernel<int64_t><<<blocks, threads>>>(
          static_cast<const int64_t *>(x->data),
          static_cast<int64_t *>(out->data), total, size, k);
      break;
    default:
      gpu_set_last_error("diag: unsupported dtype");
      return C_FAILED;
    }
  } else {
    gpu_set_last_error("diag: input must be 1D or 2D");
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

REGISTER_IXUCA_KERNEL(diag, INSIGHT_DTYPE_F32, diag_kernel_gpu);
REGISTER_IXUCA_KERNEL(diag, INSIGHT_DTYPE_F64, diag_kernel_gpu);
REGISTER_IXUCA_KERNEL(diag, INSIGHT_DTYPE_I32, diag_kernel_gpu);
REGISTER_IXUCA_KERNEL(diag, INSIGHT_DTYPE_I64, diag_kernel_gpu);
