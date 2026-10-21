// backends/cuda/kernels/creation/eye.cu
/**
 * @file eye.cu
 * @brief CUDA kernel for the eye operation.
 *
 * Creates an identity matrix (2D array with ones on the diagonal).
 * Supports all numeric dtypes including complex types.
 * Uses two kernels: one to zero-fill, one to set diagonal elements.
 */
#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <string>

/**
 * @brief CUDA kernel to zero-fill an array.
 *
 * Each thread writes zero to one element.
 *
 * @tparam T Element type
 * @param dst Output array
 * @param n Total number of elements (rows * cols)
 */
template <typename T> __global__ void eye_zero_kernel(T *dst, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = T(0);
  }
}

/**
 * @brief CUDA kernel to set diagonal elements to one.
 *
 * Each thread handles one row. For row i, sets element (i, i+k) to 1.
 *
 * @tparam T Element type
 * @param dst Output array
 * @param rows Number of rows
 * @param cols Number of columns
 * @param k Diagonal offset (0 = main diagonal, >0 = upper, <0 = lower)
 */
template <typename T>
__global__ void eye_diag_kernel(T *dst, int64_t rows, int64_t cols, int64_t k) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows) {
    int64_t j = i + k;
    if (j >= 0 && j < cols) {
      dst[i * cols + j] = T(1);
    }
  }
}

/**
 * @brief CUDA kernel to zero-fill a complex float array.
 *
 * @param dst Output array
 * @param n Total number of elements
 */
__global__ void eye_zero_c32_kernel(cuFloatComplex *dst, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = make_cuFloatComplex(0.0f, 0.0f);
  }
}

/**
 * @brief CUDA kernel to set diagonal elements to one (complex float).
 *
 * @param dst Output array
 * @param rows Number of rows
 * @param cols Number of columns
 * @param k Diagonal offset
 */
__global__ void eye_diag_c32_kernel(cuFloatComplex *dst, int64_t rows,
                                    int64_t cols, int64_t k) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows) {
    int64_t j = i + k;
    if (j >= 0 && j < cols) {
      dst[i * cols + j] = make_cuFloatComplex(1.0f, 0.0f);
    }
  }
}

/**
 * @brief CUDA kernel to zero-fill a complex double array.
 *
 * @param dst Output array
 * @param n Total number of elements
 */
__global__ void eye_zero_c64_kernel(cuDoubleComplex *dst, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = make_cuDoubleComplex(0.0, 0.0);
  }
}

/**
 * @brief CUDA kernel to set diagonal elements to one (complex double).
 *
 * @param dst Output array
 * @param rows Number of rows
 * @param cols Number of columns
 * @param k Diagonal offset
 */
__global__ void eye_diag_c64_kernel(cuDoubleComplex *dst, int64_t rows,
                                    int64_t cols, int64_t k) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows) {
    int64_t j = i + k;
    if (j >= 0 && j < cols) {
      dst[i * cols + j] = make_cuDoubleComplex(1.0, 0.0);
    }
  }
}

extern "C" {

/**
 * @brief GPU entry point for the eye kernel.
 *
 * Creates an identity matrix by first zero-filling the output array,
 * then setting diagonal elements to one.
 *
 * @param inputs  [0] = InsightArray* output, [1] = int64_t* k (diagonal offset)
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status eye_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("eye_kernel_gpu: output array is null");
    return C_FAILED;
  }
  if (!inputs[1]) {
    gpu_set_last_error("eye_kernel_gpu: k parameter is null");
    return C_FAILED;
  }

  int64_t k = *static_cast<int64_t *>(inputs[1]);
  int64_t rows = out->dims[0];
  int64_t cols = out->dims[1];
  int64_t total = rows * cols;
  if (total == 0)
    return C_SUCCESS;
  int32_t dtype = out->dtype;

  int threads = creation_threads();
  int blocks_zero = creation_blocks(total);
  int blocks_diag = creation_blocks(rows);

  switch (dtype) {
  case INSIGHT_DTYPE_BOOL: {
    eye_zero_kernel<<<blocks_zero, threads>>>(static_cast<bool *>(out->data),
                                              total);
    eye_diag_kernel<<<blocks_diag, threads>>>(static_cast<bool *>(out->data),
                                              rows, cols, k);
    break;
  }
  case INSIGHT_DTYPE_U8: {
    eye_zero_kernel<<<blocks_zero, threads>>>(static_cast<uint8_t *>(out->data),
                                              total);
    eye_diag_kernel<<<blocks_diag, threads>>>(static_cast<uint8_t *>(out->data),
                                              rows, cols, k);
    break;
  }
  case INSIGHT_DTYPE_I8: {
    eye_zero_kernel<<<blocks_zero, threads>>>(static_cast<int8_t *>(out->data),
                                              total);
    eye_diag_kernel<<<blocks_diag, threads>>>(static_cast<int8_t *>(out->data),
                                              rows, cols, k);
    break;
  }
  case INSIGHT_DTYPE_I16: {
    eye_zero_kernel<<<blocks_zero, threads>>>(static_cast<int16_t *>(out->data),
                                              total);
    eye_diag_kernel<<<blocks_diag, threads>>>(static_cast<int16_t *>(out->data),
                                              rows, cols, k);
    break;
  }
  case INSIGHT_DTYPE_I32: {
    eye_zero_kernel<<<blocks_zero, threads>>>(static_cast<int32_t *>(out->data),
                                              total);
    eye_diag_kernel<<<blocks_diag, threads>>>(static_cast<int32_t *>(out->data),
                                              rows, cols, k);
    break;
  }
  case INSIGHT_DTYPE_I64: {
    eye_zero_kernel<<<blocks_zero, threads>>>(static_cast<int64_t *>(out->data),
                                              total);
    eye_diag_kernel<<<blocks_diag, threads>>>(static_cast<int64_t *>(out->data),
                                              rows, cols, k);
    break;
  }
  case INSIGHT_DTYPE_U16: {
    eye_zero_kernel<<<blocks_zero, threads>>>(
        static_cast<uint16_t *>(out->data), total);
    eye_diag_kernel<<<blocks_diag, threads>>>(
        static_cast<uint16_t *>(out->data), rows, cols, k);
    break;
  }
  case INSIGHT_DTYPE_U32: {
    eye_zero_kernel<<<blocks_zero, threads>>>(
        static_cast<uint32_t *>(out->data), total);
    eye_diag_kernel<<<blocks_diag, threads>>>(
        static_cast<uint32_t *>(out->data), rows, cols, k);
    break;
  }
  case INSIGHT_DTYPE_U64: {
    eye_zero_kernel<<<blocks_zero, threads>>>(
        static_cast<uint64_t *>(out->data), total);
    eye_diag_kernel<<<blocks_diag, threads>>>(
        static_cast<uint64_t *>(out->data), rows, cols, k);
    break;
  }
  case INSIGHT_DTYPE_F32: {
    eye_zero_kernel<<<blocks_zero, threads>>>(static_cast<float *>(out->data),
                                              total);
    eye_diag_kernel<<<blocks_diag, threads>>>(static_cast<float *>(out->data),
                                              rows, cols, k);
    break;
  }
  case INSIGHT_DTYPE_F64: {
    eye_zero_kernel<<<blocks_zero, threads>>>(static_cast<double *>(out->data),
                                              total);
    eye_diag_kernel<<<blocks_diag, threads>>>(static_cast<double *>(out->data),
                                              rows, cols, k);
    break;
  }
  case INSIGHT_DTYPE_C32: {
    eye_zero_c32_kernel<<<blocks_zero, threads>>>(
        static_cast<cuFloatComplex *>(out->data), total);
    eye_diag_c32_kernel<<<blocks_diag, threads>>>(
        static_cast<cuFloatComplex *>(out->data), rows, cols, k);
    break;
  }
  case INSIGHT_DTYPE_C64: {
    eye_zero_c64_kernel<<<blocks_zero, threads>>>(
        static_cast<cuDoubleComplex *>(out->data), total);
    eye_diag_c64_kernel<<<blocks_diag, threads>>>(
        static_cast<cuDoubleComplex *>(out->data), rows, cols, k);
    break;
  }
  default:
    gpu_set_last_error(
        ("eye_kernel_gpu: unsupported dtype " + std::to_string(dtype)).c_str());
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

// Register for all supported types
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_BOOL, eye_kernel_gpu);
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_U8, eye_kernel_gpu);
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_I8, eye_kernel_gpu);
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_I16, eye_kernel_gpu);
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_I32, eye_kernel_gpu);
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_I64, eye_kernel_gpu);
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_U16, eye_kernel_gpu);
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_U32, eye_kernel_gpu);
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_U64, eye_kernel_gpu);
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_F32, eye_kernel_gpu);
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_F64, eye_kernel_gpu);
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_C32, eye_kernel_gpu);
REGISTER_IXUCA_KERNEL(eye, INSIGHT_DTYPE_C64, eye_kernel_gpu);
