// backends/cuda/kernels/linalg/common.cuh
/**
 * @file common.cuh
 * @brief Common utilities for CUDA linear algebra kernels.
 *
 * Provides cuBLAS/cuSOLVER handle management and error handling.
 */

#pragma once

#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

// ============================================================================
// cuBLAS / cuSOLVER handle management (thread-local, per-device)
// ============================================================================

struct LinalgHandle {
  cublasHandle_t blas = nullptr;
  cusolverDnHandle_t solver = nullptr;
  int device_id = -1;

  ~LinalgHandle() {
    if (blas)
      cublasDestroy(blas);
    if (solver)
      cusolverDnDestroy(solver);
  }

  void ensure(int dev) {
    if (device_id != dev) {
      if (blas)
        cublasDestroy(blas);
      if (solver)
        cusolverDnDestroy(solver);
      cublasCreate(&blas);
      cusolverDnCreate(&solver);
      device_id = dev;
    }
  }
};

inline LinalgHandle &linalg_get_handle() {
  thread_local LinalgHandle handle;
  return handle;
}

// ============================================================================
// Error handling
// ============================================================================

inline void cublas_check(cublasStatus_t status, const char *op) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    char msg[256];
    const char *err_str = "unknown";
    switch (status) {
    case CUBLAS_STATUS_NOT_INITIALIZED:
      err_str = "not initialized";
      break;
    case CUBLAS_STATUS_ALLOC_FAILED:
      err_str = "alloc failed";
      break;
    case CUBLAS_STATUS_INVALID_VALUE:
      err_str = "invalid value";
      break;
    case CUBLAS_STATUS_ARCH_MISMATCH:
      err_str = "arch mismatch";
      break;
    case CUBLAS_STATUS_MAPPING_ERROR:
      err_str = "mapping error";
      break;
    case CUBLAS_STATUS_EXECUTION_FAILED:
      err_str = "execution failed";
      break;
    case CUBLAS_STATUS_INTERNAL_ERROR:
      err_str = "internal error";
      break;
    default:
      break;
    }
    snprintf(msg, sizeof(msg), "%s: cuBLAS error: %s", op, err_str);
    gpu_set_last_error(msg);
  }
}

inline void cusolver_check(cusolverStatus_t status, const char *op) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    char msg[256];
    snprintf(msg, sizeof(msg), "%s: cuSOLVER error: %d", op, (int)status);
    gpu_set_last_error(msg);
  }
}

// ============================================================================
// Row-major / Column-major conversion helpers
//
// Insight7 uses row-major. cuBLAS/cuSOLVER use column-major.
// Key insight: A_row(m,n) == A_col^T(n,m) in column-major storage.
// So we can often avoid explicit transposition by swapping arguments.
// ============================================================================

// ============================================================================
// Small helper kernels
// ============================================================================

template <typename T> __global__ void linalg_identity_kernel(T *dst, int n) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && j < n) {
    dst[i * n + j] = (i == j) ? T(1) : T(0);
  }
}

template <typename T>
__global__ void linalg_copy_matrix_kernel(T *dst, const T *src, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

template <typename T>
__global__ void linalg_extract_diag_product_kernel(T *dst, const T *lu, int n) {
  // Single thread: compute product of diagonal elements
  T prod = T(1);
  for (int i = 0; i < n; ++i) {
    prod *= lu[i * n + i];
  }
  *dst = prod;
}

// ============================================================================
// Thread/block configuration
// ============================================================================

inline int linalg_threads() { return 256; }

inline int linalg_blocks(int64_t n) {
  return static_cast<int>((n + linalg_threads() - 1) / linalg_threads());
}

// ============================================================================
// Row-major ↔ Column-major transpose via cuBLAS geam
//
// Insight7 uses row-major. cuSOLVER expects column-major.
// A_row(m,n) in memory = A_col^T(n,m) in column-major.
// To get true column-major A(m,n), we need to transpose.
// ============================================================================

inline void linalg_transpose_to_colmajor(cublasHandle_t blas, void *dst,
                                         const void *src, int m, int n,
                                         int32_t dtype) {
  if (dtype == INSIGHT_DTYPE_F64) {
    const double alpha = 1.0, beta = 0.0;
    cublasDgeam(blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha,
                (const double *)src, n, &beta, nullptr, m, (double *)dst, m);
  } else {
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgeam(blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha,
                (const float *)src, n, &beta, nullptr, m, (float *)dst, m);
  }
}

inline void linalg_transpose_from_colmajor(cublasHandle_t blas, void *dst,
                                           const void *src, int m, int n,
                                           int32_t dtype) {
  if (dtype == INSIGHT_DTYPE_F64) {
    const double alpha = 1.0, beta = 0.0;
    cublasDgeam(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha,
                (const double *)src, m, &beta, nullptr, n, (double *)dst, n);
  } else {
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgeam(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha,
                (const float *)src, m, &beta, nullptr, n, (float *)dst, n);
  }
}
