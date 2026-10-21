// backends/cuda/kernels/linalg/matmul.cu
/**
 * @file matmul.cu
 * @brief CUDA kernel for matrix multiplication using cuBLAS.
 *
 * Supports vector-vector, matrix-vector, matrix-matrix, and batched matmul.
 */

#include "common.cuh"

static C_Status matmul_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *a = (InsightArray *)inputs[1];
  InsightArray *b = (InsightArray *)inputs[2];

  if (!out || !a || !b) {
    gpu_set_last_error("matmul: null array pointer");
    return C_FAILED;
  }

  auto &handle = linalg_get_handle();
  handle.ensure(a->device_id);

  int ndim_a = a->ndim;
  int ndim_b = b->ndim;
  int32_t dtype = out->dtype;

  // Case 1: vector * vector -> scalar (dot product)
  if (ndim_a == 1 && ndim_b == 1) {
    int n = (int)a->numel;
    if (dtype == INSIGHT_DTYPE_F64) {
      double result_val;
      cublasDdot(handle.blas, n, (const double *)a->data, 1,
                 (const double *)b->data, 1, &result_val);
      cudaMemcpy((double *)out->data, &result_val, sizeof(double),
                 cudaMemcpyHostToDevice);
    } else {
      float result_val;
      cublasSdot(handle.blas, n, (const float *)a->data, 1,
                 (const float *)b->data, 1, &result_val);
      cudaMemcpy((float *)out->data, &result_val, sizeof(float),
                 cudaMemcpyHostToDevice);
    }
    return C_SUCCESS;
  }

  // Case 2: matrix * vector
  if (ndim_a == 2 && ndim_b == 1) {
    int m = (int)a->dims[0];
    int n = (int)a->dims[1];
    if (dtype == INSIGHT_DTYPE_F64) {
      const double alpha = 1.0, beta = 0.0;
      // Row-major A(m,n) * x(n) = y(m)
      // cuBLAS column-major: treat A as A_col^T(n,m), use CUBLAS_OP_T
      cublasDgemv(handle.blas, CUBLAS_OP_T, n, m, &alpha,
                  (const double *)a->data, n, (const double *)b->data, 1, &beta,
                  (double *)out->data, 1);
    } else {
      const float alpha = 1.0f, beta = 0.0f;
      cublasSgemv(handle.blas, CUBLAS_OP_T, n, m, &alpha,
                  (const float *)a->data, n, (const float *)b->data, 1, &beta,
                  (float *)out->data, 1);
    }
    return C_SUCCESS;
  }

  // Case 3: vector * matrix
  if (ndim_a == 1 && ndim_b == 2) {
    int m = (int)b->dims[0];
    int n = (int)b->dims[1];
    if (dtype == INSIGHT_DTYPE_F64) {
      const double alpha = 1.0, beta = 0.0;
      cublasDgemv(handle.blas, CUBLAS_OP_T, m, n, &alpha,
                  (const double *)b->data, m, (const double *)a->data, 1, &beta,
                  (double *)out->data, 1);
    } else {
      const float alpha = 1.0f, beta = 0.0f;
      cublasSgemv(handle.blas, CUBLAS_OP_T, m, n, &alpha,
                  (const float *)b->data, m, (const float *)a->data, 1, &beta,
                  (float *)out->data, 1);
    }
    return C_SUCCESS;
  }

  // Case 4: matrix * matrix (2D)
  if (ndim_a == 2 && ndim_b == 2) {
    int m = (int)a->dims[0];
    int k = (int)a->dims[1];
    int n = (int)b->dims[1];
    if (dtype == INSIGHT_DTYPE_F64) {
      const double alpha = 1.0, beta = 0.0;
      // Row-major: C(m,n) = A(m,k) * B(k,n)
      // cuBLAS column-major: C_col(n,m) = B_col^T(n,k) * A_col^T(k,m)
      // i.e., cublasDgemm(N, N, n, m, k, alpha, B, n, A, k, beta, C, n)
      cublasDgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                  (const double *)b->data, n, (const double *)a->data, k, &beta,
                  (double *)out->data, n);
    } else {
      const float alpha = 1.0f, beta = 0.0f;
      cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                  (const float *)b->data, n, (const float *)a->data, k, &beta,
                  (float *)out->data, n);
    }
    return C_SUCCESS;
  }

  // Case 5: batched matrix multiplication (ND)
  {
    int ndim = out->ndim;
    int batch_dims = ndim - 2;
    int64_t batch_size = 1;
    for (int i = 0; i < batch_dims; ++i) {
      batch_size *= out->dims[i];
    }
    int m = (int)out->dims[batch_dims];
    int n = (int)out->dims[batch_dims + 1];
    int k_a = (ndim_a >= 2) ? (int)a->dims[ndim_a - 1] : 1;
    int k = k_a;

    int64_t a_stride = (int64_t)m * k;
    int64_t b_stride = (int64_t)k * n;
    int64_t c_stride = (int64_t)m * n;

    if (dtype == INSIGHT_DTYPE_F64) {
      const double *a_data = (const double *)a->data;
      const double *b_data = (const double *)b->data;
      double *c_data = (double *)out->data;
      const double alpha = 1.0, beta = 0.0;
      for (int64_t batch = 0; batch < batch_size; ++batch) {
        cublasDgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                    b_data + batch * b_stride, n, a_data + batch * a_stride, k,
                    &beta, c_data + batch * c_stride, n);
      }
    } else {
      const float *a_data = (const float *)a->data;
      const float *b_data = (const float *)b->data;
      float *c_data = (float *)out->data;
      const float alpha = 1.0f, beta = 0.0f;
      for (int64_t batch = 0; batch < batch_size; ++batch) {
        cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                    b_data + batch * b_stride, n, a_data + batch * a_stride, k,
                    &beta, c_data + batch * c_stride, n);
      }
    }
  }

  return C_SUCCESS;
}

REGISTER_IXUCA_KERNEL(matmul, INSIGHT_DTYPE_F32, matmul_kernel_gpu);
REGISTER_IXUCA_KERNEL(matmul, INSIGHT_DTYPE_F64, matmul_kernel_gpu);
