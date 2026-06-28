// backends/cuda/kernels/linalg/solve_triangular.cu
/**
 * @file solve_triangular.cu
 * @brief CUDA kernel for triangular solve using cuBLAS trsm.
 *
 * Solves A * X = B where A is triangular.
 * lower=1 means A is lower triangular, lower=0 means upper.
 * Row-major lower triangular = column-major upper triangular (transpose).
 */

#include "common.cuh"

static C_Status solve_triangular_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *A = (InsightArray *)inputs[1];
  InsightArray *B = (InsightArray *)inputs[2];
  int lower = *(int *)inputs[3];

  if (!out || !A || !B) {
    gpu_set_last_error("solve_triangular: null array pointer");
    return C_FAILED;
  }

  auto &handle = linalg_get_handle();
  handle.ensure(A->device_id);

  int n = (int)A->dims[0];
  int nrhs = (B->ndim == 1) ? 1 : (int)B->dims[1];

  // Copy B to output
  size_t b_bytes = (size_t)B->numel * insight_dtype_size(B->dtype);
  cudaMemcpy(out->data, B->data, b_bytes, cudaMemcpyDeviceToDevice);

  // cuBLAS trsm: solves op(A) * X = alpha * B, in-place on B.
  // cuBLAS uses column-major. Row-major A(m,n) = column-major A^T(n,m).
  // Row-major lower A → column-major A^T is upper → CUBLAS_FILL_MODE_UPPER.
  cublasFillMode_t uplo =
      lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
  cublasOperation_t op = CUBLAS_OP_T; // because row-major = col-major^T

  if (A->dtype == INSIGHT_DTYPE_F64) {
    const double alpha = 1.0;
    if (nrhs == 1) {
      cublasDtrsv(handle.blas, uplo, op, diag, n, (const double *)A->data, n,
                  (double *)out->data, 1);
    } else {
      cublasDtrsm(handle.blas, CUBLAS_SIDE_LEFT, uplo, op, diag, n, nrhs,
                  &alpha, (const double *)A->data, n, (double *)out->data, n);
    }
  } else {
    const float alpha = 1.0f;
    if (nrhs == 1) {
      cublasStrsv(handle.blas, uplo, op, diag, n, (const float *)A->data, 1,
                  (float *)out->data, 1);
    } else {
      cublasStrsm(handle.blas, CUBLAS_SIDE_LEFT, uplo, op, diag, n, nrhs,
                  &alpha, (const float *)A->data, n, (float *)out->data, n);
    }
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_ILUVATAR_KERNEL(solve_triangular, INSIGHT_DTYPE_F32,
                    solve_triangular_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(solve_triangular, INSIGHT_DTYPE_F64,
                    solve_triangular_kernel_gpu);
