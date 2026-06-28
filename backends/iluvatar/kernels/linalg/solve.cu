// backends/cuda/kernels/linalg/solve.cu
/**
 * @file solve.cu
 * @brief CUDA kernel for solving linear system AX = B using cuSOLVER.
 */

#include "common.cuh"

static C_Status solve_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *A = (InsightArray *)inputs[1];
  InsightArray *B = (InsightArray *)inputs[2];

  if (!out || !A || !B) {
    gpu_set_last_error("solve: null array pointer");
    return C_FAILED;
  }

  auto &handle = linalg_get_handle();
  handle.ensure(A->device_id);

  int n = (int)A->dims[0];
  int nrhs = (B->ndim == 1) ? 1 : (int)B->dims[1];

  size_t a_bytes = (size_t)n * n * insight_dtype_size(A->dtype);
  size_t b_bytes = (size_t)B->numel * insight_dtype_size(B->dtype);

  // Allocate column-major workspace for A and B
  void *A_col = nullptr;
  void *B_col = nullptr;
  cudaMalloc(&A_col, a_bytes);
  cudaMalloc(&B_col, b_bytes);

  // Transpose A and B from row-major to column-major
  linalg_transpose_to_colmajor(handle.blas, A_col, A->data, n, n, A->dtype);
  linalg_transpose_to_colmajor(handle.blas, B_col, B->data, n, nrhs, B->dtype);

  int *d_ipiv = nullptr;
  int *d_info = nullptr;
  cudaMalloc(&d_ipiv, n * sizeof(int));
  cudaMalloc(&d_info, sizeof(int));

  if (A->dtype == INSIGHT_DTYPE_F64) {
    int lwork = 0;
    cusolverDnDgetrf_bufferSize(handle.solver, n, n, (double *)A_col, n,
                                &lwork);
    double *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(double));
    cusolverDnDgetrf(handle.solver, n, n, (double *)A_col, n, d_work, d_ipiv,
                     d_info);
    cusolverDnDgetrs(handle.solver, CUBLAS_OP_N, n, nrhs, (double *)A_col, n,
                     d_ipiv, (double *)B_col, n, d_info);
    cudaFree(d_work);
  } else {
    int lwork = 0;
    cusolverDnSgetrf_bufferSize(handle.solver, n, n, (float *)A_col, n, &lwork);
    float *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(float));
    cusolverDnSgetrf(handle.solver, n, n, (float *)A_col, n, d_work, d_ipiv,
                     d_info);
    cusolverDnSgetrs(handle.solver, CUBLAS_OP_N, n, nrhs, (float *)A_col, n,
                     d_ipiv, (float *)B_col, n, d_info);
    cudaFree(d_work);
  }

  // Transpose result from column-major back to row-major into output
  linalg_transpose_from_colmajor(handle.blas, out->data, B_col, n, nrhs,
                                 B->dtype);

  cudaFree(A_col);
  cudaFree(B_col);
  cudaFree(d_ipiv);
  cudaFree(d_info);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_ILUVATAR_KERNEL(solve, INSIGHT_DTYPE_F32, solve_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(solve, INSIGHT_DTYPE_F64, solve_kernel_gpu);
