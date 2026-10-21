// backends/cuda/kernels/linalg/svd.cu
/**
 * @file svd.cu
 * @brief CUDA kernel for SVD using cuSOLVER.
 *
 * Row-major A(m,n): transpose to column-major A_col(m,n),
 * call gesvd(m, n, A_col, m, S, U_col, ldu, VT_col, ldvt),
 * then transpose U_col and VT_col back to row-major.
 *
 * cuSOLVER gesvd computes: A_col = U * diag(S) * V^T  (column-major)
 * Output:
 *   U_col(m, u_cols) col-major -> U_row(m, u_cols) via transpose
 *   VT_col(vt_rows, n) col-major -> VT_row(vt_rows, n) via transpose
 *
 * full=0 (economy): u_cols = vt_rows = min(m,n), U(m,k), VT(k,n)
 * full=1 (full):    u_cols = m, vt_rows = n, U(m,m), VT(n,n)
 */

#include "common.cuh"

static C_Status svd_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *U = (InsightArray *)outputs[0];
  InsightArray *S = (InsightArray *)outputs[1];
  InsightArray *VT = (InsightArray *)outputs[2];
  InsightArray *x = (InsightArray *)inputs[3];
  int full = *(int *)inputs[4];

  if (!U || !S || !VT || !x) {
    gpu_set_last_error("svd: null array pointer");
    return C_FAILED;
  }

  auto &handle = linalg_get_handle();
  handle.ensure(x->device_id);

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];
  int min_mn = (m < n) ? m : n;

  size_t elem_size = insight_dtype_size(x->dtype);
  size_t data_bytes = (size_t)m * n * elem_size;

  // Dimensions for U and VT in column-major
  char jobu, jobvt;
  int u_cols, vt_rows; // logical column-major shapes
  int ldu, ldvt;

  if (full) {
    jobu = 'A';
    jobvt = 'A';
    u_cols = m;  // U is (m, m) column-major
    vt_rows = n; // VT is (n, n) column-major
    ldu = m;
    ldvt = n;
  } else {
    jobu = 'S';
    jobvt = 'S';
    u_cols = min_mn;  // U is (m, min_mn) column-major
    vt_rows = min_mn; // VT is (min_mn, n) column-major
    ldu = m;
    ldvt = min_mn;
  }

  // Allocate workspace
  void *A_col = nullptr;
  cudaMalloc(&A_col, data_bytes);
  // U_col: (m, u_cols) column-major
  void *U_col = nullptr;
  cudaMalloc(&U_col, (size_t)m * u_cols * elem_size);
  // VT_col: (vt_rows, n) column-major
  void *VT_col = nullptr;
  cudaMalloc(&VT_col, (size_t)vt_rows * n * elem_size);

  int *d_info = nullptr;
  cudaMalloc(&d_info, sizeof(int));

  // Step 1: Transpose A_row(m,n) -> A_col(m,n) column-major
  linalg_transpose_to_colmajor(handle.blas, A_col, x->data, m, n, x->dtype);

  // Step 2: gesvd on A_col(m,n) column-major
  // Computes A = U * diag(S) * VT in column-major
  if (x->dtype == INSIGHT_DTYPE_F64) {
    int lwork = 0;
    cusolverDnDgesvd_bufferSize(handle.solver, m, n, &lwork);
    double *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(double));
    cusolverDnDgesvd(handle.solver, jobu, jobvt, m, n, (double *)A_col, m,
                     (double *)S->data, (double *)U_col, ldu, (double *)VT_col,
                     ldvt, d_work, lwork, nullptr, d_info);
    cudaFree(d_work);
  } else {
    int lwork = 0;
    cusolverDnSgesvd_bufferSize(handle.solver, m, n, &lwork);
    float *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(float));
    cusolverDnSgesvd(handle.solver, jobu, jobvt, m, n, (float *)A_col, m,
                     (float *)S->data, (float *)U_col, ldu, (float *)VT_col,
                     ldvt, d_work, lwork, nullptr, d_info);
    cudaFree(d_work);
  }

  // Step 3: Transpose U_col(m, u_cols) -> U_row(m, u_cols)
  linalg_transpose_from_colmajor(handle.blas, U->data, U_col, m, u_cols,
                                 x->dtype);

  // Step 4: Transpose VT_col(vt_rows, n) -> VT_row(vt_rows, n)
  linalg_transpose_from_colmajor(handle.blas, VT->data, VT_col, vt_rows, n,
                                 x->dtype);

  cudaFree(A_col);
  cudaFree(U_col);
  cudaFree(VT_col);
  cudaFree(d_info);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_IXUCA_KERNEL(svd, INSIGHT_DTYPE_F32, svd_kernel_gpu);
REGISTER_IXUCA_KERNEL(svd, INSIGHT_DTYPE_F64, svd_kernel_gpu);
