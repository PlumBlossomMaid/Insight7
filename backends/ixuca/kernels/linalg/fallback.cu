// backends/cuda/kernels/linalg/fallback.cu
/**
 * @file fallback.cu
 * @brief CUDA kernels that fall back to CPU implementation.
 *
 * These operations return C_FALLBACK to trigger the framework's automatic
 * CPU fallback with data transfer. This is used for:
 * - Operations with complex row-major/column-major conversion (Cholesky, QR,
 * LU, LQ)
 * - Operations requiring complex types or multiple decomposition steps
 * - Operations not performance-critical for the competition
 */

#include "common.cuh"

// Cholesky: complex row-major ↔ column-major conversion with potrf
static C_Status cholesky_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(cholesky, INSIGHT_DTYPE_F32, cholesky_kernel_gpu);
REGISTER_IXUCA_KERNEL(cholesky, INSIGHT_DTYPE_F64, cholesky_kernel_gpu);

// Cholesky solve: depends on Cholesky
static C_Status cholesky_solve_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(cholesky_solve, INSIGHT_DTYPE_F32,
                         cholesky_solve_kernel_gpu);
REGISTER_IXUCA_KERNEL(cholesky_solve, INSIGHT_DTYPE_F64,
                         cholesky_solve_kernel_gpu);

// QR decomposition: complex Householder + orgqr + row-major conversion
static C_Status qr_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(qr, INSIGHT_DTYPE_F32, qr_kernel_gpu);
REGISTER_IXUCA_KERNEL(qr, INSIGHT_DTYPE_F64, qr_kernel_gpu);

// LQ decomposition: depends on QR of transpose
static C_Status lq_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(lq, INSIGHT_DTYPE_F32, lq_kernel_gpu);
REGISTER_IXUCA_KERNEL(lq, INSIGHT_DTYPE_F64, lq_kernel_gpu);

// LU decomposition: complex row-major ↔ column-major conversion
static C_Status lu_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(lu, INSIGHT_DTYPE_F32, lu_kernel_gpu);
REGISTER_IXUCA_KERNEL(lu, INSIGHT_DTYPE_F64, lu_kernel_gpu);

// LU unpack: depends on LU
static C_Status lu_unpack_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(lu_unpack, INSIGHT_DTYPE_F32, lu_unpack_kernel_gpu);
REGISTER_IXUCA_KERNEL(lu_unpack, INSIGHT_DTYPE_F64, lu_unpack_kernel_gpu);

// cond: requires norm + inv, matrix 1-norm/inf-norm not on GPU
static C_Status cond_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(cond, INSIGHT_DTYPE_F32, cond_kernel_gpu);
REGISTER_IXUCA_KERNEL(cond, INSIGHT_DTYPE_F64, cond_kernel_gpu);

// matrix_rank: requires svdvals internally
static C_Status matrix_rank_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(matrix_rank, INSIGHT_DTYPE_F32,
                         matrix_rank_kernel_gpu);
REGISTER_IXUCA_KERNEL(matrix_rank, INSIGHT_DTYPE_F64,
                         matrix_rank_kernel_gpu);

// pinv: requires SVD + matrix multiplication
static C_Status pinv_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(pinv, INSIGHT_DTYPE_F32, pinv_kernel_gpu);
REGISTER_IXUCA_KERNEL(pinv, INSIGHT_DTYPE_F64, pinv_kernel_gpu);

// lstsq: requires gels or SVD
static C_Status lstsq_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(lstsq, INSIGHT_DTYPE_F32, lstsq_kernel_gpu);
REGISTER_IXUCA_KERNEL(lstsq, INSIGHT_DTYPE_F64, lstsq_kernel_gpu);

// eig: general eigenvalue, requires complex types
static C_Status eig_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(eig, INSIGHT_DTYPE_F32, eig_kernel_gpu);
REGISTER_IXUCA_KERNEL(eig, INSIGHT_DTYPE_F64, eig_kernel_gpu);

// eigvals: general eigenvalues only
static C_Status eigvals_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(eigvals, INSIGHT_DTYPE_F32, eigvals_kernel_gpu);
REGISTER_IXUCA_KERNEL(eigvals, INSIGHT_DTYPE_F64, eigvals_kernel_gpu);

// cov: uses mean + matmul, composed of other ops
static C_Status cov_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_IXUCA_KERNEL(cov, INSIGHT_DTYPE_F32, cov_kernel_gpu);
REGISTER_IXUCA_KERNEL(cov, INSIGHT_DTYPE_F64, cov_kernel_gpu);
