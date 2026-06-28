// backends/cuda/kernels/linalg/eigh.cu
/**
 * @file eigh.cu
 * @brief CUDA kernel for symmetric eigenvalue decomposition using cuSOLVER.
 *
 * A is symmetric, so row-major = column-major (A = A^T).
 * We can call syevd directly on the row-major data.
 * uplo: 0 = 'L' (lower), 1 = 'U' (upper).
 * cuSOLVER uplo: 'L' = CUBLAS_FILL_MODE_LOWER, 'U' = CUBLAS_FILL_MODE_UPPER.
 * Row-major lower triangle = column-major upper triangle, and vice versa.
 * So uplo=0 (row-major lower) → cuSOLVER 'U' (column-major upper).
 */

#include "common.cuh"

static C_Status eigh_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *eigenvalues = (InsightArray *)outputs[0];
  InsightArray *eigenvectors = (InsightArray *)outputs[1];
  InsightArray *x = (InsightArray *)inputs[2];
  int uplo_code = *(int *)inputs[3]; // 0 = 'L', 1 = 'U'

  if (!eigenvalues || !eigenvectors || !x) {
    gpu_set_last_error("eigh: null array pointer");
    return C_FAILED;
  }

  auto &handle = linalg_get_handle();
  handle.ensure(x->device_id);

  int n = (int)x->dims[0];
  size_t nbytes = (size_t)n * n * insight_dtype_size(x->dtype);

  // Copy input to eigenvectors (in-place syevd)
  cudaMemcpy(eigenvectors->data, x->data, nbytes, cudaMemcpyDeviceToDevice);

  // Row-major lower = column-major upper, and vice versa
  cublasFillMode_t uplo =
      (uplo_code == 0) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  int *d_info = nullptr;
  cudaMalloc(&d_info, sizeof(int));

  if (x->dtype == INSIGHT_DTYPE_F64) {
    int lwork = 0;
    cusolverDnDsyevd_bufferSize(handle.solver, CUSOLVER_EIG_MODE_VECTOR, uplo,
                                n, (double *)eigenvectors->data, n,
                                (double *)eigenvalues->data, &lwork);
    double *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(double));
    cusolverDnDsyevd(handle.solver, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                     (double *)eigenvectors->data, n,
                     (double *)eigenvalues->data, d_work, lwork, d_info);
    cudaFree(d_work);
  } else {
    int lwork = 0;
    cusolverDnSsyevd_bufferSize(handle.solver, CUSOLVER_EIG_MODE_VECTOR, uplo,
                                n, (float *)eigenvectors->data, n,
                                (float *)eigenvalues->data, &lwork);
    float *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(float));
    cusolverDnSsyevd(handle.solver, CUSOLVER_EIG_MODE_VECTOR, uplo, n,
                     (float *)eigenvectors->data, n, (float *)eigenvalues->data,
                     d_work, lwork, d_info);
    cudaFree(d_work);
  }

  cudaFree(d_info);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_ILUVATAR_KERNEL(eigh, INSIGHT_DTYPE_F32, eigh_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(eigh, INSIGHT_DTYPE_F64, eigh_kernel_gpu);
