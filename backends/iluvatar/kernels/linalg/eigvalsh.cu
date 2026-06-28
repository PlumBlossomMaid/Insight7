// backends/cuda/kernels/linalg/eigvalsh.cu
/**
 * @file eigvalsh.cu
 * @brief CUDA kernel for symmetric eigenvalues only using cuSOLVER.
 */

#include "common.cuh"

static C_Status eigvalsh_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *eigenvalues = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  int uplo_code = *(int *)inputs[2];

  if (!eigenvalues || !x) {
    gpu_set_last_error("eigvalsh: null array pointer");
    return C_FAILED;
  }

  auto &handle = linalg_get_handle();
  handle.ensure(x->device_id);

  int n = (int)x->dims[0];
  size_t nbytes = (size_t)n * n * insight_dtype_size(x->dtype);

  // Copy input (in-place syevd overwrites it)
  void *A_work = nullptr;
  cudaMalloc(&A_work, nbytes);
  cudaMemcpy(A_work, x->data, nbytes, cudaMemcpyDeviceToDevice);

  cublasFillMode_t uplo =
      (uplo_code == 0) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  int *d_info = nullptr;
  cudaMalloc(&d_info, sizeof(int));

  if (x->dtype == INSIGHT_DTYPE_F64) {
    int lwork = 0;
    cusolverDnDsyevd_bufferSize(handle.solver, CUSOLVER_EIG_MODE_NOVECTOR, uplo,
                                n, (double *)A_work, n,
                                (double *)eigenvalues->data, &lwork);
    double *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(double));
    cusolverDnDsyevd(handle.solver, CUSOLVER_EIG_MODE_NOVECTOR, uplo, n,
                     (double *)A_work, n, (double *)eigenvalues->data, d_work,
                     lwork, d_info);
    cudaFree(d_work);
  } else {
    int lwork = 0;
    cusolverDnSsyevd_bufferSize(handle.solver, CUSOLVER_EIG_MODE_NOVECTOR, uplo,
                                n, (float *)A_work, n,
                                (float *)eigenvalues->data, &lwork);
    float *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(float));
    cusolverDnSsyevd(handle.solver, CUSOLVER_EIG_MODE_NOVECTOR, uplo, n,
                     (float *)A_work, n, (float *)eigenvalues->data, d_work,
                     lwork, d_info);
    cudaFree(d_work);
  }

  cudaFree(A_work);
  cudaFree(d_info);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_ILUVATAR_KERNEL(eigvalsh, INSIGHT_DTYPE_F32, eigvalsh_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(eigvalsh, INSIGHT_DTYPE_F64, eigvalsh_kernel_gpu);
