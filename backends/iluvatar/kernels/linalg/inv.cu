// backends/cuda/kernels/linalg/inv.cu
/**
 * @file inv.cu
 * @brief CUDA kernel for matrix inverse using cuSOLVER (getrf + getri).
 */

#include "common.cuh"

static C_Status inv_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    gpu_set_last_error("inv: null array pointer");
    return C_FAILED;
  }

  auto &handle = linalg_get_handle();
  handle.ensure(x->device_id);

  int n = (int)x->dims[0];
  size_t nbytes = (size_t)n * n * insight_dtype_size(x->dtype);

  // Copy input to work buffer
  void *work_A = nullptr;
  cudaMalloc(&work_A, nbytes);
  cudaMemcpy(work_A, x->data, nbytes, cudaMemcpyDeviceToDevice);

  int *d_ipiv = nullptr;
  int *d_info = nullptr;
  cudaMalloc(&d_ipiv, n * sizeof(int));
  cudaMalloc(&d_info, sizeof(int));

  if (x->dtype == INSIGHT_DTYPE_F64) {
    int lwork = 0;
    cusolverDnDgetrf_bufferSize(handle.solver, n, n, (double *)work_A, n,
                                &lwork);
    double *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(double));

    // LU factorization
    cusolverDnDgetrf(handle.solver, n, n, (double *)work_A, n, d_work, d_ipiv,
                     d_info);

    // Inverse via getri: solve A * inv(A) = I
    // cuSOLVER doesn't have getri directly. Use getrs with identity.
    // First, create identity matrix in output
    dim3 t(16, 16);
    dim3 b((n + 15) / 16, (n + 15) / 16);
    linalg_identity_kernel<double><<<b, t>>>((double *)out->data, n);

    cusolverDnDgetrs(handle.solver, CUBLAS_OP_N, n, n, (double *)work_A, n,
                     d_ipiv, (double *)out->data, n, d_info);

    cudaFree(d_work);
  } else {
    int lwork = 0;
    cusolverDnSgetrf_bufferSize(handle.solver, n, n, (float *)work_A, n,
                                &lwork);
    float *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(float));

    cusolverDnSgetrf(handle.solver, n, n, (float *)work_A, n, d_work, d_ipiv,
                     d_info);

    dim3 t(16, 16);
    dim3 b((n + 15) / 16, (n + 15) / 16);
    linalg_identity_kernel<float><<<b, t>>>((float *)out->data, n);

    cusolverDnSgetrs(handle.solver, CUBLAS_OP_N, n, n, (float *)work_A, n,
                     d_ipiv, (float *)out->data, n, d_info);

    cudaFree(d_work);
  }

  cudaFree(work_A);
  cudaFree(d_ipiv);
  cudaFree(d_info);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_ILUVATAR_KERNEL(inv, INSIGHT_DTYPE_F32, inv_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(inv, INSIGHT_DTYPE_F64, inv_kernel_gpu);
