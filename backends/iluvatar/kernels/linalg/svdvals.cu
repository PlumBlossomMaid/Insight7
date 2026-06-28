// backends/cuda/kernels/linalg/svdvals.cu
/**
 * @file svdvals.cu
 * @brief CUDA kernel for singular values only using cuSOLVER.
 *
 * Row-major A(m,n): transpose to column-major A_col(m,n),
 * call gesvd with jobu='N', jobvt='N' to compute only S.
 */

#include "common.cuh"

static C_Status svdvals_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    gpu_set_last_error("svdvals: null array pointer");
    return C_FAILED;
  }

  auto &handle = linalg_get_handle();
  handle.ensure(x->device_id);

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];
  size_t elem_size = insight_dtype_size(x->dtype);
  size_t data_bytes = (size_t)m * n * elem_size;

  // Step 1: Transpose A_row(m,n) -> A_col(m,n) column-major
  void *A_work = nullptr;
  cudaMalloc(&A_work, data_bytes);
  linalg_transpose_to_colmajor(handle.blas, A_work, x->data, m, n, x->dtype);

  int *d_info = nullptr;
  cudaMalloc(&d_info, sizeof(int));

  // Step 2: gesvd('N','N', m, n, A_col, m, S, ...) -- only singular values
  if (x->dtype == INSIGHT_DTYPE_F64) {
    int lwork = 0;
    cusolverDnDgesvd_bufferSize(handle.solver, m, n, &lwork);
    double *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(double));
    cusolverDnDgesvd(handle.solver, 'N', 'N', m, n, (double *)A_work, m,
                     (double *)out->data, nullptr, m, nullptr, 1, d_work, lwork,
                     nullptr, d_info);
    cudaFree(d_work);
  } else {
    int lwork = 0;
    cusolverDnSgesvd_bufferSize(handle.solver, m, n, &lwork);
    float *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(float));
    cusolverDnSgesvd(handle.solver, 'N', 'N', m, n, (float *)A_work, m,
                     (float *)out->data, nullptr, m, nullptr, 1, d_work, lwork,
                     nullptr, d_info);
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

REGISTER_ILUVATAR_KERNEL(svdvals, INSIGHT_DTYPE_F32, svdvals_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(svdvals, INSIGHT_DTYPE_F64, svdvals_kernel_gpu);
