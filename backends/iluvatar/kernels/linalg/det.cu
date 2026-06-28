// backends/cuda/kernels/linalg/det.cu
/**
 * @file det.cu
 * @brief CUDA kernel for matrix determinant using cuSOLVER (LU decomposition).
 */

#include "common.cuh"

template <typename T>
__global__ void det_from_lu_kernel(T *dst, const T *lu, const int *ipiv,
                                   int n) {
  T prod = T(1);
  int sign = 1;
  for (int i = 0; i < n; ++i) {
    prod *= lu[i * n + i];
    // ipiv is 1-based: if ipiv[i] != i+1, a swap happened
    if (ipiv[i] != i + 1)
      sign = -sign;
  }
  *dst = (sign == 1) ? prod : -prod;
}

static C_Status det_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    gpu_set_last_error("det: null array pointer");
    return C_FAILED;
  }

  auto &handle = linalg_get_handle();
  handle.ensure(x->device_id);

  int n = (int)x->dims[0];
  size_t nbytes = (size_t)n * n * insight_dtype_size(x->dtype);

  void *lu_data = nullptr;
  cudaMalloc(&lu_data, nbytes);
  cudaMemcpy(lu_data, x->data, nbytes, cudaMemcpyDeviceToDevice);

  int *d_ipiv = nullptr;
  int *d_info = nullptr;
  cudaMalloc(&d_ipiv, n * sizeof(int));
  cudaMalloc(&d_info, sizeof(int));

  if (x->dtype == INSIGHT_DTYPE_F64) {
    int lwork = 0;
    cusolverDnDgetrf_bufferSize(handle.solver, n, n, (double *)lu_data, n,
                                &lwork);
    double *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(double));
    cusolverDnDgetrf(handle.solver, n, n, (double *)lu_data, n, d_work, d_ipiv,
                     d_info);
    cudaFree(d_work);

    det_from_lu_kernel<double>
        <<<1, 1>>>((double *)out->data, (const double *)lu_data, d_ipiv, n);
  } else {
    int lwork = 0;
    cusolverDnSgetrf_bufferSize(handle.solver, n, n, (float *)lu_data, n,
                                &lwork);
    float *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(float));
    cusolverDnSgetrf(handle.solver, n, n, (float *)lu_data, n, d_work, d_ipiv,
                     d_info);
    cudaFree(d_work);

    det_from_lu_kernel<float>
        <<<1, 1>>>((float *)out->data, (const float *)lu_data, d_ipiv, n);
  }

  cudaFree(lu_data);
  cudaFree(d_ipiv);
  cudaFree(d_info);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_ILUVATAR_KERNEL(det, INSIGHT_DTYPE_F32, det_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(det, INSIGHT_DTYPE_F64, det_kernel_gpu);
