// backends/cuda/kernels/linalg/slogdet.cu
/**
 * @file slogdet.cu
 * @brief CUDA kernel for sign and log-determinant using cuSOLVER.
 */

#include "common.cuh"
#include <cmath>

template <typename T>
__global__ void slogdet_kernel(T *sign_out, double *logdet_out, const T *lu,
                               const int *ipiv, int n) {
  int sign = 1;
  double logdet = 0.0;
  for (int i = 0; i < n; ++i) {
    T diag = lu[i * n + i];
    if (diag < T(0)) {
      sign = -sign;
      logdet += log((double)(-diag));
    } else {
      logdet += log((double)diag);
    }
    if (ipiv[i] != i + 1)
      sign = -sign;
  }
  *sign_out = (T)sign;
  *logdet_out = logdet;
}

static C_Status slogdet_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *sign_out = (InsightArray *)outputs[0];
  InsightArray *logdet_out = (InsightArray *)outputs[1];
  InsightArray *x = (InsightArray *)inputs[2];

  if (!sign_out || !logdet_out || !x) {
    gpu_set_last_error("slogdet: null array pointer");
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

    slogdet_kernel<double><<<1, 1>>>((double *)sign_out->data,
                                     (double *)logdet_out->data,
                                     (const double *)lu_data, d_ipiv, n);
  } else {
    int lwork = 0;
    cusolverDnSgetrf_bufferSize(handle.solver, n, n, (float *)lu_data, n,
                                &lwork);
    float *d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(float));
    cusolverDnSgetrf(handle.solver, n, n, (float *)lu_data, n, d_work, d_ipiv,
                     d_info);
    cudaFree(d_work);

    slogdet_kernel<float><<<1, 1>>>((float *)sign_out->data,
                                    (double *)logdet_out->data,
                                    (const float *)lu_data, d_ipiv, n);
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

REGISTER_IXUCA_KERNEL(slogdet, INSIGHT_DTYPE_F32, slogdet_kernel_gpu);
REGISTER_IXUCA_KERNEL(slogdet, INSIGHT_DTYPE_F64, slogdet_kernel_gpu);
