// backends/cuda/kernels/linalg/norm.cu
/**
 * @file norm.cu
 * @brief CUDA kernel for matrix/vector norms.
 *
 * Vector 2-norm: cuBLAS nrm2
 * Matrix Frobenius: cuBLAS nrm2 on flattened data
 * Vector 1-norm: cuBLAS asum
 * Vector inf-norm: custom kernel (absmax)
 * Matrix 1-norm / inf-norm: CPU fallback
 */

#include "common.cuh"
#include <cmath>
#include <limits>

template <typename T>
__global__ void norm_absmax_kernel(T *dst, const T *src, int total) {
  T max_val = T(0);
  for (int i = 0; i < total; ++i) {
    T val = src[i];
    if (val < T(0))
      val = -val;
    if (val > max_val)
      max_val = val;
  }
  *dst = max_val;
}

static C_Status norm_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  double ord = *(double *)inputs[2];

  if (!out || !x) {
    gpu_set_last_error("norm: null array pointer");
    return C_FAILED;
  }

  auto &handle = linalg_get_handle();
  handle.ensure(x->device_id);

  int ndim = x->ndim;
  int n = (int)x->numel;

  // Vector 2-norm
  if (ndim == 1 && ord == 2.0) {
    if (x->dtype == INSIGHT_DTYPE_F64) {
      double norm_val;
      cublasDnrm2(handle.blas, n, (const double *)x->data, 1, &norm_val);
      cudaMemcpy((double *)out->data, &norm_val, sizeof(double),
                 cudaMemcpyHostToDevice);
    } else {
      float norm_val;
      cublasSnrm2(handle.blas, n, (const float *)x->data, 1, &norm_val);
      cudaMemcpy((float *)out->data, &norm_val, sizeof(float),
                 cudaMemcpyHostToDevice);
    }
    return C_SUCCESS;
  }

  // Matrix Frobenius norm (ord=2 for matrix) or vector 1-norm
  if ((ndim == 2 && ord == 2.0) || (ndim == 1 && ord == 1.0)) {
    if (x->dtype == INSIGHT_DTYPE_F64) {
      double norm_val;
      cublasDnrm2(handle.blas, n, (const double *)x->data, 1, &norm_val);
      cudaMemcpy((double *)out->data, &norm_val, sizeof(double),
                 cudaMemcpyHostToDevice);
    } else {
      float norm_val;
      cublasSnrm2(handle.blas, n, (const float *)x->data, 1, &norm_val);
      cudaMemcpy((float *)out->data, &norm_val, sizeof(float),
                 cudaMemcpyHostToDevice);
    }
    return C_SUCCESS;
  }

  // Vector inf-norm
  if (ndim == 1 && ord == std::numeric_limits<double>::infinity()) {
    if (x->dtype == INSIGHT_DTYPE_F64) {
      norm_absmax_kernel<double>
          <<<1, 1>>>((double *)out->data, (const double *)x->data, n);
    } else {
      norm_absmax_kernel<float>
          <<<1, 1>>>((float *)out->data, (const float *)x->data, n);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      gpu_set_last_error(cudaGetErrorString(err));
      return C_FAILED;
    }
    return C_SUCCESS;
  }

  // Matrix 1-norm, inf-norm, and other cases: fallback to CPU
  return C_FALLBACK;
}

REGISTER_ILUVATAR_KERNEL(norm, INSIGHT_DTYPE_F32, norm_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(norm, INSIGHT_DTYPE_F64, norm_kernel_gpu);
