// backends/cuda/kernels/linalg/trace.cu
/**
 * @file trace.cu
 * @brief CUDA kernel for trace (sum of diagonal elements).
 */

#include "common.cuh"

template <typename T>
__global__ void trace_kernel(T *dst, const T *src, int n) {
  T sum = T(0);
  for (int i = 0; i < n; ++i) {
    sum += src[i * n + i];
  }
  *dst = sum;
}

static C_Status trace_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    gpu_set_last_error("trace: null array pointer");
    return C_FAILED;
  }

  int n = (int)x->dims[0];

  if (x->dtype == INSIGHT_DTYPE_F64) {
    trace_kernel<double>
        <<<1, 1>>>((double *)out->data, (const double *)x->data, n);
  } else {
    trace_kernel<float>
        <<<1, 1>>>((float *)out->data, (const float *)x->data, n);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_ILUVATAR_KERNEL(trace, INSIGHT_DTYPE_F32, trace_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(trace, INSIGHT_DTYPE_F64, trace_kernel_gpu);
