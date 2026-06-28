// backends/cuda/kernels/linalg/dot.cu
/**
 * @file dot.cu
 * @brief CUDA kernel for dot product using cuBLAS.
 */

#include "common.cuh"

static C_Status dot_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *a = (InsightArray *)inputs[1];
  InsightArray *b = (InsightArray *)inputs[2];

  if (!out || !a || !b) {
    gpu_set_last_error("dot: null array pointer");
    return C_FAILED;
  }

  auto &handle = linalg_get_handle();
  handle.ensure(a->device_id);

  int n = (int)a->numel;

  if (a->dtype == INSIGHT_DTYPE_F64) {
    double result_val;
    cublasDdot(handle.blas, n, (const double *)a->data, 1,
               (const double *)b->data, 1, &result_val);
    cudaMemcpy((double *)out->data, &result_val, sizeof(double),
               cudaMemcpyHostToDevice);
  } else {
    float result_val;
    cublasSdot(handle.blas, n, (const float *)a->data, 1,
               (const float *)b->data, 1, &result_val);
    cudaMemcpy((float *)out->data, &result_val, sizeof(float),
               cudaMemcpyHostToDevice);
  }

  return C_SUCCESS;
}

REGISTER_ILUVATAR_KERNEL(dot, INSIGHT_DTYPE_F32, dot_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(dot, INSIGHT_DTYPE_F64, dot_kernel_gpu);
