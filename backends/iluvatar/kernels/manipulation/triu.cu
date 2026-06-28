// backends/cuda/kernels/manipulation/triu.cu
/**
 * @file triu.cu
 * @brief CUDA kernel for triu operation.
 */

#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cstring>
#include <cuda_runtime.h>

template <typename T>
__global__ void triu_kernel(const T *src, T *dst, int64_t rows, int64_t cols,
                            int64_t k) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t n = rows * cols;
  if (idx < n) {
    int64_t row = idx / cols;
    int64_t col = idx % cols;
    if (col >= row + k) {
      dst[idx] = src[idx];
    } else {
      dst[idx] = T(0);
    }
  }
}

extern "C" {

C_Status triu_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);

  if (!out || !x) {
    gpu_set_last_error("triu: null array pointer");
    return C_FAILED;
  }

  int k = *static_cast<int *>(inputs[2]);
  int64_t rows = x->dims[0];
  int64_t cols = x->dims[1];
  int64_t n = rows * cols;

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    triu_kernel<float><<<blocks, threads>>>(static_cast<const float *>(x->data),
                                            static_cast<float *>(out->data),
                                            rows, cols, k);
    break;
  case INSIGHT_DTYPE_F64:
    triu_kernel<double>
        <<<blocks, threads>>>(static_cast<const double *>(x->data),
                              static_cast<double *>(out->data), rows, cols, k);
    break;
  case INSIGHT_DTYPE_I32:
    triu_kernel<int32_t>
        <<<blocks, threads>>>(static_cast<const int32_t *>(x->data),
                              static_cast<int32_t *>(out->data), rows, cols, k);
    break;
  case INSIGHT_DTYPE_I64:
    triu_kernel<int64_t>
        <<<blocks, threads>>>(static_cast<const int64_t *>(x->data),
                              static_cast<int64_t *>(out->data), rows, cols, k);
    break;
  default:
    gpu_set_last_error("triu: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(triu, INSIGHT_DTYPE_F32, triu_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(triu, INSIGHT_DTYPE_F64, triu_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(triu, INSIGHT_DTYPE_I32, triu_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(triu, INSIGHT_DTYPE_I64, triu_kernel_gpu);
