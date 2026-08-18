// backends/cuda/kernels/reduction/nanvar.cu
/**
 * @file nanvar.cu
 * @brief CUDA kernel for nanvar reduction.
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void nanvar_kernel(T *dst, const T *src, int64_t batch_size,
                              int64_t reduce_size, int ddof) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) {
    int64_t count = 0;
    T sum = 0;
    for (int64_t j = 0; j < reduce_size; ++j) {
      T val = src[i * reduce_size + j];
      if (!is_nan_device(val)) {
        sum += val;
        ++count;
      }
    }
    if (count == 0) {
      dst[i] = 0;
      return;
    }
    T mean = sum / (T)count;
    T sq_sum = 0;
    for (int64_t j = 0; j < reduce_size; ++j) {
      T val = src[i * reduce_size + j];
      if (!is_nan_device(val)) {
        T diff = val - mean;
        sq_sum += diff * diff;
      }
    }
    T divisor = (T)(count - ddof);
    if (divisor <= 0)
      divisor = 1;
    dst[i] = sq_sum / divisor;
  }
}

extern "C" {

C_Status nanvar_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *prepared = static_cast<InsightArray *>(inputs[1]);

  if (!out || !prepared) {
    gpu_set_last_error("nanvar: null array pointer");
    return C_FAILED;
  }

  int64_t batch_size = *static_cast<int64_t *>(inputs[2]);
  int64_t reduce_size = *static_cast<int64_t *>(inputs[3]);

  int threads = reduction_threads();
  int blocks = reduction_blocks(batch_size);

  int ddof = *static_cast<int *>(inputs[4]);
  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    nanvar_kernel<float>
        <<<blocks, threads>>>(static_cast<float *>(out->data),
                              static_cast<const float *>(prepared->data),
                              batch_size, reduce_size, ddof);
    break;
  case INSIGHT_DTYPE_F64:
    nanvar_kernel<double>
        <<<blocks, threads>>>(static_cast<double *>(out->data),
                              static_cast<const double *>(prepared->data),
                              batch_size, reduce_size, ddof);
    break;
  default:
    gpu_set_last_error("nanvar: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(nanvar, INSIGHT_DTYPE_F32, nanvar_kernel_gpu);
REGISTER_IXUCA_KERNEL(nanvar, INSIGHT_DTYPE_F64, nanvar_kernel_gpu);
