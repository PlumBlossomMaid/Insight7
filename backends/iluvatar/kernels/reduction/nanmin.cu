// backends/cuda/kernels/reduction/nanmin.cu
/**
 * @file nanmin.cu
 * @brief CUDA kernel for nanmin reduction.
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void nanmin_kernel(T *dst, const T *src, int64_t batch_size,
                              int64_t reduce_size) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) {
    T min_val = 0;
    bool found = false;
    for (int64_t j = 0; j < reduce_size; ++j) {
      T val = src[i * reduce_size + j];
      if (!is_nan_device(val)) {
        if (!found) {
          min_val = val;
          found = true;
        } else if (val < min_val) {
          min_val = val;
        }
      }
    }
    dst[i] = found ? min_val : 0;
  }
}

extern "C" {

C_Status nanmin_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *prepared = static_cast<InsightArray *>(inputs[1]);

  if (!out || !prepared) {
    gpu_set_last_error("nanmin: null array pointer");
    return C_FAILED;
  }

  int64_t batch_size = *static_cast<int64_t *>(inputs[2]);
  int64_t reduce_size = *static_cast<int64_t *>(inputs[3]);

  int threads = reduction_threads();
  int blocks = reduction_blocks(batch_size);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    nanmin_kernel<float><<<blocks, threads>>>(
        static_cast<float *>(out->data),
        static_cast<const float *>(prepared->data), batch_size, reduce_size);
    break;
  case INSIGHT_DTYPE_F64:
    nanmin_kernel<double><<<blocks, threads>>>(
        static_cast<double *>(out->data),
        static_cast<const double *>(prepared->data), batch_size, reduce_size);
    break;
  default:
    gpu_set_last_error("nanmin: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(nanmin, INSIGHT_DTYPE_F32, nanmin_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(nanmin, INSIGHT_DTYPE_F64, nanmin_kernel_gpu);
