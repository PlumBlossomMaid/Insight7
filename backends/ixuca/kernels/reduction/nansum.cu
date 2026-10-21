// backends/cuda/kernels/reduction/nansum.cu
/**
 * @file nansum.cu
 * @brief CUDA kernel for nansum reduction (sum ignoring NaN).
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* sum_out
 *   inputs[1] = InsightArray* count_out
 *   inputs[2] = InsightArray* prepared
 *   inputs[3] = int64_t* batch_size
 *   inputs[4] = int64_t* reduce_size
 *
 * Output layout:
 *   outputs[0] = InsightArray* sum_out
 *   outputs[1] = InsightArray* count_out
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void nansum_kernel(T *sum_dst, int64_t *count_dst, const T *src,
                              int64_t batch_size, int64_t reduce_size) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) {
    T sum = 0;
    int64_t cnt = 0;
    for (int64_t j = 0; j < reduce_size; ++j) {
      T val = src[i * reduce_size + j];
      if (!is_nan_device(val)) {
        sum += val;
        ++cnt;
      }
    }
    sum_dst[i] = sum;
    count_dst[i] = cnt;
  }
}

extern "C" {

C_Status nansum_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *sum_out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *count_out = static_cast<InsightArray *>(outputs[1]);
  InsightArray *prepared = static_cast<InsightArray *>(inputs[2]);

  if (!sum_out || !count_out || !prepared) {
    gpu_set_last_error("nansum: null array pointer");
    return C_FAILED;
  }

  int64_t batch_size = *static_cast<int64_t *>(inputs[3]);
  int64_t reduce_size = *static_cast<int64_t *>(inputs[4]);

  int threads = reduction_threads();
  int blocks = reduction_blocks(batch_size);

  switch (prepared->dtype) {
  case INSIGHT_DTYPE_F32:
    nansum_kernel<float><<<blocks, threads>>>(
        static_cast<float *>(sum_out->data),
        static_cast<int64_t *>(count_out->data),
        static_cast<const float *>(prepared->data), batch_size, reduce_size);
    break;
  case INSIGHT_DTYPE_F64:
    nansum_kernel<double><<<blocks, threads>>>(
        static_cast<double *>(sum_out->data),
        static_cast<int64_t *>(count_out->data),
        static_cast<const double *>(prepared->data), batch_size, reduce_size);
    break;
  default:
    gpu_set_last_error("nansum: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(nansum, INSIGHT_DTYPE_F32, nansum_kernel_gpu);
REGISTER_IXUCA_KERNEL(nansum, INSIGHT_DTYPE_F64, nansum_kernel_gpu);
