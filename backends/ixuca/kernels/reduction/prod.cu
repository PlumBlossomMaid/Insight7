// backends/cuda/kernels/reduction/prod.cu
/**
 * @file prod.cu
 * @brief CUDA kernel for product reduction.
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void prod_kernel(T *dst, const T *src, int64_t total_out,
                            int64_t reduce_size) {
  extern __shared__ char sdata_raw[];
  T *sdata = reinterpret_cast<T *>(sdata_raw);

  int tid = threadIdx.x;
  int idx = blockIdx.x;

  T prod = T(1);
  for (int64_t j = tid; j < reduce_size; j += blockDim.x) {
    prod *= src[idx * reduce_size + j];
  }

  sdata[tid] = prod;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] *= sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    dst[idx] = sdata[0];
  }
}

extern "C" {

C_Status prod_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *prepared = static_cast<InsightArray *>(inputs[1]);

  if (!out || !prepared) {
    gpu_set_last_error("prod: null array pointer");
    return C_FAILED;
  }

  int64_t total_out = out->numel;
  int64_t reduce_size = prepared->numel / total_out;

  int threads = reduction_threads();
  dim3 blocks(total_out);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    prod_kernel<float><<<blocks, threads, threads * sizeof(float)>>>(
        static_cast<float *>(out->data),
        static_cast<const float *>(prepared->data), total_out, reduce_size);
    break;
  case INSIGHT_DTYPE_F64:
    prod_kernel<double><<<blocks, threads, threads * sizeof(double)>>>(
        static_cast<double *>(out->data),
        static_cast<const double *>(prepared->data), total_out, reduce_size);
    break;
  case INSIGHT_DTYPE_I32:
    prod_kernel<int32_t><<<blocks, threads, threads * sizeof(int32_t)>>>(
        static_cast<int32_t *>(out->data),
        static_cast<const int32_t *>(prepared->data), total_out, reduce_size);
    break;
  case INSIGHT_DTYPE_I64:
    prod_kernel<int64_t><<<blocks, threads, threads * sizeof(int64_t)>>>(
        static_cast<int64_t *>(out->data),
        static_cast<const int64_t *>(prepared->data), total_out, reduce_size);
    break;
  default:
    gpu_set_last_error("prod: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(prod, INSIGHT_DTYPE_F32, prod_kernel_gpu);
REGISTER_IXUCA_KERNEL(prod, INSIGHT_DTYPE_F64, prod_kernel_gpu);
REGISTER_IXUCA_KERNEL(prod, INSIGHT_DTYPE_I32, prod_kernel_gpu);
REGISTER_IXUCA_KERNEL(prod, INSIGHT_DTYPE_I64, prod_kernel_gpu);
