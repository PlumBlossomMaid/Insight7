// backends/cuda/kernels/reduction/sum.cu
/**
 * @file sum.cu
 * @brief CUDA kernel for sum reduction.
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void sum_kernel(T *dst, const T *src, int64_t total_out,
                           int64_t reduce_size) {
  extern __shared__ char sdata_raw[];
  T *sdata = reinterpret_cast<T *>(sdata_raw);

  int tid = threadIdx.x;
  int idx = blockIdx.x;

  T sum = T(0);
  for (int64_t j = tid; j < reduce_size; j += blockDim.x) {
    sum += src[idx * reduce_size + j];
  }

  sdata[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    dst[idx] = sdata[0];
  }
}

__global__ void sum_c64_kernel(cuFloatComplex *dst, const cuFloatComplex *src,
                               int64_t total_out, int64_t reduce_size) {
  extern __shared__ char sdata_raw[];
  cuFloatComplex *sdata = reinterpret_cast<cuFloatComplex *>(sdata_raw);
  int tid = threadIdx.x;
  int idx = blockIdx.x;
  cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
  for (int64_t j = tid; j < reduce_size; j += blockDim.x) {
    sum = cuCaddf(sum, src[idx * reduce_size + j]);
  }
  sdata[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = cuCaddf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
  if (tid == 0)
    dst[idx] = sdata[0];
}

__global__ void sum_c128_kernel(cuDoubleComplex *dst,
                                const cuDoubleComplex *src, int64_t total_out,
                                int64_t reduce_size) {
  extern __shared__ char sdata_raw[];
  cuDoubleComplex *sdata = reinterpret_cast<cuDoubleComplex *>(sdata_raw);
  int tid = threadIdx.x;
  int idx = blockIdx.x;
  cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
  for (int64_t j = tid; j < reduce_size; j += blockDim.x) {
    sum = cuCadd(sum, src[idx * reduce_size + j]);
  }
  sdata[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = cuCadd(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
  if (tid == 0)
    dst[idx] = sdata[0];
}

extern "C" {

C_Status sum_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *prepared = static_cast<InsightArray *>(inputs[1]);

  if (!out || !prepared) {
    gpu_set_last_error("sum: null array pointer");
    return C_FAILED;
  }

  int64_t total_out = out->numel;
  int64_t reduce_size = prepared->numel / total_out;

  int threads = reduction_threads();
  dim3 blocks(total_out);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    sum_kernel<float><<<blocks, threads, threads * sizeof(float)>>>(
        static_cast<float *>(out->data),
        static_cast<const float *>(prepared->data), total_out, reduce_size);
    break;
  case INSIGHT_DTYPE_F64:
    sum_kernel<double><<<blocks, threads, threads * sizeof(double)>>>(
        static_cast<double *>(out->data),
        static_cast<const double *>(prepared->data), total_out, reduce_size);
    break;
  case INSIGHT_DTYPE_I32:
    sum_kernel<int32_t><<<blocks, threads, threads * sizeof(int32_t)>>>(
        static_cast<int32_t *>(out->data),
        static_cast<const int32_t *>(prepared->data), total_out, reduce_size);
    break;
  case INSIGHT_DTYPE_I64:
    sum_kernel<int64_t><<<blocks, threads, threads * sizeof(int64_t)>>>(
        static_cast<int64_t *>(out->data),
        static_cast<const int64_t *>(prepared->data), total_out, reduce_size);
    break;
  case INSIGHT_DTYPE_C32:
    sum_c64_kernel<<<blocks, threads, threads * sizeof(cuFloatComplex)>>>(
        static_cast<cuFloatComplex *>(out->data),
        static_cast<const cuFloatComplex *>(prepared->data), total_out,
        reduce_size);
    break;
  case INSIGHT_DTYPE_C64:
    sum_c128_kernel<<<blocks, threads, threads * sizeof(cuDoubleComplex)>>>(
        static_cast<cuDoubleComplex *>(out->data),
        static_cast<const cuDoubleComplex *>(prepared->data), total_out,
        reduce_size);
    break;
  default:
    gpu_set_last_error("sum: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(sum, INSIGHT_DTYPE_F32, sum_kernel_gpu);
REGISTER_IXUCA_KERNEL(sum, INSIGHT_DTYPE_F64, sum_kernel_gpu);
REGISTER_IXUCA_KERNEL(sum, INSIGHT_DTYPE_I32, sum_kernel_gpu);
REGISTER_IXUCA_KERNEL(sum, INSIGHT_DTYPE_I64, sum_kernel_gpu);
REGISTER_IXUCA_KERNEL(sum, INSIGHT_DTYPE_C32, sum_kernel_gpu);
REGISTER_IXUCA_KERNEL(sum, INSIGHT_DTYPE_C64, sum_kernel_gpu);
