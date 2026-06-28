// backends/cuda/kernels/elementwise/pow.cu
/**
 * @file pow.cu
 * @brief CUDA kernel for power operation.
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cmath>

template <typename T>
__global__ void pow_kernel_float(const T *a, const T *b, T *out,
                                 const ElementwiseMetadata *meta) {
  int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= meta->numel)
    return;
  int64_t a_off =
      meta->a_offset + elementwise_offset(linear, meta, meta->a_strides);
  int64_t b_off =
      meta->b_offset + elementwise_offset(linear, meta, meta->b_strides);
  int64_t out_off =
      meta->out_offset + elementwise_offset(linear, meta, meta->out_strides);
  out[out_off] = std::pow(a[a_off], b[b_off]);
}

template <typename T>
__global__ void pow_kernel_int(const T *a, const T *b, T *out,
                               const ElementwiseMetadata *meta) {
  int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= meta->numel)
    return;
  int64_t a_off =
      meta->a_offset + elementwise_offset(linear, meta, meta->a_strides);
  int64_t b_off =
      meta->b_offset + elementwise_offset(linear, meta, meta->b_strides);
  int64_t out_off =
      meta->out_offset + elementwise_offset(linear, meta, meta->out_strides);
  out[out_off] = static_cast<T>(
      std::pow(static_cast<double>(a[a_off]), static_cast<double>(b[b_off])));
}

C_Status pow_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *a = (InsightArray *)inputs[0];
  InsightArray *b = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a || !b || !out) {
    gpu_set_last_error("pow: null array pointer");
    return C_FAILED;
  }

  ElementwiseMetadata *meta = alloc_elementwise_metadata(a, b, out);
  dim3 blocks = elementwise_blocks(out->numel);
  dim3 threads = elementwise_threads();

  switch (a->dtype) {
  case INSIGHT_DTYPE_F32:
    pow_kernel_float<float><<<blocks, threads>>>(
        (float *)a->data, (float *)b->data, (float *)out->data, meta);
    break;
  case INSIGHT_DTYPE_F64:
    pow_kernel_float<double><<<blocks, threads>>>(
        (double *)a->data, (double *)b->data, (double *)out->data, meta);
    break;
  case INSIGHT_DTYPE_I8:
    pow_kernel_int<int8_t><<<blocks, threads>>>(
        (int8_t *)a->data, (int8_t *)b->data, (int8_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_I16:
    pow_kernel_int<int16_t><<<blocks, threads>>>(
        (int16_t *)a->data, (int16_t *)b->data, (int16_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_I32:
    pow_kernel_int<int32_t><<<blocks, threads>>>(
        (int32_t *)a->data, (int32_t *)b->data, (int32_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_I64:
    pow_kernel_int<int64_t><<<blocks, threads>>>(
        (int64_t *)a->data, (int64_t *)b->data, (int64_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_U8:
    pow_kernel_int<uint8_t><<<blocks, threads>>>(
        (uint8_t *)a->data, (uint8_t *)b->data, (uint8_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_U16:
    pow_kernel_int<uint16_t><<<blocks, threads>>>(
        (uint16_t *)a->data, (uint16_t *)b->data, (uint16_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_U32:
    pow_kernel_int<uint32_t><<<blocks, threads>>>(
        (uint32_t *)a->data, (uint32_t *)b->data, (uint32_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_U64:
    pow_kernel_int<uint64_t><<<blocks, threads>>>(
        (uint64_t *)a->data, (uint64_t *)b->data, (uint64_t *)out->data, meta);
    break;
  default:
    free_elementwise_metadata(meta);
    gpu_set_last_error("pow: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  free_elementwise_metadata(meta);

  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_ILUVATAR_KERNEL(pow, INSIGHT_DTYPE_F32, pow_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pow, INSIGHT_DTYPE_F64, pow_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pow, INSIGHT_DTYPE_I8, pow_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pow, INSIGHT_DTYPE_I16, pow_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pow, INSIGHT_DTYPE_I32, pow_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pow, INSIGHT_DTYPE_I64, pow_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pow, INSIGHT_DTYPE_U8, pow_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pow, INSIGHT_DTYPE_U16, pow_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pow, INSIGHT_DTYPE_U32, pow_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pow, INSIGHT_DTYPE_U64, pow_kernel_gpu);
