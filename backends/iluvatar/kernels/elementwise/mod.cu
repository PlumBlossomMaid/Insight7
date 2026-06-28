// backends/cuda/kernels/elementwise/mod.cu
/**
 * @file mod.cu
 * @brief CUDA kernel for modulo operation.
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cmath>

template <typename T>
__global__ void mod_kernel_int(const T *a, const T *b, T *out,
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
  out[out_off] = a[a_off] % b[b_off];
}

__global__ void mod_kernel_float(const float *a, const float *b, float *out,
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
  out[out_off] = fmodf(a[a_off], b[b_off]);
}

__global__ void mod_kernel_double(const double *a, const double *b, double *out,
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
  out[out_off] = fmod(a[a_off], b[b_off]);
}

C_Status mod_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *a = (InsightArray *)inputs[0];
  InsightArray *b = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a || !b || !out) {
    gpu_set_last_error("mod: null array pointer");
    return C_FAILED;
  }

  ElementwiseMetadata *meta = alloc_elementwise_metadata(a, b, out);
  dim3 blocks = elementwise_blocks(out->numel);
  dim3 threads = elementwise_threads();

  switch (a->dtype) {
  case INSIGHT_DTYPE_I8:
    mod_kernel_int<int8_t><<<blocks, threads>>>(
        (int8_t *)a->data, (int8_t *)b->data, (int8_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_I16:
    mod_kernel_int<int16_t><<<blocks, threads>>>(
        (int16_t *)a->data, (int16_t *)b->data, (int16_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_I32:
    mod_kernel_int<int32_t><<<blocks, threads>>>(
        (int32_t *)a->data, (int32_t *)b->data, (int32_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_I64:
    mod_kernel_int<int64_t><<<blocks, threads>>>(
        (int64_t *)a->data, (int64_t *)b->data, (int64_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_U8:
    mod_kernel_int<uint8_t><<<blocks, threads>>>(
        (uint8_t *)a->data, (uint8_t *)b->data, (uint8_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_U16:
    mod_kernel_int<uint16_t><<<blocks, threads>>>(
        (uint16_t *)a->data, (uint16_t *)b->data, (uint16_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_U32:
    mod_kernel_int<uint32_t><<<blocks, threads>>>(
        (uint32_t *)a->data, (uint32_t *)b->data, (uint32_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_U64:
    mod_kernel_int<uint64_t><<<blocks, threads>>>(
        (uint64_t *)a->data, (uint64_t *)b->data, (uint64_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_F32:
    mod_kernel_float<<<blocks, threads>>>((float *)a->data, (float *)b->data,
                                          (float *)out->data, meta);
    break;
  case INSIGHT_DTYPE_F64:
    mod_kernel_double<<<blocks, threads>>>((double *)a->data, (double *)b->data,
                                           (double *)out->data, meta);
    break;
  default:
    free_elementwise_metadata(meta);
    gpu_set_last_error("mod: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(mod, INSIGHT_DTYPE_I8, mod_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(mod, INSIGHT_DTYPE_I16, mod_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(mod, INSIGHT_DTYPE_I32, mod_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(mod, INSIGHT_DTYPE_I64, mod_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(mod, INSIGHT_DTYPE_U8, mod_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(mod, INSIGHT_DTYPE_U16, mod_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(mod, INSIGHT_DTYPE_U32, mod_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(mod, INSIGHT_DTYPE_U64, mod_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(mod, INSIGHT_DTYPE_F32, mod_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(mod, INSIGHT_DTYPE_F64, mod_kernel_gpu);
