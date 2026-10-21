// backends/cuda/kernels/elementwise/mul.cu
/**
 * @file mul.cu
 * @brief CUDA kernel for Multiplication operation.
 *
 * Computes elementwise Multiplication of two arrays with stride support and
 * broadcasting.
 *
 * @param inputs  [0] = InsightArray* left operand
 *                [1] = InsightArray* right operand
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ============================================================================
// Kernel Implementations
// ============================================================================

template <typename T>
__global__ void mul_kernel(const T *a, const T *b, T *out,
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
  out[out_off] = a[a_off] * b[b_off];
}

// Half-precision specializations (use hmul() to avoid ambiguous * on CUDA 11.8)
__global__ void mul_f16_kernel(const __half *a, const __half *b, __half *out,
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
  out[out_off] = hmul(a[a_off], b[b_off]);
}
__global__ void mul_bf16_kernel(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                                __nv_bfloat16 *out,
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
  out[out_off] = hmul(a[a_off], b[b_off]);
}

__global__ void mul_c32_kernel(const cuFloatComplex *a, const cuFloatComplex *b,
                               cuFloatComplex *out,
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
  out[out_off] = cuCmulf(a[a_off], b[b_off]);
}
__global__ void mul_c64_kernel(const cuDoubleComplex *a,
                               const cuDoubleComplex *b, cuDoubleComplex *out,
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
  out[out_off] = cuCmul(a[a_off], b[b_off]);
}

// ============================================================================
// Wrapper Function
// ============================================================================

C_Status mul_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *a = (InsightArray *)inputs[0];
  InsightArray *b = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!a || !b || !out) {
    gpu_set_last_error("mul: null array pointer");
    return C_FAILED;
  }

  ElementwiseMetadata *meta = alloc_elementwise_metadata(a, b, out);
  dim3 blocks = elementwise_blocks(out->numel);
  dim3 threads = elementwise_threads();

  switch (a->dtype) {
  case INSIGHT_DTYPE_F32:
    mul_kernel<float><<<blocks, threads>>>((float *)a->data, (float *)b->data,
                                           (float *)out->data, meta);
    break;
  case INSIGHT_DTYPE_F64:
    mul_kernel<double><<<blocks, threads>>>(
        (double *)a->data, (double *)b->data, (double *)out->data, meta);
    break;
  case INSIGHT_DTYPE_I8:
    mul_kernel<int8_t><<<blocks, threads>>>(
        (int8_t *)a->data, (int8_t *)b->data, (int8_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_I16:
    mul_kernel<int16_t><<<blocks, threads>>>(
        (int16_t *)a->data, (int16_t *)b->data, (int16_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_I32:
    mul_kernel<int32_t><<<blocks, threads>>>(
        (int32_t *)a->data, (int32_t *)b->data, (int32_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_I64:
    mul_kernel<int64_t><<<blocks, threads>>>(
        (int64_t *)a->data, (int64_t *)b->data, (int64_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_U8:
    mul_kernel<uint8_t><<<blocks, threads>>>(
        (uint8_t *)a->data, (uint8_t *)b->data, (uint8_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_U16:
    mul_kernel<uint16_t><<<blocks, threads>>>(
        (uint16_t *)a->data, (uint16_t *)b->data, (uint16_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_U32:
    mul_kernel<uint32_t><<<blocks, threads>>>(
        (uint32_t *)a->data, (uint32_t *)b->data, (uint32_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_U64:
    mul_kernel<uint64_t><<<blocks, threads>>>(
        (uint64_t *)a->data, (uint64_t *)b->data, (uint64_t *)out->data, meta);
    break;
  case INSIGHT_DTYPE_BOOL:
    mul_kernel<bool><<<blocks, threads>>>((bool *)a->data, (bool *)b->data,
                                          (bool *)out->data, meta);
    break;
  case INSIGHT_DTYPE_C32:
    mul_c32_kernel<<<blocks, threads>>>((cuFloatComplex *)a->data,
                                        (cuFloatComplex *)b->data,
                                        (cuFloatComplex *)out->data, meta);
    break;
  case INSIGHT_DTYPE_C64:
    mul_c64_kernel<<<blocks, threads>>>((cuDoubleComplex *)a->data,
                                        (cuDoubleComplex *)b->data,
                                        (cuDoubleComplex *)out->data, meta);
    break;
  case INSIGHT_DTYPE_F16:
    mul_f16_kernel<<<blocks, threads>>>((__half *)a->data, (__half *)b->data,
                                        (__half *)out->data, meta);
    break;
  case INSIGHT_DTYPE_BF16:
    mul_bf16_kernel<<<blocks, threads>>>((__nv_bfloat16 *)a->data,
                                         (__nv_bfloat16 *)b->data,
                                         (__nv_bfloat16 *)out->data, meta);
    break;
  default:
    free_elementwise_metadata(meta);
    gpu_set_last_error("mul: unsupported dtype");
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

// ============================================================================
// Kernel Registration
// ============================================================================

REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_F32, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_F64, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_I8, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_I16, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_I32, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_I64, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_U8, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_U16, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_U32, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_U64, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_BOOL, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_C32, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_C64, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_F16, mul_kernel_gpu);
REGISTER_IXUCA_KERNEL(mul, INSIGHT_DTYPE_BF16, mul_kernel_gpu);
