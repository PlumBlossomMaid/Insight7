// backends/cuda/kernels/cast/cast_i32.cu
/**
 * @file cast_i32.cu
 * @brief CUDA kernel for casting from i32 to all types.
 *
 * Copies layout from source to destination and converts data.
 *
 * @param inputs  [0] = InsightArray* source
 *                [1] = int32_t* target_dtype
 * @param outputs [0] = InsightArray* destination
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status cast_i32_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *src = (InsightArray *)inputs[0];
  InsightArray *dst = (InsightArray *)outputs[0];
  int32_t target_dtype = *(int32_t *)inputs[1];

  if (!src || !dst) {
    gpu_set_last_error("cast_i32: null array pointer");
    return C_FAILED;
  }

  // Copy layout from source to destination
  copy_layout(dst, src);
  dst->dtype = target_dtype;

  // Data conversion
  int64_t n = src->numel;
  const int32_t *src_data = (const int32_t *)src->data;

  int blocks, threads;
  cast_launch_config(n, blocks, threads);

  switch (target_dtype) {
  case INSIGHT_DTYPE_BOOL: {
    cast_to_bool_kernel<int32_t>
        <<<blocks, threads>>>(src_data, (bool *)dst->data, n);
    break;
  }
  case INSIGHT_DTYPE_U8: {
    cast_kernel<int32_t, uint8_t>
        <<<blocks, threads>>>(src_data, (uint8_t *)dst->data, n);
    break;
  }
  case INSIGHT_DTYPE_I8: {
    cast_kernel<int32_t, int8_t>
        <<<blocks, threads>>>(src_data, (int8_t *)dst->data, n);
    break;
  }
  case INSIGHT_DTYPE_I16: {
    cast_kernel<int32_t, int16_t>
        <<<blocks, threads>>>(src_data, (int16_t *)dst->data, n);
    break;
  }
  case INSIGHT_DTYPE_I64: {
    cast_kernel<int32_t, int64_t>
        <<<blocks, threads>>>(src_data, (int64_t *)dst->data, n);
    break;
  }
  case INSIGHT_DTYPE_U16: {
    cast_kernel<int32_t, uint16_t>
        <<<blocks, threads>>>(src_data, (uint16_t *)dst->data, n);
    break;
  }
  case INSIGHT_DTYPE_U32: {
    cast_kernel<int32_t, uint32_t>
        <<<blocks, threads>>>(src_data, (uint32_t *)dst->data, n);
    break;
  }
  case INSIGHT_DTYPE_U64: {
    cast_kernel<int32_t, uint64_t>
        <<<blocks, threads>>>(src_data, (uint64_t *)dst->data, n);
    break;
  }
  case INSIGHT_DTYPE_F32: {
    cast_kernel<int32_t, float>
        <<<blocks, threads>>>(src_data, (float *)dst->data, n);
    break;
  }
  case INSIGHT_DTYPE_F64: {
    cast_kernel<int32_t, double>
        <<<blocks, threads>>>(src_data, (double *)dst->data, n);
    break;
  }
  case INSIGHT_DTYPE_C32: {
    cast_kernel<int32_t, cuFloatComplex>
        <<<blocks, threads>>>(src_data, (cuFloatComplex *)dst->data, n);
    break;
  }
  case INSIGHT_DTYPE_C64: {
    cast_kernel<int32_t, cuDoubleComplex>
        <<<blocks, threads>>>(src_data, (cuDoubleComplex *)dst->data, n);
    break;
  }
  default:
    gpu_set_last_error("cast_i32: unsupported target type");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_IXUCA_KERNEL(cast, INSIGHT_DTYPE_I32, cast_i32_kernel_gpu);
