// backends/cuda/kernels/cast/cast_c32.cu
/**
 * @file cast_c32.cu
 * @brief CUDA kernel for casting from c32 to all types.
 *
 * Copies layout from source to destination and converts data.
 *
 * @param inputs  [0] = InsightArray* source
 *                [1] = int32_t* target_dtype
 * @param outputs [0] = InsightArray* destination
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status cast_c32_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *src = (InsightArray *)inputs[0];
  InsightArray *dst = (InsightArray *)outputs[0];
  int32_t target_dtype = *(int32_t *)inputs[1];

  if (!src || !dst) {
    gpu_set_last_error("cast_c32: null array pointer");
    return C_FAILED;
  }

  // Copy layout from source to destination
  copy_layout(dst, src);
  dst->dtype = target_dtype;

  // Data conversion
  int64_t n = src->numel;
  const cuFloatComplex *src_data = (const cuFloatComplex *)src->data;

  int blocks, threads;
  cast_launch_config(n, blocks, threads);

  switch (target_dtype) {
  case INSIGHT_DTYPE_BOOL: {
    cast_c32_to_bool_kernel<<<blocks, threads>>>(src_data, (bool *)dst->data,
                                                 n);
    break;
  }
  case INSIGHT_DTYPE_F32: {
    cast_c32_to_f32_kernel<<<blocks, threads>>>(src_data, (float *)dst->data,
                                                n);
    break;
  }
  case INSIGHT_DTYPE_F64: {
    cast_c32_to_f64_kernel<<<blocks, threads>>>(src_data, (double *)dst->data,
                                                n);
    break;
  }
  case INSIGHT_DTYPE_C64: {
    cast_c32_to_c64_kernel<<<blocks, threads>>>(
        src_data, (cuDoubleComplex *)dst->data, n);
    break;
  }
  default:
    gpu_set_last_error("cast_c32: unsupported target type");
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

REGISTER_ILUVATAR_KERNEL(cast, INSIGHT_DTYPE_C32, cast_c32_kernel_gpu);
