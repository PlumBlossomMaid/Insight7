// backends/cpu/kernels/unary/clip.cpp
/**
 * @file clip.cpp
 * @brief CPU kernel for value clipping.
 *
 * Computes clip(x) = min(max(x, lo), hi) elementwise with stride support.
 * The bounds are passed as double* in inputs[1] (lo) and inputs[2] (hi)
 * and cast to the input dtype internally.
 *
 * @param inputs  [0] = InsightArray* input, [1] = double* lo, [2] = double* hi
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <algorithm>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

C_Status clip_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("clip: null array pointer");
    return C_FAILED;
  }
  if (!inputs[1] || !inputs[2]) {
    cpu_set_last_error("clip: bounds are null");
    return C_FAILED;
  }

  double lo = *static_cast<double *>(inputs[1]);
  double hi = *static_cast<double *>(inputs[2]);

  if (x->numel != out->numel) {
    cpu_set_last_error("clip: shape mismatch");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_I8:
    UNARY_KERNEL_LOOP(int8_t, ([lo, hi](int8_t v) {
      return static_cast<int8_t>(
          std::min(std::max(static_cast<double>(v), lo), hi));
    }));
    break;
  case INSIGHT_DTYPE_I16:
    UNARY_KERNEL_LOOP(int16_t, ([lo, hi](int16_t v) {
      return static_cast<int16_t>(
          std::min(std::max(static_cast<double>(v), lo), hi));
    }));
    break;
  case INSIGHT_DTYPE_I32:
    UNARY_KERNEL_LOOP(int32_t, ([lo, hi](int32_t v) {
      return static_cast<int32_t>(
          std::min(std::max(static_cast<double>(v), lo), hi));
    }));
    break;
  case INSIGHT_DTYPE_I64:
    UNARY_KERNEL_LOOP(int64_t, ([lo, hi](int64_t v) {
      return static_cast<int64_t>(
          std::min(std::max(static_cast<double>(v), lo), hi));
    }));
    break;
  case INSIGHT_DTYPE_U8:
    UNARY_KERNEL_LOOP(uint8_t, ([lo, hi](uint8_t v) {
      return static_cast<uint8_t>(
          std::min(std::max(static_cast<double>(v), lo), hi));
    }));
    break;
  case INSIGHT_DTYPE_U16:
    UNARY_KERNEL_LOOP(uint16_t, ([lo, hi](uint16_t v) {
      return static_cast<uint16_t>(
          std::min(std::max(static_cast<double>(v), lo), hi));
    }));
    break;
  case INSIGHT_DTYPE_U32:
    UNARY_KERNEL_LOOP(uint32_t, ([lo, hi](uint32_t v) {
      return static_cast<uint32_t>(
          std::min(std::max(static_cast<double>(v), lo), hi));
    }));
    break;
  case INSIGHT_DTYPE_U64:
    UNARY_KERNEL_LOOP(uint64_t, ([lo, hi](uint64_t v) {
      return static_cast<uint64_t>(
          std::min(std::max(static_cast<double>(v), lo), hi));
    }));
    break;
  case INSIGHT_DTYPE_F32:
    UNARY_KERNEL_LOOP(float, ([lo, hi](float v) {
      return static_cast<float>(
          std::min(std::max(static_cast<double>(v), lo), hi));
    }));
    break;
  case INSIGHT_DTYPE_F64:
    UNARY_KERNEL_LOOP(double, ([lo, hi](double v) {
      return std::min(std::max(v, lo), hi);
    }));
    break;
  case INSIGHT_DTYPE_F16:
    UNARY_HALF_LOOP(uint16_t, insight::f16_to_f32, insight::f32_to_f16,
                    ([lo, hi](float v) {
                      return static_cast<float>(
                          std::min(std::max(static_cast<double>(v), lo), hi));
                    }));
    break;
  case INSIGHT_DTYPE_BF16:
    UNARY_HALF_LOOP(uint16_t, insight::bf16_to_f32, insight::f32_to_bf16,
                    ([lo, hi](float v) {
                      return static_cast<float>(
                          std::min(std::max(static_cast<double>(v), lo), hi));
                    }));
    break;

  default:
    cpu_set_last_error("clip: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(clip, INSIGHT_DTYPE_I8, clip_kernel_cpu);
REGISTER_CPU_KERNEL(clip, INSIGHT_DTYPE_I16, clip_kernel_cpu);
REGISTER_CPU_KERNEL(clip, INSIGHT_DTYPE_I32, clip_kernel_cpu);
REGISTER_CPU_KERNEL(clip, INSIGHT_DTYPE_I64, clip_kernel_cpu);
REGISTER_CPU_KERNEL(clip, INSIGHT_DTYPE_U8, clip_kernel_cpu);
REGISTER_CPU_KERNEL(clip, INSIGHT_DTYPE_U16, clip_kernel_cpu);
REGISTER_CPU_KERNEL(clip, INSIGHT_DTYPE_U32, clip_kernel_cpu);
REGISTER_CPU_KERNEL(clip, INSIGHT_DTYPE_U64, clip_kernel_cpu);
REGISTER_CPU_KERNEL(clip, INSIGHT_DTYPE_F32, clip_kernel_cpu);
REGISTER_CPU_KERNEL(clip, INSIGHT_DTYPE_F64, clip_kernel_cpu);
REGISTER_CPU_KERNEL(clip, INSIGHT_DTYPE_F16, clip_kernel_cpu);
REGISTER_CPU_KERNEL(clip, INSIGHT_DTYPE_BF16, clip_kernel_cpu);
