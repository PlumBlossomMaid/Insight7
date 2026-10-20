// backends/cpu/kernels/unary/relu.cpp
/**
 * @file relu.cpp
 * @brief CPU kernel for ReLU activation.
 *
 * Computes relu(x) = max(x, 0) elementwise with stride support.
 * Supports integer and floating point data types.
 *
 * @param inputs  [0] = InsightArray* input
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <algorithm>

#ifdef __cplusplus
extern "C" {
#endif

C_Status relu_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("relu: null array pointer");
    return C_FAILED;
  }

  if (x->numel != out->numel) {
    cpu_set_last_error("relu: shape mismatch");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_I8:
    UNARY_KERNEL_LOOP(int8_t, [](int8_t v) { return v > 0 ? v : 0; });
    break;
  case INSIGHT_DTYPE_I16:
    UNARY_KERNEL_LOOP(int16_t, [](int16_t v) { return v > 0 ? v : 0; });
    break;
  case INSIGHT_DTYPE_I32:
    UNARY_KERNEL_LOOP(int32_t, [](int32_t v) { return v > 0 ? v : 0; });
    break;
  case INSIGHT_DTYPE_I64:
    UNARY_KERNEL_LOOP(int64_t, [](int64_t v) { return v > 0 ? v : 0; });
    break;
  case INSIGHT_DTYPE_F32:
    UNARY_KERNEL_LOOP(float, [](float v) { return v > 0.0f ? v : 0.0f; });
    break;
  case INSIGHT_DTYPE_F64:
    UNARY_KERNEL_LOOP(double, [](double v) { return v > 0.0 ? v : 0.0; });
    break;
  case INSIGHT_DTYPE_F16:
    UNARY_HALF_LOOP(uint16_t, insight::f16_to_f32, insight::f32_to_f16,
                    [](float v) { return v > 0.0f ? v : 0.0f; });
    break;
  case INSIGHT_DTYPE_BF16:
    UNARY_HALF_LOOP(uint16_t, insight::bf16_to_f32, insight::f32_to_bf16,
                    [](float v) { return v > 0.0f ? v : 0.0f; });
    break;

  default:
    cpu_set_last_error("relu: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(relu, INSIGHT_DTYPE_I8, relu_kernel_cpu);
REGISTER_CPU_KERNEL(relu, INSIGHT_DTYPE_I16, relu_kernel_cpu);
REGISTER_CPU_KERNEL(relu, INSIGHT_DTYPE_I32, relu_kernel_cpu);
REGISTER_CPU_KERNEL(relu, INSIGHT_DTYPE_I64, relu_kernel_cpu);
REGISTER_CPU_KERNEL(relu, INSIGHT_DTYPE_F32, relu_kernel_cpu);
REGISTER_CPU_KERNEL(relu, INSIGHT_DTYPE_F64, relu_kernel_cpu);
REGISTER_CPU_KERNEL(relu, INSIGHT_DTYPE_F16, relu_kernel_cpu);
REGISTER_CPU_KERNEL(relu, INSIGHT_DTYPE_BF16, relu_kernel_cpu);
