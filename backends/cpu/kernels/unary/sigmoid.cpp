// backends/cpu/kernels/unary/sigmoid.cpp
/**
 * @file sigmoid.cpp
 * @brief CPU kernel for Sigmoid activation.
 *
 * Computes sigmoid(x) = 1 / (1 + exp(-x)) elementwise with stride support.
 * Supports floating point data types (including half precision).
 *
 * @param inputs  [0] = InsightArray* input
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

C_Status sigmoid_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("sigmoid: null array pointer");
    return C_FAILED;
  }

  if (x->numel != out->numel) {
    cpu_set_last_error("sigmoid: shape mismatch");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32:
    UNARY_KERNEL_LOOP(float, [](float v) {
      return 1.0f / (1.0f + std::exp(-v));
    });
    break;
  case INSIGHT_DTYPE_F64:
    UNARY_KERNEL_LOOP(double, [](double v) {
      return 1.0 / (1.0 + std::exp(-v));
    });
    break;
  case INSIGHT_DTYPE_F16:
    UNARY_HALF_LOOP(uint16_t, insight::f16_to_f32, insight::f32_to_f16,
                    [](float v) { return 1.0f / (1.0f + std::exp(-v)); });
    break;
  case INSIGHT_DTYPE_BF16:
    UNARY_HALF_LOOP(uint16_t, insight::bf16_to_f32, insight::f32_to_bf16,
                    [](float v) { return 1.0f / (1.0f + std::exp(-v)); });
    break;

  default:
    cpu_set_last_error("sigmoid: only float32, float64, f16 and bf16 supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(sigmoid, INSIGHT_DTYPE_F32, sigmoid_kernel_cpu);
REGISTER_CPU_KERNEL(sigmoid, INSIGHT_DTYPE_F64, sigmoid_kernel_cpu);
REGISTER_CPU_KERNEL(sigmoid, INSIGHT_DTYPE_F16, sigmoid_kernel_cpu);
REGISTER_CPU_KERNEL(sigmoid, INSIGHT_DTYPE_BF16, sigmoid_kernel_cpu);
