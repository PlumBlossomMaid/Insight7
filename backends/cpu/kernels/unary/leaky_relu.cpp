// backends/cpu/kernels/unary/leaky_relu.cpp
/**
 * @file leaky_relu.cpp
 * @brief CPU kernel for Leaky ReLU activation.
 *
 * Computes leaky_relu(x) = max(x, alpha * x) elementwise with stride support.
 * The alpha slope is passed as a double* in inputs[1] and cast to the
 * input dtype internally.
 *
 * @param inputs  [0] = InsightArray* input, [1] = double* alpha
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

C_Status leaky_relu_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *x = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!x || !out) {
    cpu_set_last_error("leaky_relu: null array pointer");
    return C_FAILED;
  }
  if (!inputs[1]) {
    cpu_set_last_error("leaky_relu: alpha is null");
    return C_FAILED;
  }

  double alpha = *static_cast<double *>(inputs[1]);

  if (x->numel != out->numel) {
    cpu_set_last_error("leaky_relu: shape mismatch");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32:
    UNARY_KERNEL_LOOP(float, [alpha](float v) {
      return v > 0.0f ? v : static_cast<float>(alpha) * v;
    });
    break;
  case INSIGHT_DTYPE_F64:
    UNARY_KERNEL_LOOP(double,
                      [alpha](double v) { return v > 0.0 ? v : alpha * v; });
    break;
  case INSIGHT_DTYPE_F16:
    UNARY_HALF_LOOP(uint16_t, insight::f16_to_f32, insight::f32_to_f16,
                    [alpha](float v) {
                      return v > 0.0f ? v : static_cast<float>(alpha) * v;
                    });
    break;
  case INSIGHT_DTYPE_BF16:
    UNARY_HALF_LOOP(uint16_t, insight::bf16_to_f32, insight::f32_to_bf16,
                    [alpha](float v) {
                      return v > 0.0f ? v : static_cast<float>(alpha) * v;
                    });
    break;

  default:
    cpu_set_last_error(
        "leaky_relu: only float32, float64, f16 and bf16 supported");
    return C_FAILED;
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(leaky_relu, INSIGHT_DTYPE_F32, leaky_relu_kernel_cpu);
REGISTER_CPU_KERNEL(leaky_relu, INSIGHT_DTYPE_F64, leaky_relu_kernel_cpu);
REGISTER_CPU_KERNEL(leaky_relu, INSIGHT_DTYPE_F16, leaky_relu_kernel_cpu);
REGISTER_CPU_KERNEL(leaky_relu, INSIGHT_DTYPE_BF16, leaky_relu_kernel_cpu);
