// backends/cpu/kernels/random/multinomial.cpp
/**
 * @file multinomial.cpp
 * @brief CPU kernel for multinomial sampling.
 *
 * Draws num_samples indices from categories 0..n-1 with probabilities
 * proportional to probs (inverse-CDF sampling). With replacement=false
 * each category is drawn at most once.
 *
 * @param inputs  [0] = InsightArray* output array (int64, shape [num_samples])
 *                [1] = InsightArray* probs array (1D, float)
 *                [2] = int64_t* num_samples
 *                [3] = int* replacement (0/1)
 *                [4] = uint64_t* device seed (unused on CPU; global RNG used)
 * @param outputs [0] = InsightArray* output array
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "common.h"

#include <vector>

template <typename P>
static C_Status multinomial_impl(InsightArray *out, InsightArray *probs,
                                 int64_t num_samples, int replacement) {
  int64_t n = probs->numel;
  if (n == 0 || num_samples == 0)
    return C_SUCCESS;

  // Copy probabilities to a working buffer (continuous, normalized).
  std::vector<double> w(n);
  const P *p_src = static_cast<const P *>(probs->data);
  int64_t p_off = probs->offset;
  for (int64_t i = 0; i < n; ++i) {
    w[i] = static_cast<double>(p_src[p_off + i]);
  }

  int64_t *out_data = static_cast<int64_t *>(out->data);

  for (int64_t s = 0; s < num_samples; ++s) {
    // Renormalize over remaining categories.
    double total = 0.0;
    for (int64_t i = 0; i < n; ++i)
      total += w[i];
    if (total <= 0.0) {
      cpu_set_last_error("multinomial: total probability must be positive");
      return C_FAILED;
    }

    // Inverse-CDF sample: pick smallest i with cumulative >= u * total.
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(
                   cpu_get_rng()) *
               total;
    double cum = 0.0;
    int64_t picked = n - 1;
    for (int64_t i = 0; i < n; ++i) {
      cum += w[i];
      if (cum >= u) {
        picked = i;
        break;
      }
    }
    out_data[s] = picked;

    if (!replacement) {
      w[picked] = 0.0; // draw without replacement
    }
  }

  return C_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

C_Status multinomial_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *probs = (InsightArray *)inputs[1];

  if (!out || !probs) {
    cpu_set_last_error("multinomial: null array pointer");
    return C_FAILED;
  }
  if (!inputs[2] || !inputs[3]) {
    cpu_set_last_error("multinomial: num_samples or replacement is null");
    return C_FAILED;
  }

  int64_t num_samples = *(int64_t *)inputs[2];
  int replacement = *(int *)inputs[3];

  switch (probs->dtype) {
  case INSIGHT_DTYPE_F32:
    return multinomial_impl<float>(out, probs, num_samples, replacement);
  case INSIGHT_DTYPE_F64:
    return multinomial_impl<double>(out, probs, num_samples, replacement);
  default:
    cpu_set_last_error("multinomial: unsupported dtype (need f32/f64)");
    return C_FAILED;
  }
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(multinomial, INSIGHT_DTYPE_I64, multinomial_kernel_cpu);