// backends/cpu/kernels/random/multinomial.cpp
/**
 * @file multinomial.cpp
 * @brief CPU kernel for multinomial index sampling (Paddle semantics).
 *
 * Insight.multinomial(probs, num_samples, replacement) draws `num_samples`
 * category indices from 0..d-1 with probabilities proportional to probs.
 * For bit-exactness with NumPy this matches the index generation inside
 * `np.random.choice(np.arange(d), num_samples, replacement, p=probs)`:
 *
 *   - replacement=true : cdf = cumsum(p); cdf /= cdf[-1];
 *                        u = random_sample(num); idx = searchsorted right
 *                        (mtrand.pyx choice replace=True branch)
 *   - replacement=false: repeated random-sample + cdf rebuild + searchsorted
 *                        + unique dedup until `num_samples` indices are found
 *                        (mtrand.pyx choice replace=False, p branch)
 *
 * `take(x, multinomial(p, n, r))` therefore equals np.random.choice(x, ...).
 */

#include "common.h"
#include "numpy_rng_state.h"

#include <algorithm>
#include <cmath>
#include <vector>

// searchsorted(x, v, side='right'): first index i with cdf[i] > v.
static int64_t searchsorted_right(const std::vector<double> &cdf, double v) {
  return (int64_t)(std::upper_bound(cdf.begin(), cdf.end(), v) - cdf.begin());
}

template <typename P>
static C_Status multinomial_impl(InsightArray *out, InsightArray *probs,
                                 int64_t num_samples, int replacement,
                                 numpy_rk_state *st) {
  int64_t d = probs->numel;
  if (d == 0 || num_samples == 0)
    return C_SUCCESS;

  // Copy probabilities to a double working buffer.
  std::vector<double> p(d);
  const P *p_src = static_cast<const P *>(probs->data);
  int64_t p_off = probs->offset;
  for (int64_t i = 0; i < d; ++i) {
    p[i] = static_cast<double>(p_src[p_off + i]);
  }

  int64_t *out_data = static_cast<int64_t *>(out->data);

  // Verify at least `num_samples` non-zero weights when sampling without
  // replacement (NumPy raises ValueError otherwise).
  if (!replacement) {
    int64_t nz = 0;
    for (int64_t i = 0; i < d; ++i) {
      if (p[i] > 0.0)
        ++nz;
    }
    if (nz < num_samples) {
      cpu_set_last_error(
          "multinomial: fewer non-zero entries in probs than num_samples "
          "when replacement=false");
      return C_FAILED;
    }
  }

  std::vector<double> cdf(d);

  if (replacement) {
    // NumPy: cdf = cumsum(p); cdf /= cdf[-1]
    double acc = 0.0;
    bool any_positive = false;
    for (int64_t i = 0; i < d; ++i) {
      acc += p[i];
      cdf[i] = acc;
      if (p[i] > 0.0)
        any_positive = true;
    }
    if (!any_positive || cdf[d - 1] <= 0.0) {
      cpu_set_last_error("multinomial: total probability must be positive");
      return C_FAILED;
    }
    double total = cdf[d - 1];
    for (int64_t i = 0; i < d; ++i) {
      cdf[i] /= total;
    }

    // NumPy draws all uniform samples first, then searchsorted each.
    std::vector<double> u(num_samples);
    for (int64_t s = 0; s < num_samples; ++s) {
      u[s] = numpy_rk_double(st);
    }
    for (int64_t s = 0; s < num_samples; ++s) {
      out_data[s] = searchsorted_right(cdf, u[s]);
    }
  } else {
    // NumPy replace=False with p: loop until `num_samples` unique indices
    // are found. Each round samples as many uniforms as still missing,
    // zeroes the probabilities already taken, rebuilds the CDF and keeps
    // only fresh unique indices.
    std::vector<double> p_work = p;
    int64_t n_uniq = 0;
    while (n_uniq < num_samples) {
      int64_t missing = num_samples - n_uniq;
      std::vector<double> x(missing);
      for (int64_t i = 0; i < missing; ++i) {
        x[i] = numpy_rk_double(st);
      }
      if (n_uniq > 0) {
        // p[...] = 0 for indices already found (out_data holds them).
        for (int64_t i = 0; i < n_uniq; ++i) {
          p_work[out_data[i]] = 0.0;
        }
      }
      double acc = 0.0;
      bool any_positive = false;
      for (int64_t i = 0; i < d; ++i) {
        acc += p_work[i];
        cdf[i] = acc;
        if (p_work[i] > 0.0)
          any_positive = true;
      }
      if (!any_positive || cdf[d - 1] <= 0.0) {
        cpu_set_last_error("multinomial: total probability must be positive");
        return C_FAILED;
      }
      double total = cdf[d - 1];
      for (int64_t i = 0; i < d; ++i) {
        cdf[i] /= total;
      }

      std::vector<int64_t> fresh;
      fresh.reserve(missing);
      for (int64_t i = 0; i < missing; ++i) {
        fresh.push_back(searchsorted_right(cdf, x[i]));
      }
      // NumPy: _, uniq_idx = np.unique(new, return_index=True);
      //        uniq_idx.sort(); new = new.take(uniq_idx)
      // This keeps first-occurrence order of `new`, NOT a sorted dedup.
      std::vector<int64_t> keep;
      keep.reserve(fresh.size());
      for (int64_t v : fresh) {
        if (std::find(keep.begin(), keep.end(), v) == keep.end()) {
          keep.push_back(v);
        }
      }
      for (size_t i = 0; i < keep.size() && n_uniq < num_samples; ++i) {
        out_data[n_uniq++] = keep[i];
      }
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

  int32_t device_id = out->device_id;
  numpy_rk_state *st = ins::numpy_rng::lock_device(out->device_type, device_id);

  C_Status rc;
  switch (probs->dtype) {
  case INSIGHT_DTYPE_F32:
    rc = multinomial_impl<float>(out, probs, num_samples, replacement, st);
    break;
  case INSIGHT_DTYPE_F64:
    rc = multinomial_impl<double>(out, probs, num_samples, replacement, st);
    break;
  default:
    ins::numpy_rng::unlock_device();
    cpu_set_last_error("multinomial: unsupported dtype (need f32/f64)");
    return C_FAILED;
  }

  ins::numpy_rng::unlock_device();
  return rc;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(multinomial, INSIGHT_DTYPE_I64, multinomial_kernel_cpu);