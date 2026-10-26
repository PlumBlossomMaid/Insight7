// backends/cpu/kernels/random/numpy_rng.h
/**
 * @file numpy_rng.h
 * @brief NumPy 2.4.6 legacy RandomState bit-exact RNG primitives.
 *
 * Ported verbatim from NumPy's legacy MT19937 stack so that fixed-seed
 * output is bit-identical to `np.random.seed(s); np.random.<dist>(...)`:
 *
 *   - randomkit.c:     rk_seed / rk_random / rk_interval / rk_double /
 *                      rk_gauss
 *   - mt19937.c/.h:    mt19937_seed / mt19937_init_by_array /
 *                      mt19937_next / mt19937_next64 / mt19937_next_double
 *   - distributions.c: random_interval / random_bounded_uint64 /
 *                      random_poisson / random_binomial_btpe /
 *                      random_multinomial / random_loggam
 *   - legacy-distributions.c: legacy_gauss / legacy_standard_exponential /
 *                      legacy_standard_gamma / legacy_gamma / legacy_beta /
 *                      legacy_chisquare / legacy_random_binomial /
 *                      legacy_random_poisson / legacy_random_multinomial
 *
 * NumPy is BSD-3-Clause (see randomkit.c header / numpy LICENSE.txt).
 * Source line references are annotated per function.
 */

#ifndef INSIGHT_CPU_NUMPY_RNG_H
#define INSIGHT_CPU_NUMPY_RNG_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <algorithm>

#ifdef __cplusplus
extern "C" {
#endif

#define RK_STATE_LEN 624

// ============================================================================
// Core state (mirrors rk_state in randomkit.h, plus binomial cache)
// ============================================================================

typedef struct s_numpy_rk_state {
  uint32_t key[RK_STATE_LEN];
  int pos;
  double gauss;
  int has_gauss;
  int has_binomial;
  int64_t nsave;
  double psave;
  // binomial parameter cache (distributions.c binomial_t)
  double r, q, fm, m, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
} numpy_rk_state;

// ============================================================================
// Seeding
// ============================================================================

/**
 * @brief Seed the MT19937 state (Knuth's PRNG, as in the Mersenne Twister
 * reference implementation).
 *
 * Source: randomkit.c:177-191 (rk_seed), identical to mt19937.c mt19937_seed.
 */
static inline void numpy_rk_seed(numpy_rk_state *state, uint32_t seed) {
  int pos;
  seed &= 0xffffffffU;
  for (pos = 0; pos < RK_STATE_LEN; pos++) {
    state->key[pos] = seed;
    seed = (1812433253U * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffU;
  }
  state->pos = RK_STATE_LEN;
  state->gauss = 0.0;
  state->has_gauss = 0;
  state->has_binomial = 0;
}

/**
 * @brief Initialize by an array (mt19937_init_by_array).
 *
 * Source: mt19937.c:79-123.
 */
static inline void numpy_mt19937_init_by_array(numpy_rk_state *state,
                                               const uint32_t *init_key,
                                               int key_length) {
  int i = 1;
  int j = 0;
  uint32_t *mt = state->key;
  int k;

  /* init_genrand(state, 19650218UL) */
  mt[0] = 19650218U & 0xffffffffU;
  for (int mti = 1; mti < RK_STATE_LEN; mti++) {
    mt[mti] = (1812433253U * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
    mt[mti] &= 0xffffffffU;
  }
  state->pos = RK_STATE_LEN;

  k = (RK_STATE_LEN > key_length ? RK_STATE_LEN : key_length);
  for (; k; k--) {
    mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) +
            init_key[j] + j;
    mt[i] &= 0xffffffffU;
    i++;
    j++;
    if (i >= RK_STATE_LEN) {
      mt[0] = mt[RK_STATE_LEN - 1];
      i = 1;
    }
    if (j >= key_length) {
      j = 0;
    }
  }
  for (k = RK_STATE_LEN - 1; k; k--) {
    mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) - i;
    mt[i] &= 0xffffffffU;
    i++;
    if (i >= RK_STATE_LEN) {
      mt[0] = mt[RK_STATE_LEN - 1];
      i = 1;
    }
  }
  mt[0] = 0x80000000U; /* MSB is 1; assuring non-zero initial array */
  state->gauss = 0.0;
  state->has_gauss = 0;
  state->has_binomial = 0;
}

// ============================================================================
// Core 32-bit generator
// ============================================================================

/* Magic Mersenne Twister constants (randomkit.c:206-212) */
#define RK_N 624
#define RK_M 397
#define RK_MATRIX_A 0x9908b0dfUL
#define RK_UPPER_MASK 0x80000000UL
#define RK_LOWER_MASK 0x7fffffffUL

/**
 * @brief Generate one 32-bit random integer (MT19937 with tempering).
 *
 * Source: randomkit.c:216-244 (rk_random), identical to mt19937.c next().
 */
static inline uint32_t numpy_rk_random(numpy_rk_state *state) {
  uint32_t y;

  if (state->pos == RK_STATE_LEN) {
    int i;
    for (i = 0; i < RK_N - RK_M; i++) {
      y = (state->key[i] & RK_UPPER_MASK) | (state->key[i + 1] & RK_LOWER_MASK);
      state->key[i] =
          state->key[i + RK_M] ^ (y >> 1) ^ (-(y & 1) & RK_MATRIX_A);
    }
    for (; i < RK_N - 1; i++) {
      y = (state->key[i] & RK_UPPER_MASK) | (state->key[i + 1] & RK_LOWER_MASK);
      state->key[i] =
          state->key[i + (RK_M - RK_N)] ^ (y >> 1) ^ (-(y & 1) & RK_MATRIX_A);
    }
    y = (state->key[RK_N - 1] & RK_UPPER_MASK) | (state->key[0] & RK_LOWER_MASK);
    state->key[RK_N - 1] =
        state->key[RK_M - 1] ^ (y >> 1) ^ (-(y & 1) & RK_MATRIX_A);
    state->pos = 0;
  }
  y = state->key[state->pos++];

  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  return y;
}

static inline uint64_t numpy_rk_uint64(numpy_rk_state *state) {
  uint64_t upper = (uint64_t)numpy_rk_random(state) << 32;
  uint64_t lower = (uint64_t)numpy_rk_random(state);
  return upper | lower;
}

/**
 * @brief Generate a random uint64 in [0, max] (inclusive) using mask
 * rejection.
 *
 * Source: distributions.c:1078-1095 (random_interval).
 */
static inline uint64_t numpy_random_interval(numpy_rk_state *state,
                                             uint64_t max) {
  uint64_t mask, value;
  if (max == 0) {
    return 0;
  }
  mask = max;
  /* Smallest bit mask >= max */
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;

  /* Search a random value in [0..mask] <= max */
  if (max <= 0xffffffffUL) {
    while ((value = (numpy_rk_random(state) & mask)) > max)
      ;
  } else {
    while ((value = (numpy_rk_uint64(state) & mask)) > max)
      ;
  }
  return value;
}

/**
 * @brief Smallest bit mask >= max (distributions.c:1106-1114 gen_mask).
 */
static inline uint64_t numpy_gen_mask(uint64_t max) {
  uint64_t mask = max;
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;
  return mask;
}

/**
 * @brief Generate a random uint64 in [off, off+rng] (inclusive) using
 * masked rejection, exactly as NumPy's `random_bounded_uint64` with
 * use_masked=True.
 *
 * Source: distributions.c:1421-1443 (random_bounded_uint64) +
 * distributions.c:1140-1154 (bounded_masked_uint64) +
 * distributions.c:1177-1196 (buffered_bounded_masked_uint32).
 */
static inline uint64_t numpy_random_bounded_uint64(numpy_rk_state *state,
                                                   uint64_t off, uint64_t rng,
                                                   uint64_t mask) {
  if (rng == 0) {
    return off;
  } else if (rng <= 0xFFFFFFFFUL) {
    /* 32-bit path */
    if (rng == 0xFFFFFFFFUL) {
      return off + (uint64_t)numpy_rk_random(state);
    }
    uint32_t val;
    while ((val = (uint32_t)(numpy_rk_random(state) & (uint32_t)mask)) > rng)
      ;
    return off + (uint64_t)val;
  } else if (rng == 0xFFFFFFFFFFFFFFFFULL) {
    return off + numpy_rk_uint64(state);
  } else {
    /* 64-bit masked rejection */
    uint64_t val;
    while ((val = (numpy_rk_uint64(state) & mask)) > rng)
      ;
    return off + val;
  }
}

/**
 * @brief Uniform double in [0, 1), 53-bit fixed point.
 *
 * Source: randomkit.c:486-490 (rk_double); identical to
 * mt19937.h mt19937_next_double.
 */
static inline double numpy_rk_double(numpy_rk_state *state) {
  int32_t a = (int32_t)(numpy_rk_random(state) >> 5);
  int32_t b = (int32_t)(numpy_rk_random(state) >> 6);
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

/**
 * @brief Standard normal via the polar method, with a cached second value.
 *
 * Source: randomkit.c:498-523 (rk_gauss); identical to
 * legacy-distributions.c legacy_gauss.
 */
static inline double numpy_rk_gauss(numpy_rk_state *state) {
  if (state->has_gauss) {
    const double tmp = state->gauss;
    state->gauss = 0.0;
    state->has_gauss = 0;
    return tmp;
  } else {
    double f, x1, x2, r2;
    do {
      x1 = 2.0 * numpy_rk_double(state) - 1.0;
      x2 = 2.0 * numpy_rk_double(state) - 1.0;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);
    /* Polar method (Box-Muller variant) */
    f = sqrt(-2.0 * log(r2) / r2);
    state->gauss = f * x1;
    state->has_gauss = 1;
    return f * x2;
  }
}

// ============================================================================
// Legacy distributions (legacy-distributions.c)
// ============================================================================

/**
 * @brief Standard exponential: -log(1 - U), U in [0,1).
 *
 * Source: legacy-distributions.c:42-45.
 */
static inline double numpy_legacy_standard_exponential(numpy_rk_state *state) {
  return -log(1.0 - numpy_rk_double(state));
}

/**
 * @brief Standard gamma (shape >= 0).
 *
 * Source: legacy-distributions.c:47-90.
 */
static inline double numpy_legacy_standard_gamma(numpy_rk_state *state,
                                                 double shape) {
  double b, c;
  double U, V, X, Y;

  if (shape == 1.0) {
    return numpy_legacy_standard_exponential(state);
  } else if (shape == 0.0) {
    return 0.0;
  } else if (shape < 1.0) {
    for (;;) {
      U = numpy_rk_double(state);
      V = numpy_legacy_standard_exponential(state);
      if (U <= 1.0 - shape) {
        X = pow(U, 1. / shape);
        if (X <= V) {
          return X;
        }
      } else {
        Y = -log((1 - U) / shape);
        X = pow(1.0 - shape + shape * Y, 1. / shape);
        if (X <= (V + Y)) {
          return X;
        }
      }
    }
  } else {
    b = shape - 1. / 3.;
    c = 1. / sqrt(9 * b);
    for (;;) {
      do {
        X = numpy_rk_gauss(state);
        V = 1.0 + c * X;
      } while (V <= 0.0);

      V = V * V * V;
      U = numpy_rk_double(state);
      if (U < 1.0 - 0.0331 * (X * X) * (X * X))
        return (b * V);
      if (log(U) < 0.5 * X * X + b * (1. - V + log(V)))
        return (b * V);
    }
  }
}

/**
 * @brief Gamma with scale (legacy-distributions.c:92-94).
 */
static inline double numpy_legacy_gamma(numpy_rk_state *state, double shape,
                                        double scale) {
  return scale * numpy_legacy_standard_gamma(state, shape);
}

/**
 * @brief Beta (legacy-distributions.c:188-220).
 */
static inline double numpy_legacy_beta(numpy_rk_state *state, double a,
                                       double b) {
  double Ga, Gb;

  if ((a <= 1.0) && (b <= 1.0)) {
    double U, V, X, Y;
    /* Johnk's algorithm */
    while (1) {
      U = numpy_rk_double(state);
      V = numpy_rk_double(state);
      X = pow(U, 1.0 / a);
      Y = pow(V, 1.0 / b);

      if ((X + Y) <= 1.0) {
        if (X + Y > 0) {
          return X / (X + Y);
        } else {
          double logX = log(U) / a;
          double logY = log(V) / b;
          double logM = logX > logY ? logX : logY;
          logX -= logM;
          logY -= logM;
          return exp(logX - log(exp(logX) + exp(logY)));
        }
      }
    }
  } else {
    Ga = numpy_legacy_standard_gamma(state, a);
    Gb = numpy_legacy_standard_gamma(state, b);
    return Ga / (Ga + Gb);
  }
}

/**
 * @brief Chi-square (legacy-distributions.c:111-113).
 */
static inline double numpy_legacy_chisquare(numpy_rk_state *state, double df) {
  return 2.0 * numpy_legacy_standard_gamma(state, df / 2.0);
}

/**
 * @brief Log-gamma function (distributions.c:344-377 random_loggam),
 * used by the btpe binomial and Poisson-PTRS acceptance steps.
 */
static inline double numpy_random_loggam(double x) {
  double x0, x2, lg2pi, gl, gl0;
  int k, n;

  static double a[10] = {
      8.333333333333333e-02, -2.777777777777778e-03, 7.936507936507937e-04,
      -5.952380952380952e-04, 8.417508417508418e-04, -1.917526917526918e-03,
      6.410256410256410e-03, -2.955065359477124e-02, 1.796443723688307e-01,
      -1.39243221690590e+00};

  if ((x == 1.0) || (x == 2.0)) {
    return 0.0;
  } else if (x < 7.0) {
    n = (int)(7 - x);
  } else {
    n = 0;
  }
  x0 = x + n;
  x2 = (1.0 / x0) * (1.0 / x0);
  /* log(2 * M_PI) */
  lg2pi = 1.8378770664093453e+00;
  gl0 = a[9];
  for (k = 8; k >= 0; k--) {
    gl0 *= x2;
    gl0 += a[k];
  }
  gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * log(x0) - x0;
  if (x < 7.0) {
    for (k = 1; k <= n; k++) {
      gl -= log(x0 - 1.0);
      x0 -= 1.0;
    }
  }
  return gl;
}

/**
 * @brief Poisson (distributions.c:550-620 random_poisson_mult / ptrs).
 */
static inline int64_t numpy_random_poisson(numpy_rk_state *state, double lam) {
  if (lam >= 10) {
    /* Transformed rejection (Hoermann) */
    int64_t k;
    double U, V, slam, loglam, a, b, invalpha, vr, us;

    slam = sqrt(lam);
    loglam = log(lam);
    b = 0.931 + 2.53 * slam;
    a = -0.059 + 0.02483 * b;
    invalpha = 1.1239 + 1.1328 / (b - 3.4);
    vr = 0.9277 - 3.6224 / (b - 2);

    while (1) {
      U = numpy_rk_double(state) - 0.5;
      V = numpy_rk_double(state);
      us = 0.5 - fabs(U);
      k = (int64_t)floor((2 * a / us + b) * U + lam + 0.43);
      if ((us >= 0.07) && (V <= vr)) {
        return k;
      }
      if ((k < 0) || ((us < 0.013) && (V > us))) {
        continue;
      }
      if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <=
          (-lam + (double)k * loglam -
           numpy_random_loggam((double)k + 1))) {
        return k;
      }
    }
  } else if (lam == 0) {
    return 0;
  } else {
    /* Multiplication method */
    int64_t X = 0;
    double prod = 1.0;
    double enlam = exp(-lam);
    while (1) {
      double U = numpy_rk_double(state);
      prod *= U;
      if (prod > enlam) {
        X += 1;
      } else {
        return X;
      }
    }
  }
}

/**
 * @brief Binomial via inversion (distributions.c:803-843
 * random_binomial_inversion).
 */
static inline int64_t numpy_random_binomial_inversion(numpy_rk_state *state,
                                                      int64_t n, double p) {
  double q, qn, np, px, U;
  int64_t X, bound;

  q = 1.0 - p;
  qn = exp(n * log(q));
  np = n * p;
  bound = (int64_t)std::min((double)n, np + 10.0 * sqrt(np * q + 1));
  X = 0;
  px = qn;
  U = numpy_rk_double(state);
  while (U > px) {
    X++;
    if (X > bound) {
      X = 0;
      px = qn;
      U = numpy_rk_double(state);
    } else {
      U -= px;
      px = ((n - X + 1) * p * px) / (X * q);
    }
  }
  return X;
}

/**
 * @brief Binomial via BTPE (distributions.c:622-801 random_binomial_btpe).
 */
static inline int64_t numpy_random_binomial_btpe(numpy_rk_state *state,
                                                 int64_t n, double p) {
  double r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
  double a, u, v, s, F, rho, t, A, nrq, x1, x2, f1, f2, z, z2, w, w2, x;
  int64_t m, y, k;

  /* initialize */
  r = std::min(p, 1.0 - p);
  q = 1.0 - r;
  fm = n * r + r;
  m = (int64_t)floor(fm);
  p1 = floor(2.195 * sqrt(n * r * q) - 4.6 * q) + 0.5;
  xm = m + 0.5;
  xl = xm - p1;
  xr = xm + p1;
  c = 0.134 + 20.5 / (15.3 + m);
  a = (fm - xl) / (fm - xl * r);
  laml = a * (1.0 + a / 2.0);
  a = (xr - fm) / (xr * q);
  lamr = a * (1.0 + a / 2.0);
  p2 = p1 * (1.0 + 2.0 * c);
  p3 = p2 + c / laml;
  p4 = p3 + c / lamr;

Step10:
  nrq = n * r * q;
  u = numpy_rk_double(state) * p4;
  v = numpy_rk_double(state);
  if (u > p1)
    goto Step20;
  y = (int64_t)floor(xm - p1 * v + u);
  goto Step60;

Step20:
  if (u > p2)
    goto Step30;
  x = xl + (u - p1) / c;
  v = v * c + 1.0 - fabs(m - x + 0.5) / p1;
  if (v > 1.0)
    goto Step10;
  y = (int64_t)floor(x);
  goto Step50;

Step30:
  if (u > p3)
    goto Step40;
  y = (int64_t)floor(xl + log(v) / laml);
  /* Reject if v==0.0 since previous cast is undefined */
  if ((y < 0) || (v == 0.0))
    goto Step10;
  v = v * (u - p2) * laml;
  goto Step50;

Step40:
  y = (int64_t)floor(xr - log(v) / lamr);
  /* Reject if v==0.0 since previous cast is undefined */
  if ((y > n) || (v == 0.0))
    goto Step10;
  v = v * (u - p3) * lamr;

Step50:
  k = (int64_t)llabs(y - m);
  if ((k > 20) && (k < ((nrq) / 2.0 - 1)))
    goto Step52;

  s = r / q;
  a = s * (n + 1);
  F = 1.0;
  if (m < y) {
    for (int64_t i = m + 1; i <= y; i++) {
      F *= (a / i - s);
    }
  } else if (m > y) {
    for (int64_t i = y + 1; i <= m; i++) {
      F /= (a / i - s);
    }
  }
  if (v > F)
    goto Step10;
  goto Step60;

Step52:
  rho =
      (k / (nrq)) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
  t = -k * k / (2 * nrq);
  A = log(v);
  if (A < (t - rho))
    goto Step60;
  if (A > (t + rho))
    goto Step10;

  x1 = (double)y + 1;
  f1 = (double)m + 1;
  z = (double)n + 1 - (double)m;
  w = (double)n - (double)y + 1;
  x2 = x1 * x1;
  f2 = f1 * f1;
  z2 = z * z;
  w2 = w * w;
  if (A > (xm * log(f1 / x1) + (n - m + 0.5) * log(z / w) +
           (y - m) * log(w * r / (x1 * q)) +
           (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) / f1 /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) / z /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) / x1 /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) / w /
               166320.)) {
    goto Step10;
  }

Step60:
  if (p > 0.5) {
    y = n - y;
  }
  return y;
}

/**
 * @brief Binomial, choosing inversion or BTPE like the legacy path
 * (legacy-distributions.c:270-300 legacy_random_binomial_original).
 *
 * NumPy's legacy path branches on `p*n <= 30` (inversion) vs BTPE, with
 * the p>0.5 case handled symmetrically. `npy_log1p` is log(1+x).
 */
static inline int64_t numpy_legacy_random_binomial(numpy_rk_state *state,
                                                   int64_t n, double p) {
  double q;

  if (p <= 0.5) {
    if (p * n <= 30.0) {
      return numpy_random_binomial_inversion(state, n, p);
    } else {
      return numpy_random_binomial_btpe(state, n, p);
    }
  } else {
    q = 1.0 - p;
    if (q * n <= 30.0) {
      return n - numpy_random_binomial_inversion(state, n, q);
    } else {
      return n - numpy_random_binomial_btpe(state, n, q);
    }
  }
}

/**
 * @brief Multinomial: sequential binomial draws with remaining probability
 * (distributions.c:1729-1747 random_multinomial).
 *
 * @param n Total number of trials
 * @param mnix Output counts per category (length d)
 * @param pix Probabilities per category (length d)
 * @param d   Number of categories
 */
static inline void numpy_random_multinomial(numpy_rk_state *state, int64_t n,
                                            int64_t *mnix, const double *pix,
                                            int64_t d) {
  double remaining_p = 1.0;
  int64_t j;
  int64_t dn = n;
  for (j = 0; j < (d - 1); j++) {
    mnix[j] = numpy_legacy_random_binomial(state, dn, pix[j] / remaining_p);
    dn = dn - mnix[j];
    if (dn <= 0) {
      break;
    }
    remaining_p -= pix[j];
  }
  if (dn > 0) {
    mnix[d - 1] = dn;
  }
}

#ifdef __cplusplus
}
#endif

#endif // INSIGHT_CPU_NUMPY_RNG_H
