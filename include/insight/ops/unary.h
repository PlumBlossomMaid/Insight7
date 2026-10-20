// include/insight/ops/unary.h
/**
 * @file unary.h
 * @brief Unary elementwise operations on Insight arrays.
 *
 * All functions in this file operate elementwise on a single input array
 * and return a new array of the same shape (unless otherwise noted).
 *
 * @see elementwise.h for binary elementwise operations
 * @see reduction.h for reduction operations
 */
#pragma once
#include "insight/core/array.h"

namespace ins {

// ============================================================================
// Basic unary
// ============================================================================

/**
 * @brief Elementwise absolute value.
 * @param x Input array (numeric types)
 * @return Array of same shape with |x_i| for each element
 */
Array abs(const Array &x);

/**
 * @brief Elementwise negation (unary minus).
 * @param x Input array
 * @return Array of same shape with -x_i for each element
 */
Array negative(const Array &x);
inline Array neg(const Array &x) { return negative(x); }

/**
 * @brief Elementwise square.
 * @param x Input array
 * @return Array of same shape with x_i^2 for each element
 */
Array square(const Array &x);
inline Array sqr(const Array &x) { return square(x); }

// ============================================================================
// Exponential / Logarithm
// ============================================================================

/** @brief Elementwise natural exponential (e^x). */
Array exp(const Array &x);
/** @brief Elementwise base-2 exponential (2^x). */
Array exp2(const Array &x);
/** @brief Elementwise expm1 (e^x - 1, accurate for small x). */
Array expm1(const Array &x);

/** @brief Elementwise natural logarithm. */
Array log(const Array &x);
/** @brief Elementwise base-2 logarithm. */
Array log2(const Array &x);
/** @brief Elementwise base-10 logarithm. */
Array log10(const Array &x);
/** @brief Elementwise log1p (ln(1+x), accurate for small x). */
Array log1p(const Array &x);

// ============================================================================
// Power / Root
// ============================================================================

/** @brief Elementwise square root. */
Array sqrt(const Array &x);
/** @brief Elementwise cube root. */
Array cbrt(const Array &x);
/** @brief Elementwise reciprocal (1/x). */
Array reciprocal(const Array &x);

// ============================================================================
// Trigonometric
// ============================================================================

/** @brief Elementwise sine. */
Array sin(const Array &x);
/** @brief Elementwise cosine. */
Array cos(const Array &x);
/** @brief Elementwise tangent. */
Array tan(const Array &x);
/** @brief Elementwise arc sine (inverse sine). */
Array asin(const Array &x);
/** @brief Elementwise arc cosine (inverse cosine). */
Array acos(const Array &x);
/** @brief Elementwise arc tangent (inverse tangent). */
Array atan(const Array &x);

// ============================================================================
// Hyperbolic
// ============================================================================

/** @brief Elementwise hyperbolic sine. */
Array sinh(const Array &x);
/** @brief Elementwise hyperbolic cosine. */
Array cosh(const Array &x);
/** @brief Elementwise hyperbolic tangent. */
Array tanh(const Array &x);
/** @brief Elementwise inverse hyperbolic sine. */
Array asinh(const Array &x);
/** @brief Elementwise inverse hyperbolic cosine. */
Array acosh(const Array &x);
/** @brief Elementwise inverse hyperbolic tangent. */
Array atanh(const Array &x);

// ============================================================================
// Rounding
// ============================================================================

/** @brief Elementwise floor (round toward -infinity). */
Array floor(const Array &x);
/** @brief Elementwise ceil (round toward +infinity). */
Array ceil(const Array &x);
/** @brief Elementwise truncation (round toward zero). */
Array trunc(const Array &x);
/** @brief Elementwise round to nearest integer (halfway cases: round to even).
 */
Array rint(const Array &x);

// ============================================================================
// Sign / Classification
// ============================================================================

/**
 * @brief Elementwise sign function.
 * @return -1 for negative, 0 for zero, +1 for positive (float types)
 */
Array sign(const Array &x);

/** @brief Elementwise NaN check. Returns bool array. */
Array isnan(const Array &x);
/** @brief Elementwise infinity check. Returns bool array. */
Array isinf(const Array &x);
/** @brief Elementwise finite check. Returns bool array. */
Array isfinite(const Array &x);

// ============================================================================
// Complex
// ============================================================================

/** @brief Elementwise complex conjugate. */
Array conj(const Array &x);

/** @brief Elementwise angle/phase of complex or real numbers. Returns F32/F64.
 */
Array angle(const Array &x);

// ============================================================================
// Conversion
// ============================================================================

/** @brief Elementwise degrees to radians conversion. */
Array deg2rad(const Array &x);
/** @brief Elementwise radians to degrees conversion. */
Array rad2deg(const Array &x);

// ============================================================================
// Bitwise unary
// ============================================================================

/** @brief Elementwise logical NOT (boolean input/output). */
Array logical_not(const Array &x);
/** @brief Elementwise bitwise NOT (integer types only). */
Array bitwise_not(const Array &x);
/** @brief Alias for bitwise_not. */
inline Array invert(const Array &x) { return bitwise_not(x); }

// ============================================================================
// Activation functions
// ============================================================================

/**
 * @brief Rectified linear unit: relu(x) = max(x, 0).
 * @param x Input array (numeric types)
 * @return Array of same shape with relu applied elementwise
 */
Array relu(const Array &x);

/**
 * @brief Sigmoid activation: sigmoid(x) = 1 / (1 + exp(-x)).
 * @param x Input array (floating point types)
 * @return Array of same shape with sigmoid applied elementwise
 */
Array sigmoid(const Array &x);

/**
 * @brief Leaky rectified linear unit: leaky_relu(x) = max(x, alpha*x).
 *
 * @param x Input array (floating point types)
 * @param alpha Slope for negative inputs (default 0.01)
 * @return Array of same shape with leaky_relu applied elementwise
 */
Array leaky_relu(const Array &x, double alpha = 0.01);

/**
 * @brief Clip values to a [min, max] range: clip(x) = min(max(x, lo), hi).
 *
 * @param x Input array (numeric types)
 * @param min_val Lower bound
 * @param max_val Upper bound
 * @return Array of same shape with values clamped to the given range
 */
Array clip(const Array &x, double min_val, double max_val);

/** @brief Alias for clip(). */
inline Array clamp(const Array &x, double min_val, double max_val) {
  return clip(x, min_val, max_val);
}

} // namespace ins
