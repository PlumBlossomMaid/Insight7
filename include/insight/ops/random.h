// insight/ops/random.h
/**
 * @file random.h
 * @brief Random number generation operations.
 */

#pragma once

#include "insight/core/array.h"
#include "insight/core/dtype.h"
#include "insight/core/place.h"
#include "insight/core/shape.h"
#include <optional>
#include <string>
#include <vector>

namespace ins {

// ============================================================================
// Seed Management
// ============================================================================

/**
 * @brief Set random seed for all available devices.
 *
 * Each device receives base_seed + device_id.
 *
 * @param base_seed Base seed value
 */
void seed(uint64_t base_seed);

/**
 * @brief Get current base seed.
 * @return uint64_t Current base seed
 */
uint64_t get_seed();

// ============================================================================
// Basic Distributions
// ============================================================================

/**
 * @brief Uniform distribution in [0, 1).
 */
Array rand(const Shape &shape, DType dtype = DType::F32,
           const Place &place = get_device());

/**
 * @brief Normal distribution (mean=0, std=1).
 */
Array randn(const Shape &shape, DType dtype = DType::F32,
            const Place &place = get_device());

/**
 * @brief Random integers in [low, high).
 */
Array randint(int64_t low, int64_t high, const Shape &shape,
              DType dtype = DType::I64, const Place &place = get_device());

/**
 * @brief Normal distribution with specified parameters.
 */
Array normal(double mean, double std, const Shape &shape,
             DType dtype = DType::F32, const Place &place = get_device());

/**
 * @brief Uniform distribution in [low, high).
 */
Array uniform(double low, double high, const Shape &shape,
              DType dtype = DType::F32, const Place &place = get_device());

/**
 * @brief Random permutation of [0, n-1].
 */
Array randperm(int64_t n, DType dtype = DType::I64,
               const Place &place = get_device());

// ============================================================================
// Sampling and shuffling
// ============================================================================

/**
 * @brief Sample indices from a probability distribution (Paddle
 * multinomial).
 *
 * Draws num_samples indices from categories 0..n-1 with probabilities
 * proportional to probs (which need not sum to 1). With replacement=false
 * each category is drawn at most once.
 *
 * @param probs 1D array of non-negative weights (any float dtype)
 * @param num_samples Number of indices to draw
 * @param replacement Allow the same index to be drawn multiple times
 * @param place Device placement
 * @return 1D int64 array of sampled indices (shape (num_samples,))
 */
Array multinomial(const Array &probs, int64_t num_samples,
                  bool replacement = false, const Place &place = get_device());

/**
 * @brief Random choice of elements from a 1D array (NumPy choice subset).
 *
 * @param x 1D array of candidates
 * @param num Number of samples to draw
 * @param replace Allow repeats (default true)
 * @param p Optional 1D probability weights (same length as x)
 * @return 1D array of drawn elements (same dtype as x, shape (num,))
 */
Array choice(const Array &x, int64_t num = 1, bool replace = true,
             std::optional<Array> p = std::nullopt);

/**
 * @brief Randomly shuffle the elements of an array along axis 0 (copy).
 *
 * Returns a new array whose axis-0 elements are a random permutation of
 * the input (each row is moved as a whole). The input is unchanged.
 *
 * @param x Input array
 * @return Shuffled copy
 */
Array shuffle(const Array &x);

// ============================================================================
// Like Functions
// ============================================================================

Array rand_like(const Array &x);
Array randn_like(const Array &x);
Array randint_like(const Array &x, int64_t low, int64_t high,
                   DType dtype = DType::UNKNOWN);
Array normal_like(const Array &x, double mean, double std,
                  DType dtype = DType::UNKNOWN);
Array uniform_like(const Array &x, double low, double high,
                   DType dtype = DType::UNKNOWN);

// ============================================================================
// Additional Distributions
// ============================================================================

/**
 * @brief Exponential distribution.
 * @param scale Scale parameter (beta)
 */
Array exponential(double scale, const Shape &shape, DType dtype = DType::F32,
                  const Place &place = get_device());

/**
 * @brief Gamma distribution.
 * @param shape_param Shape parameter (alpha)
 * @param rate Rate parameter (beta)
 */
Array gamma(double shape_param, double rate, const Shape &shape,
            DType dtype = DType::F32, const Place &place = get_device());

/**
 * @brief Beta distribution.
 * @param a Alpha parameter
 * @param b Beta parameter
 */
Array beta(double a, double b, const Shape &shape, DType dtype = DType::F32,
           const Place &place = get_device());

/**
 * @brief Binomial distribution.
 * @param n Number of trials
 * @param p Probability of success
 */
Array binomial(int64_t n, double p, const Shape &shape,
               DType dtype = DType::I64, const Place &place = get_device());

/**
 * @brief Poisson distribution.
 * @param lam Lambda parameter
 */
Array poisson(double lam, const Shape &shape, DType dtype = DType::I64,
              const Place &place = get_device());

/**
 * @brief Chi-square distribution.
 * @param df Degrees of freedom
 */
Array chisquare(double df, const Shape &shape, DType dtype = DType::F32,
                const Place &place = get_device());

} // namespace ins