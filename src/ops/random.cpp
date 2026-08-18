// src/ops/random.cpp
/**
 * @file random.cpp
 * @brief Random number generation operations.
 *
 * Provides various random distributions and seed management.
 * Supports both CPU and GPU backends with automatic device discovery.
 */

#include "insight/ops/random.h"
#include "insight/core/exception.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include "insight/core/place.h"
#include "insight/init.h"
#include <atomic>
#include <vector>

namespace ins {

static std::atomic<uint64_t> g_global_base_seed{0};

// ============================================================================
// Seed Management
// ============================================================================

/**
 * @brief Set random seed for all available devices.
 *
 * Each device receives a unique seed: base_seed + device_id.
 * This ensures statistical independence while maintaining reproducibility.
 *
 * @param base_seed Base seed value
 */
void seed(uint64_t base_seed) {
  g_global_base_seed.store(base_seed);

  auto devices = get_available_devices();
  for (const auto &dev : devices) {
    uint64_t device_seed = base_seed + static_cast<uint64_t>(dev.device_id());
    OpRegistry::launch_schema("set_seed", dev, DType::U64,
                              {OpRegistry::scalar_arg(&device_seed)}, {});
  }
}

/**
 * @brief Get the current global base seed.
 *
 * @return uint64_t Current base seed
 */
uint64_t get_seed() { return g_global_base_seed.load(); }

// ============================================================================
// Helper function for random operations
// ============================================================================

static OpSchema random_schema(const char *name) {
  return OpSchema(name, OpKind::Creation, {},
                  {{"out", OpArgumentKind::Array,
                    OpArgumentAccess::WriteOnly}})
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

template <typename... Args>
static Array random_op(const char *op_name, const Shape &shape, DType dtype,
                       const Place &place, Args &&...extra_args) {
  Array result(shape, dtype, place);
  OpSchema schema = random_schema(op_name);
  ArrayIterator iter = schema.make_creation_iterator({result});
  auto arrays = iter.arrays();

  std::vector<OpRegistry::Arg> inputs;
  inputs.push_back(OpRegistry::array_arg(arrays[0].layout_ptr()));
  (inputs.push_back(OpRegistry::scalar_arg(&extra_args)), ...);

  uint64_t device_seed = get_seed() + static_cast<uint64_t>(place.device_id());
  inputs.push_back(OpRegistry::scalar_arg(&device_seed));

  OpRegistry::launch_schema(
      op_name, place, dtype, inputs,
      {OpRegistry::array_arg(arrays[0].layout_ptr())});

  return result;
}

// ============================================================================
// Basic Distributions
// ============================================================================

/**
 * @brief Generate uniformly distributed random numbers in [0, 1).
 *
 * @param shape Output shape
 * @param dtype Data type (float32 or float64)
 * @param place Device placement
 * @return Array Filled with uniform random numbers
 */
Array rand(const Shape &shape, DType dtype, const Place &place) {
  return random_op("rand", shape, dtype, place);
}

/**
 * @brief Generate normally distributed random numbers (mean=0, std=1).
 *
 * @param shape Output shape
 * @param dtype Data type (float32 or float64)
 * @param place Device placement
 * @return Array Filled with normal random numbers
 */
Array randn(const Shape &shape, DType dtype, const Place &place) {
  return random_op("randn", shape, dtype, place);
}

/**
 * @brief Generate random integers in [low, high).
 *
 * @param low Lower bound (inclusive)
 * @param high Upper bound (exclusive)
 * @param shape Output shape
 * @param dtype Data type (int32 or int64)
 * @param place Device placement
 * @return Array Filled with random integers
 */
Array randint(int64_t low, int64_t high, const Shape &shape, DType dtype,
              const Place &place) {
  return random_op("randint", shape, dtype, place, low, high);
}

/**
 * @brief Generate normally distributed random numbers with specified mean and
 * std.
 *
 * @param mean Mean of the distribution
 * @param std Standard deviation of the distribution
 * @param shape Output shape
 * @param dtype Data type (float32 or float64)
 * @param place Device placement
 * @return Array Filled with normal random numbers
 */
Array normal(double mean, double std, const Shape &shape, DType dtype,
             const Place &place) {
  return random_op("normal", shape, dtype, place, mean, std);
}

/**
 * @brief Generate uniformly distributed random numbers in [low, high).
 *
 * @param low Lower bound (inclusive)
 * @param high Upper bound (exclusive)
 * @param shape Output shape
 * @param dtype Data type (float32 or float64)
 * @param place Device placement
 * @return Array Filled with uniform random numbers
 */
Array uniform(double low, double high, const Shape &shape, DType dtype,
              const Place &place) {
  return random_op("uniform", shape, dtype, place, low, high);
}

/**
 * @brief Generate a random permutation of integers 0 to n-1.
 *
 * @param n Number of elements
 * @param dtype Data type (int32 or int64)
 * @param place Device placement
 * @return Array Random permutation
 */
Array randperm(int64_t n, DType dtype, const Place &place) {
  return random_op("randperm", Shape({n}), dtype, place);
}

// ============================================================================
// Like Functions
// ============================================================================

/**
 * @brief Generate uniform random numbers with same shape/dtype/place as input.
 *
 * @param x Input array (for shape, dtype, place)
 * @return Array Filled with uniform random numbers in [0, 1)
 */
Array rand_like(const Array &x) {
  return rand(x.shape(), x.dtype(), x.place());
}

/**
 * @brief Generate normal random numbers with same shape/dtype/place as input.
 *
 * @param x Input array (for shape, dtype, place)
 * @return Array Filled with normal random numbers
 */
Array randn_like(const Array &x) {
  return randn(x.shape(), x.dtype(), x.place());
}

/**
 * @brief Generate random integers with same shape/place as input.
 *
 * @param x Input array (for shape, place)
 * @param low Lower bound (inclusive)
 * @param high Upper bound (exclusive)
 * @param dtype Data type (default: int64)
 * @return Array Filled with random integers
 */
Array randint_like(const Array &x, int64_t low, int64_t high, DType dtype) {
  if (dtype == DType::UNKNOWN) {
    dtype = is_integer(x.dtype()) ? x.dtype() : DType::I64;
  }
  return randint(low, high, x.shape(), dtype, x.place());
}

/**
 * @brief Generate normal random numbers with same shape/place as input.
 *
 * @param x Input array (for shape, place)
 * @param mean Mean of the distribution
 * @param std Standard deviation
 * @param dtype Data type (default: input's dtype if float, else F32)
 * @return Array Filled with normal random numbers
 */
Array normal_like(const Array &x, double mean, double std, DType dtype) {
  if (dtype == DType::UNKNOWN) {
    dtype = is_floating_point(x.dtype()) ? x.dtype() : DType::F32;
  }
  return normal(mean, std, x.shape(), dtype, x.place());
}

/**
 * @brief Generate uniform random numbers with same shape/place as input.
 *
 * @param x Input array (for shape, place)
 * @param low Lower bound (inclusive)
 * @param high Upper bound (exclusive)
 * @param dtype Data type (default: input's dtype if float, else F32)
 * @return Array Filled with uniform random numbers
 */
Array uniform_like(const Array &x, double low, double high, DType dtype) {
  if (dtype == DType::UNKNOWN) {
    dtype = is_floating_point(x.dtype()) ? x.dtype() : DType::F32;
  }
  return uniform(low, high, x.shape(), dtype, x.place());
}

// ============================================================================
// Additional Distributions
// ============================================================================

/**
 * @brief Generate random numbers from exponential distribution.
 *
 * @param scale Scale parameter (beta = 1/lambda)
 * @param shape Output shape
 * @param dtype Data type (float32 or float64)
 * @param place Device placement
 * @return Array Filled with exponential random numbers
 */
Array exponential(double scale, const Shape &shape, DType dtype,
                  const Place &place) {
  return random_op("exponential", shape, dtype, place, scale);
}

/**
 * @brief Generate random numbers from gamma distribution.
 *
 * @param shape_param Shape parameter (alpha)
 * @param rate Rate parameter (beta = 1/scale)
 * @param shape Output shape
 * @param dtype Data type (float32 or float64)
 * @param place Device placement
 * @return Array Filled with gamma random numbers
 */
Array gamma(double shape_param, double rate, const Shape &shape, DType dtype,
            const Place &place) {
  return random_op("gamma", shape, dtype, place, shape_param, rate);
}

/**
 * @brief Generate random numbers from beta distribution.
 *
 * @param a Alpha parameter
 * @param b Beta parameter
 * @param shape Output shape
 * @param dtype Data type (float32 or float64)
 * @param place Device placement
 * @return Array Filled with beta random numbers
 */
Array beta(double a, double b, const Shape &shape, DType dtype,
           const Place &place) {
  return random_op("beta", shape, dtype, place, a, b);
}

/**
 * @brief Generate random numbers from binomial distribution.
 *
 * @param n Number of trials
 * @param p Probability of success per trial
 * @param shape Output shape
 * @param dtype Data type (int32 or int64)
 * @param place Device placement
 * @return Array Filled with binomial random numbers
 */
Array binomial(int64_t n, double p, const Shape &shape, DType dtype,
               const Place &place) {
  return random_op("binomial", shape, dtype, place, n, p);
}

/**
 * @brief Generate random numbers from Poisson distribution.
 *
 * @param lam Lambda (mean) parameter
 * @param shape Output shape
 * @param dtype Data type (int32 or int64)
 * @param place Device placement
 * @return Array Filled with Poisson random numbers
 */
Array poisson(double lam, const Shape &shape, DType dtype, const Place &place) {
  return random_op("poisson", shape, dtype, place, lam);
}

/**
 * @brief Generate random numbers from chi-square distribution.
 *
 * @param df Degrees of freedom
 * @param shape Output shape
 * @param dtype Data type (float32 or float64)
 * @param place Device placement
 * @return Array Filled with chi-square random numbers
 */
Array chisquare(double df, const Shape &shape, DType dtype,
                const Place &place) {
  return random_op("chisquare", shape, dtype, place, df);
}

} // namespace ins