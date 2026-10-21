// include/insight/ops/reduction.h
#pragma once

#include "insight/core/array.h"
#include <optional>

namespace ins {

// ========== Basic Reduction ==========

/**
 * @brief Sum of array elements along specified axis.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing sum values
 */
Array sum(const Array &x, std::optional<int> axis = std::nullopt,
          bool keepdim = false);

/**
 * @brief Mean of array elements along specified axis.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing mean values
 */
Array mean(const Array &x, std::optional<int> axis = std::nullopt,
           bool keepdim = false);

/**
 * @brief Maximum of array elements along specified axis.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing maximum values
 */
Array max(const Array &x, std::optional<int> axis = std::nullopt,
          bool keepdim = false);

/**
 * @brief Minimum of array elements along specified axis.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing minimum values
 */
Array min(const Array &x, std::optional<int> axis = std::nullopt,
          bool keepdim = false);

/**
 * @brief Product of array elements along specified axis.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing product values
 */
Array prod(const Array &x, std::optional<int> axis = std::nullopt,
           bool keepdim = false);

/**
 * @brief Logical OR of array elements along specified axis.
 * Returns true if any element is non-zero.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing boolean values
 */
Array any(const Array &x, std::optional<int> axis = std::nullopt,
          bool keepdim = false);

/**
 * @brief Logical AND of array elements along specified axis.
 * Returns true only if all elements are non-zero.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing boolean values
 */
Array all(const Array &x, std::optional<int> axis = std::nullopt,
          bool keepdim = false);

// ========== Positional Reduction ==========

/**
 * @brief Returns indices of maximum values along specified axis.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array of indices (dtype: int64)
 */
Array argmax(const Array &x, std::optional<int> axis = std::nullopt,
             bool keepdim = false);

/**
 * @brief Returns indices of minimum values along specified axis.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array of indices (dtype: int64)
 */
Array argmin(const Array &x, std::optional<int> axis = std::nullopt,
             bool keepdim = false);

// ========== Statistical Reduction ==========

/**
 * @brief Variance of array elements along specified axis.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
 * @return Array containing variance values
 */
Array var(const Array &x, std::optional<int> axis = std::nullopt,
          bool keepdim = false, int ddof = 0);

/**
 * @brief Standard deviation of array elements along specified axis.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
 * @return Array containing standard deviation values
 */
Array std(const Array &x, std::optional<int> axis = std::nullopt,
          bool keepdim = false, int ddof = 0);

/**
 * @brief Standard error of the mean along specified axis.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
 * @return Array containing standard error values
 */
Array sem(const Array &x, std::optional<int> axis = std::nullopt,
          bool keepdim = false, int ddof = 0);

// ========== Count ==========

/**
 * @brief Count number of non-zero elements along specified axis.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array of counts (dtype: int64)
 */
Array count_nonzero(const Array &x, std::optional<int> axis = std::nullopt,
                    bool keepdim = false);

// ========== Cumulative ==========

/**
 * @brief Cumulative sum along specified axis.
 *
 * Output dtype is the same as input dtype.
 *
 * @param x Input array
 * @param axis Axis to accumulate (0 <= axis < ndim)
 * @return Array with same shape and dtype as input
 */
Array cumsum(const Array &x, int axis);

/**
 * @brief Cumulative product along specified axis.
 *
 * Output dtype is the same as input dtype.
 *
 * @param x Input array
 * @param axis Axis to accumulate (0 <= axis < ndim)
 * @return Array with same shape and dtype as input
 */
Array cumprod(const Array &x, int axis);

/**
 * @brief Cumulative maximum along specified axis.
 *
 * @param x Input array
 * @param axis Axis to accumulate (0 <= axis < ndim)
 * @return Array with same shape as input
 */
Array cummax(const Array &x, int axis);

/**
 * @brief Cumulative minimum along specified axis.
 *
 * @param x Input array
 * @param axis Axis to accumulate (0 <= axis < ndim)
 * @return Array with same shape as input
 */
Array cummin(const Array &x, int axis);

// ========== Quantile ==========

/**
 * @brief Median value along specified axis.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing median values
 */
Array median(const Array &x, std::optional<int> axis = std::nullopt,
             bool keepdim = false);

/**
 * @brief Quantile values along specified axis (q in [0, 1]).
 *
 * @param x Input array
 * @param q Quantile (single value, 0 <= q <= 1)
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing quantile values
 */
Array quantile(const Array &x, double q, std::optional<int> axis = std::nullopt,
               bool keepdim = false);

/**
 * @brief Quantile values along specified axis (multiple quantiles).
 *
 * @param x Input array
 * @param q Array of quantiles (all values in [0, 1])
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array of shape (len(q),) + output_shape
 */
Array quantile(const Array &x, const Array &q,
               std::optional<int> axis = std::nullopt, bool keepdim = false);

/**
 * @brief Percentile values along specified axis (q in [0, 100]).
 *
 * @param x Input array
 * @param q Percentile (single value, 0 <= q <= 100)
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing percentile values
 */
Array percentile(const Array &x, double q, std::optional<int> axis,
                 bool keepdim);

// ========== NaN-safe Reduction ==========

/**
 * @brief Sum of array elements, ignoring NaN values.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing sum values (NaN if all elements are NaN)
 */
Array nansum(const Array &x, std::optional<int> axis = std::nullopt,
             bool keepdim = false);

/**
 * @brief Mean of array elements, ignoring NaN values.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing mean values (NaN if all elements are NaN)
 */
Array nanmean(const Array &x, std::optional<int> axis = std::nullopt,
              bool keepdim = false);

/**
 * @brief Maximum of array elements, ignoring NaN values.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing maximum values (NaN if all elements are NaN)
 */
Array nanmax(const Array &x, std::optional<int> axis = std::nullopt,
             bool keepdim = false);

/**
 * @brief Minimum of array elements, ignoring NaN values.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing minimum values (NaN if all elements are NaN)
 */
Array nanmin(const Array &x, std::optional<int> axis = std::nullopt,
             bool keepdim = false);

/**
 * @brief Variance of array elements, ignoring NaN values.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
 * @return Array containing variance values (NaN if all elements are NaN)
 */
Array nanvar(const Array &x, std::optional<int> axis = std::nullopt,
             bool keepdim = false, int ddof = 0);

/**
 * @brief Standard deviation of array elements, ignoring NaN values.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @param ddof Delta degrees of freedom (0 for population, 1 for sample)
 * @return Array containing standard deviation values (NaN if all elements are
 * NaN)
 */
Array nanstd(const Array &x, std::optional<int> axis = std::nullopt,
             bool keepdim = false, int ddof = 0);

/**
 * @brief Median of array elements, ignoring NaN values.
 *
 * @param x Input array
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing median values (NaN if all elements are NaN)
 */
Array nanmedian(const Array &x, std::optional<int> axis = std::nullopt,
                bool keepdim = false);

/**
 * @brief Quantile values along specified axis, ignoring NaN values.
 *
 * @param x Input array
 * @param q Quantile (single value, 0 <= q <= 1)
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing quantile values (NaN if all elements are NaN)
 */
Array nanquantile(const Array &x, double q,
                  std::optional<int> axis = std::nullopt, bool keepdim = false);

/**
 * @brief Quantile values along specified axis, ignoring NaN values.
 *
 * @param x Input array
 * @param q Array of quantiles (all values in [0, 1])
 * @param axis Axis to reduce (nullopt means flatten)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array of shape (len(q),) + output_shape
 */
Array nanquantile(const Array &x, const Array &q,
                  std::optional<int> axis = std::nullopt, bool keepdim = false);

/**
 * @brief Count occurrences of each integer value in the input array.
 *
 * @param x Input array of non-negative integers
 * @param weights Optional array of weights for each element in x (same shape as
 * x)
 * @param minlength Minimum length of the output array (output will have at
 * least this many bins)
 * @return Array of counts (dtype: int64) where output[i] is the count of
 * occurrences of integer i in x, weighted by weights if provided
 */
Array bincount(const Array &x, std::optional<Array> weights = std::nullopt,
               int64_t minlength = 0);

// ========== Stable Log-Sum-Exp ==========

/**
 * @brief Log-sum-exp along specified axis.
 *
 * Computes log(sum(exp(x))) along axis in a numerically stable way:
 * m = max(x, axis), result = m + log(sum(exp(x - m))).
 * This avoids overflow/underflow for large or small values.
 *
 * @param x Input array (integer inputs are promoted to float64)
 * @param axis Axis to reduce (nullopt means flatten all)
 * @param keepdim If true, keep reduced dimensions as size 1
 * @return Array containing log-sum-exp values
 */
Array logsumexp(const Array &x, std::optional<int> axis = std::nullopt,
                bool keepdim = false);

// ========== Softmax / Log-Softmax ==========

/**
 * @brief Softmax normalization along the specified axis.
 *
 * Computes softmax(x) = exp(x - m) / sum(exp(x - m)) along axis, where
 * m = max(x, axis) for numerical stability. Output has the same shape
 * as the input.
 *
 * @param x Input array (integer inputs are promoted to float64)
 * @param axis Axis along which to normalize (default -1, the last axis)
 * @return Array of the same shape with values in (0, 1) summing to 1 along
 * axis
 */
Array softmax(const Array &x, int axis = -1);

/**
 * @brief Log-softmax along the specified axis.
 *
 * Computes log_softmax(x) = (x - m) - log(sum(exp(x - m))) along axis,
 * where m = max(x, axis) for numerical stability. Output has the same
 * shape as the input.
 *
 * @param x Input array (integer inputs are promoted to float64)
 * @param axis Axis along which to normalize (default -1, the last axis)
 * @return Array of the same shape with log-softmax values
 */
Array log_softmax(const Array &x, int axis = -1);

} // namespace ins