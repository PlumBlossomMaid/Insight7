// insight/ops/complex.h
#pragma once
#include "insight/core/array.h"

namespace ins {

/**
 * @brief Check if an array has complex data type.
 *
 * @param x Input array
 * @return true if dtype is C32 or C64
 */
bool is_complex(const Array &x);

/**
 * @brief Check if an array uses legacy complex storage format (last dimension =
 * 2).
 *
 * This is for backward compatibility with the old representation.
 * For modern code, prefer is_complex() which checks the data type.
 *
 * @param x Input array
 * @return true if last dimension exists and equals 2
 */
bool has_complex_shape(const Array &x);

/**
 * @brief Convert real array to complex by adding zero imaginary part.
 *
 * Input shape: [d1, ..., dn]
 * Output shape: [d1, ..., dn, 2]
 *
 * @param real Real part array
 * @return Complex array (real, imag=0)
 */
Array to_complex(const Array &real);

/**
 * @brief Convert two real arrays to complex.
 *
 * Input shapes: both [d1, ..., dn]
 * Output shape: [d1, ..., dn, 2]
 *
 * @param real Real part array
 * @param imag Imaginary part array
 * @return Complex array
 */
Array to_complex(const Array &real, const Array &imag);

/**
 * @brief View real array as complex array (zero-copy).
 *
 * Input must have last dimension = 2 (interleaved real, imag).
 * Input dtype: F32/F64, Output dtype: C32/C64.
 *
 * @param x Real array with shape (..., 2)
 * @return Complex array view with shape (...)
 */
Array as_complex(const Array &x);

/**
 * @brief View complex array as real array (zero-copy).
 *
 * Input dtype: C32/C64, Output dtype: F32/F64.
 * Output shape: input shape + (2,)
 *
 * @param x Complex array
 * @return Real array view with last dimension = 2
 */
Array as_real(const Array &x);

/**
 * @brief Extract real part from complex array (view).
 *
 * Input shape: (..., 2)
 * Output shape: (...)
 *
 * @param z Complex array
 * @return Real part view
 */
Array real(const Array &z);

/**
 * @brief Extract imaginary part from complex array (view).
 *
 * Input shape: (..., 2)
 * Output shape: (...)
 *
 * @param z Complex array
 * @return Imaginary part view
 */
Array imag(const Array &z);

} // namespace ins