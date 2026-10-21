// backends/cpu/kernels/cast/common.h
/**
 * @file common.h
 * @brief Common utilities for CPU cast kernels.
 *
 * Provides layout copying and helper functions for casting between types.
 */

#pragma once
#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <complex>
#include <cstring>

// Give the destination the shape of the source with a dense contiguous layout
static inline void copy_layout(InsightArray *dst, const InsightArray *src) {
  dst->ndim = src->ndim;
  dst->numel = src->numel;
  dst->offset = 0;
  for (int i = 0; i < src->ndim; ++i) {
    dst->dims[i] = src->dims[i];
  }
  int64_t stride = 1;
  for (int i = src->ndim - 1; i >= 0; --i) {
    dst->strides[i] = stride;
    stride *= src->dims[i];
  }
}

// Convert a linear index to an element offset using dimension strides
static inline int64_t cpu_offset_from_linear(int64_t linear, int64_t ndim,
                                             const int64_t *dims,
                                             const int64_t *strides) {

  int64_t offset = 0;
  int64_t remaining = linear;

  for (int64_t d = ndim - 1; d >= 0; --d) {
    int64_t idx = remaining % dims[d];
    remaining /= dims[d];
    offset += idx * strides[d];
  }

  return offset;
}

// Helper: copy data with OpenMP parallel for.
// `i` indexes the dense destination, `si` the (possibly strided) source.
#define CAST_LOOP(n, code)                                                     \
  do {                                                                         \
    const int64_t src_ndim = src->ndim;                                        \
    const int64_t src_base = src->offset;                                      \
    _Pragma("omp parallel for") for (int64_t i = 0; i < n; ++i) {              \
      const int64_t si = src_base + cpu_offset_from_linear(                    \
                                        i, src_ndim, src->dims, src->strides); \
      code;                                                                    \
    }                                                                          \
  } while (0)
