/**
 * @file common.cuh
 * @brief Common utilities for CUDA FFT kernels.
 *
 * Provides multi-plan cuFFT caching and error handling for FFT operations.
 * Uses a thread-local hash map keyed by (n, batch, kind, is_f32) so that
 * alternating between different FFT sizes (e.g. range-FFT and Doppler-FFT
 * in radar processing) reuses cached plans instead of creating new ones
 * every time (cuFFT plan creation ~5ms overhead).
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string>
#include <unordered_map>

/**
 * @brief Type of FFT transform.
 */
enum class CuFFTKind {
  C2C = 0, ///< Complex to complex
  R2C = 1, ///< Real to complex
  C2R = 2  ///< Complex to real
};

/**
 * @brief Key for cuFFT plan cache.
 *
 * Note: direction (CUFFT_FORWARD/CUFFT_INVERSE) is NOT part of the key
 * because C2C plans work for both directions (direction passed at exec time),
 * and R2C/C2R plans have implicit direction in their kind.
 */
struct CuFFTKey {
  int64_t n;
  int64_t batch;
  CuFFTKind kind;
  bool is_f32;

  bool operator==(const CuFFTKey &o) const {
    return n == o.n && batch == o.batch && kind == o.kind && is_f32 == o.is_f32;
  }
};

/**
 * @brief Hash functor for CuFFTKey.
 */
struct CuFFTKeyHash {
  size_t operator()(const CuFFTKey &k) const {
    return std::hash<int64_t>()(k.n) ^ (std::hash<int64_t>()(k.batch) << 1) ^
           (std::hash<int>()(static_cast<int>(k.kind)) << 2) ^
           (std::hash<bool>()(k.is_f32) << 3);
  }
};

/**
 * @brief Thread-local multi-plan cuFFT cache.
 *
 * Stores up to MAX_PLANS plans keyed by (n, batch, kind, is_f32).
 * Evicts the oldest entry when full.
 */
struct CuFFTPlanCache {
  static constexpr size_t MAX_PLANS = 16;
  std::unordered_map<CuFFTKey, cufftHandle, CuFFTKeyHash> plans;

  ~CuFFTPlanCache() {
    for (auto &kv : plans) {
      cufftDestroy(kv.second);
    }
  }

  /**
   * @brief Get or create a cuFFT plan.
   *
   * @return cufftHandle (0 on error)
   */
  cufftHandle get_or_create(int64_t n, int64_t batch, CuFFTKind kind,
                            bool is_f32) {
    CuFFTKey key{n, batch, kind, is_f32};
    auto it = plans.find(key);
    if (it != plans.end())
      return it->second;

    // Evict oldest if cache is full
    if (plans.size() >= MAX_PLANS) {
      auto first = plans.begin();
      cufftDestroy(first->second);
      plans.erase(first);
    }

    cufftHandle plan = create_plan_impl(n, batch, kind, is_f32);
    if (plan != 0) {
      plans[key] = plan;
    }
    return plan;
  }

private:
  cufftHandle create_plan_impl(int64_t n, int64_t batch, CuFFTKind kind,
                               bool is_f32) {
    cufftHandle plan;
    cufftResult result;

    if (batch == 1) {
      if (kind == CuFFTKind::C2C) {
        result = cufftPlan1d(&plan, static_cast<int>(n),
                             is_f32 ? CUFFT_C2C : CUFFT_Z2Z, 1);
      } else if (kind == CuFFTKind::R2C) {
        result = cufftPlan1d(&plan, static_cast<int>(n),
                             is_f32 ? CUFFT_R2C : CUFFT_D2Z, 1);
      } else { // C2R
        result = cufftPlan1d(&plan, static_cast<int>(n),
                             is_f32 ? CUFFT_C2R : CUFFT_Z2D, 1);
      }
    } else {
      int n_int = static_cast<int>(n);
      int batch_int = static_cast<int>(batch);

      if (kind == CuFFTKind::C2C) {
        result =
            cufftPlanMany(&plan, 1, &n_int, nullptr, 1, n_int, nullptr, 1,
                          n_int, is_f32 ? CUFFT_C2C : CUFFT_Z2Z, batch_int);
      } else if (kind == CuFFTKind::R2C) {
        int out_dist = static_cast<int>(n / 2 + 1);
        result =
            cufftPlanMany(&plan, 1, &n_int, nullptr, 1, n_int, nullptr, 1,
                          out_dist, is_f32 ? CUFFT_R2C : CUFFT_D2Z, batch_int);
      } else { // C2R
        int in_dist = static_cast<int>(n / 2 + 1);
        result =
            cufftPlanMany(&plan, 1, &n_int, nullptr, 1, in_dist, nullptr, 1,
                          n_int, is_f32 ? CUFFT_C2R : CUFFT_Z2D, batch_int);
      }
    }

    if (result != CUFFT_SUCCESS) {
      return 0;
    }
    return plan;
  }
};

/**
 * @brief Get thread-local multi-plan cuFFT cache.
 *
 * @return Reference to thread-local cache
 */
inline CuFFTPlanCache &cufft_get_cache() {
  thread_local CuFFTPlanCache cache;
  return cache;
}

/**
 * @brief Ensure cuFFT plan exists for given parameters.
 *
 * Creates or reuses a cached cuFFT plan from the multi-plan cache.
 * The direction parameter is accepted for API compatibility but is NOT
 * used in the cache key: C2C plans work for both directions, and
 * R2C/C2R direction is implicit in the transform kind.
 *
 * @param n         FFT length
 * @param batch     Number of transforms
 * @param direction CUFFT_FORWARD or CUFFT_INVERSE (ignored for cache key)
 * @param kind      Transform kind (C2C, R2C, C2R)
 * @param is_f32    True for single precision, false for double precision
 * @return cufftHandle Valid plan handle (0 on error)
 */
inline cufftHandle cufft_ensure_plan(int64_t n, int64_t batch, int direction,
                                     CuFFTKind kind, bool is_f32) {
  return cufft_get_cache().get_or_create(n, batch, kind, is_f32);
}

/**
 * @brief Get cuFFT error string.
 *
 * @param error cuFFT error code
 * @return Human-readable error string
 */
inline const char *cufft_get_error_string(cufftResult error) {
  switch (error) {
  case CUFFT_SUCCESS:
    return "cuFFT success";
  case CUFFT_INVALID_PLAN:
    return "cuFFT invalid plan";
  case CUFFT_ALLOC_FAILED:
    return "cuFFT allocation failed";
  case CUFFT_INVALID_TYPE:
    return "cuFFT invalid type";
  case CUFFT_INVALID_VALUE:
    return "cuFFT invalid value";
  case CUFFT_INTERNAL_ERROR:
    return "cuFFT internal error";
  case CUFFT_EXEC_FAILED:
    return "cuFFT execution failed";
  case CUFFT_SETUP_FAILED:
    return "cuFFT setup failed";
  case CUFFT_INVALID_SIZE:
    return "cuFFT invalid size";
  case CUFFT_UNALIGNED_DATA:
    return "cuFFT unaligned data";
  default:
    return "cuFFT unknown error";
  }
}

/**
 * @brief Set GPU error message from cuFFT result.
 *
 * @param op     Operation name
 * @param result cuFFT error code
 */
inline void cufft_set_error(const char *op, cufftResult result) {
  thread_local char err_buf[256];
  snprintf(err_buf, sizeof(err_buf), "%s: %s", op,
           cufft_get_error_string(result));
  gpu_set_last_error(err_buf);
}

/**
 * @brief Set GPU error message from string.
 *
 * @param msg Error message
 */
inline void cufft_set_error(const char *msg) { gpu_set_last_error(msg); }
