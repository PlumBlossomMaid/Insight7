// backends/cpu/kernels/fft/common_impl.cpp
/**
 * @file common_impl.cpp
 * @brief FFTW plan cache implementation.
 *
 * The plan cache is thread-local, but FFTW's planner is global shared state
 * and is not thread-safe, so every plan creation/destruction is serialized
 * on a process-wide mutex. A mutex is used rather than
 * fftw_make_planner_thread_safe() because the latter needs a guaranteed
 * one-time init hook per precision and is only available from FFTW 3.3.5,
 * whereas the lock is self-contained and also covers plan destruction
 * performed during cache eviction.
 */

#ifdef INSIGHT_USE_FFTW3

#include "common.h"
#include <cstring>
#include <mutex>

static thread_local FFTWCache cache = {{0}};

FFTWCache *fft_get_cache(void) { return &cache; }

static std::mutex &fft_planner_mutex(void) {
  static std::mutex m;
  return m;
}

fftw_plan fft_ensure_plan_f64(int n, int64_t batch, int direction,
                              FFTKind kind) {
  std::lock_guard<std::mutex> lock(fft_planner_mutex());
  return (fftw_plan)fft_cache_find_or_create(n, batch, direction, kind, 1);
}

fftwf_plan fft_ensure_plan_f32(int n, int64_t batch, int direction,
                               FFTKind kind) {
  std::lock_guard<std::mutex> lock(fft_planner_mutex());
  return (fftwf_plan)fft_cache_find_or_create(n, batch, direction, kind, 0);
}

#endif // INSIGHT_USE_FFTW3
