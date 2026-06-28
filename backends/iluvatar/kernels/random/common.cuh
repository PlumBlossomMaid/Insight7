// backends/cuda/kernels/random/common.cuh
/**
 * @file common.cuh
 * @brief Common utilities for CUDA random kernels.
 *
 * Provides curand state management and random distribution helpers.
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * @brief Initialize curand states for random number generation.
 *
 * Each thread gets its own curand state, initialized with the given seed
 * and unique sequence number based on thread index.
 *
 * @param states Output curand state array
 * @param seed   Random seed
 * @param n      Number of states to initialize
 */
static __global__ void init_curand_states(curandState *states,
                                          unsigned long long seed, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    curand_init(seed, idx, 0, &states[idx]);
  }
}

/**
 * @brief Get the number of threads per block for random kernels.
 *
 * @return Fixed value of 256 threads per block
 */
inline int random_threads() { return 256; }

/**
 * @brief Calculate the number of blocks needed for a given element count.
 *
 * Uses ceiling division: blocks = (n + threads - 1) / threads
 *
 * @param n Total number of elements to process
 * @return Number of blocks to launch
 */
inline int random_blocks(int64_t n) {
  return static_cast<int>((n + random_threads() - 1) / random_threads());
}

/**
 * @brief Device function for uniform float distribution [0, 1).
 *
 * @param state Pointer to curand state
 * @return Random float in [0, 1)
 */
__device__ inline float uniform_f32(curandState *state) {
  return curand_uniform(state);
}

/**
 * @brief Device function for uniform double distribution [0, 1).
 *
 * @param state Pointer to curand state
 * @return Random double in [0, 1)
 */
__device__ inline double uniform_f64(curandState *state) {
  return curand_uniform_double(state);
}

/**
 * @brief Device function for normal float distribution (0, 1).
 *
 * @param state Pointer to curand state
 * @return Random float from normal(0, 1)
 */
__device__ inline float normal_f32(curandState *state) {
  return curand_normal(state);
}

/**
 * @brief Device function for normal double distribution (0, 1).
 *
 * @param state Pointer to curand state
 * @return Random double from normal(0, 1)
 */
__device__ inline double normal_f64(curandState *state) {
  return curand_normal_double(state);
}

/**
 * @brief Device function for gamma float distribution using Marsaglia-Tsang
 * method.
 *
 * @param state Pointer to curand state
 * @param shape Shape parameter (must be > 0)
 * @return Random float from gamma(shape, 1)
 */
static __device__ float gamma_f32(curandState *state, float shape) {
  float d, c, x, v, u;

  if (shape < 1.0f) {
    // Gamma(shape) = Gamma(1+shape) * Uniform^(1/shape)
    d = 1.0f + shape - 1.0f / 3.0f;
    c = 1.0f / sqrtf(9.0f * d);
    do {
      do {
        x = normal_f32(state);
        v = 1.0f + c * x;
      } while (v <= 0.0f);
      v = v * v * v;
      u = uniform_f32(state);
    } while (u > 1.0f - 0.0331f * (x * x) * (x * x) &&
             logf(u) > 0.5f * x * x + d * (1.0f - v + logf(v)));
    return d * v * powf(uniform_f32(state), 1.0f / shape);
  }

  // shape >= 1
  d = shape - 1.0f / 3.0f;
  c = 1.0f / sqrtf(9.0f * d);
  do {
    do {
      x = normal_f32(state);
      v = 1.0f + c * x;
    } while (v <= 0.0f);
    v = v * v * v;
    u = uniform_f32(state);
  } while (u > 1.0f - 0.0331f * (x * x) * (x * x) &&
           logf(u) > 0.5f * x * x + d * (1.0f - v + logf(v)));
  return d * v;
}

/**
 * @brief Device function for gamma double distribution using Marsaglia-Tsang
 * method.
 *
 * @param state Pointer to curand state
 * @param shape Shape parameter (must be > 0)
 * @return Random double from gamma(shape, 1)
 */
static __device__ double gamma_f64(curandState *state, double shape) {
  double d, c, x, v, u;

  if (shape < 1.0) {
    d = 1.0 + shape - 1.0 / 3.0;
    c = 1.0 / sqrt(9.0 * d);
    do {
      do {
        x = normal_f64(state);
        v = 1.0 + c * x;
      } while (v <= 0.0);
      v = v * v * v;
      u = uniform_f64(state);
    } while (u > 1.0 - 0.0331 * (x * x) * (x * x) &&
             log(u) > 0.5 * x * x + d * (1.0 - v + log(v)));
    return d * v * pow(uniform_f64(state), 1.0 / shape);
  }

  d = shape - 1.0 / 3.0;
  c = 1.0 / sqrt(9.0 * d);
  do {
    do {
      x = normal_f64(state);
      v = 1.0 + c * x;
    } while (v <= 0.0);
    v = v * v * v;
    u = uniform_f64(state);
  } while (u > 1.0 - 0.0331 * (x * x) * (x * x) &&
           log(u) > 0.5 * x * x + d * (1.0 - v + log(v)));
  return d * v;
}
