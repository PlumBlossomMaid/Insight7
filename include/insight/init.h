// include/insight/init.h
#pragma once
#include "insight/core/place.h"
#include <optional>
#include <string>
#include <vector>

namespace ins {

struct InitOptions {
  std::optional<std::vector<std::string>> backends = std::nullopt;
  std::string gpu_backend;
};

/**
 * @brief Initialize Insight framework with smart backend discovery.
 *
 * - No args: CPU required, then auto-discover and load the first available GPU backend.
 * - InitOptions::gpu_backend forces a concrete GPU backend during discovery.
 * - INSIGHT_GPU_BACKEND=<name>: prefer a concrete GPU backend during discovery.
 * - Empty vector: load nothing.
 * - Specified backends: "cpu" loads CPU, "gpu" loads the selected GPU backend,
 *   and concrete names such as "cuda", "rocm", "ixuca", or "sdaa" force a backend.
 */
void init(std::optional<std::vector<std::string>> backends = std::nullopt);

/**
 * @brief Initialize with explicit options.
 */
void init(const InitOptions &options);

/**
 * @brief Initialize with specified backends (convenience overload).
 * Same as init(std::optional<...>(backends)).
 */
void init(const std::vector<std::string> &backends);

/**
 * @brief Check if Insight is initialized.
 */
bool is_initialized();

/**
 * @brief Check if a device type is available.
 */
bool has_device(DeviceKind kind);

/**
 * @brief Return the active GPU backend name.
 *
 * The public device model stays CPU/GPU. This function reports which concrete
 * GPU implementation is registered behind DeviceKind::GPU, e.g. "cuda",
 * "rocm", "ixuca", "sdaa", or "" when no GPU backend is active.
 *
 * @return Active GPU backend name, or empty string if no GPU backend is loaded.
 */
std::string active_gpu_backend_name();

/**
 * @brief Return the active GPU backend version/subtype string.
 *
 * @return Backend version/subtype string, or empty string if unavailable.
 */
std::string active_gpu_backend_version();

/**
 * @brief Load an additional backend after init().
 * Safe to call multiple times (no-op if already loaded).
 * @param backend "cpu", "gpu", or a concrete GPU backend name.
 */
void load_backend(const std::string &backend);

/**
 * @brief Add a directory to the backend search path.
 * Called before init() so backends can be found in non-standard locations
 * (e.g. Python package directory for pip-installed libraries).
 */
void add_backend_search_path(const std::string &path);

} // namespace ins
