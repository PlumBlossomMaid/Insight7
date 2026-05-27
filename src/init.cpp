// src/init.cpp
#include "insight/init.h"
#include "insight/c_api/device_ext.h"
#include "insight/c_api/kernel.h"
#include "insight/core/exception.h"
#include "insight/core/place.h"
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef interface
#undef interface
#endif

namespace ins {

static bool g_initialized = false;

// ========================================================================
// Dynamic library loading helpers
// ========================================================================

#ifdef _WIN32
using LibHandle = HMODULE;

static LibHandle load_library(const char *path) {
  LibHandle handle = LoadLibraryA(path);
  if (!handle) {
    DWORD error = GetLastError();
    char *error_msg = nullptr;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                       FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                   (LPSTR)&error_msg, 0, NULL);
    fprintf(stderr, "[insight] Failed to load library '%s': %s (code %lu)\n",
            path, error_msg ? error_msg : "Unknown error", error);
    if (error_msg)
      LocalFree(error_msg);
  }
  return handle;
}

static void *get_symbol(LibHandle lib, const char *name) {
  void *sym = reinterpret_cast<void *>(GetProcAddress(lib, name));
  if (!sym) {
    DWORD error = GetLastError();
    fprintf(stderr, "[insight] Failed to get symbol '%s': error code %lu\n",
            name, error);
  }
  return sym;
}

static void unload_library(LibHandle lib) { FreeLibrary(lib); }

static const char *backend_extension() { return ".dll"; }

static std::string backend_libname(const char *base_name) {
  // Windows DLLs have no 'lib' prefix
  return std::string(base_name) + backend_extension();
}
#else
using LibHandle = void *;

static LibHandle load_library(const char *path) {
  LibHandle handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    fprintf(stderr, "[insight] Failed to load library '%s': %s\n", path,
            dlerror());
  }
  return handle;
}

static void *get_symbol(LibHandle lib, const char *name) {
  void *sym = dlsym(lib, name);
  if (!sym) {
    fprintf(stderr, "[insight] Failed to get symbol '%s': %s\n", name,
            dlerror());
  }
  return sym;
}

static void unload_library(LibHandle lib) { dlclose(lib); }

static const char *backend_extension() { return ".so"; }

static std::string backend_libname(const char *base_name) {
  // Linux/macOS shared libs use 'lib' prefix
  return std::string("lib") + base_name + backend_extension();
}
#endif

/**
 * @brief Load a backend plugin and initialize it.
 *
 * Loads the shared library, calls InitPlugin, and stores the
 * device interface in the global registry.
 *
 * @param kind       Device kind to associate
 * @param lib_name   Library name without extension (e.g.,
 * "insight_cpu_backend")
 */
static void load_backend_plugin(DeviceKind kind, const char *lib_name) {
  // Build full path with platform-specific naming
  std::string lib_path = backend_libname(lib_name);

  LibHandle lib = load_library(lib_path.c_str());

  INS_CHECK(lib, std::string(lib_name) + "not available");

  // Get InitPlugin entry point
  typedef C_Status (*InitPluginFunc)(CustomRuntimeParams *);
  InitPluginFunc init_plugin = reinterpret_cast<InitPluginFunc>(get_symbol(
      lib, kind == DeviceKind::GPU ? "InitPluginGPU" : "InitPluginCPU"));
  if (!init_plugin) {
    unload_library(lib);
    INS_THROW("get_symbol function failed: fail to get function " +
              std::string(kind == DeviceKind::GPU ? "InitPluginGPU"
                                                  : "InitPluginCPU"));
  }

  // Prepare params and call InitPlugin
  CustomRuntimeParams params;
  std::memset(&params, 0, sizeof(params));
  params.size = sizeof(CustomRuntimeParams);
  params.interface = new C_DeviceInterface();
  params.interface->size = sizeof(C_DeviceInterface);
  params.register_kernel = insight_register_kernel;

  C_Status status = init_plugin(&params);
  if (status != C_SUCCESS) {
    delete params.interface;
    unload_library(lib);
    INS_THROW("fail to init plugin.");
  }

  // Store device interface in global registry
  set_device_interface(kind, params.interface, params.device_type);
}

/**
 * @brief Initialize a backend by calling its InitPlugin directly.
 *
 * Used for statically linked backends where InitPlugin is linked
 * into the same executable.
 *
 * @param kind        Device kind to associate
 * @param init_plugin Pointer to the backend's InitPlugin function
 * @return true on success, false on failure
 */
static bool
init_static_backend(DeviceKind kind,
                    C_Status (*init_plugin)(CustomRuntimeParams *)) {
  CustomRuntimeParams params;
  std::memset(&params, 0, sizeof(params));
  params.size = sizeof(CustomRuntimeParams);
  params.interface = new C_DeviceInterface();
  params.interface->size = sizeof(C_DeviceInterface);
  params.register_kernel = insight_register_kernel;

  C_Status status = init_plugin(&params);
  if (status != C_SUCCESS) {
    delete params.interface;
    return false;
  }

  set_device_interface(kind, params.interface, params.device_type);
  return true;
}

// ========================================================================
// Public API
// ========================================================================

void init(const std::vector<std::string> &backends) {
  if (g_initialized)
    return;

  for (const auto &backend : backends) {
    if (backend == "cpu") {
      if (!get_device_interface(DeviceKind::CPU)) {
#ifdef INSIGHT_CPU_BACKEND_EXPORTS
        load_backend_plugin(DeviceKind::CPU, "insight_cpu_backend");
#else
        init_static_backend(DeviceKind::CPU, InitPluginCPU);
#endif
      }
    } else {
      // GPU 后端：必须动态加载
      if (!get_device_interface(DeviceKind::GPU)) {
        std::string lib_name = "insight_" + backend + "_backend";
        load_backend_plugin(DeviceKind::GPU, lib_name.c_str());
      } else {
        // 已经有 GPU 后端了，说明重复加载
        INS_THROW("GPU backend already loaded, cannot load another: ", backend);
      }
    }
  }

  // 检查所有要求的后端是否都成功加载
  for (const auto &backend : backends) {
    if (backend == "cpu" && !get_device_interface(DeviceKind::CPU)) {
      INS_THROW("Failed to load CPU backend");
    } else if (backend != "cpu" && !get_device_interface(DeviceKind::GPU)) {
      INS_THROW("Failed to load GPU backend: ", backend);
    }
  }

  g_initialized = true;
}

bool is_initialized() { return g_initialized; }

bool has_device(DeviceKind kind) {
  return get_device_interface(kind) != nullptr;
}

} // namespace ins