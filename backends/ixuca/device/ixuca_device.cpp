// backends/ixuca/device/ixuca_device.cpp
// IXUCA device HAL — uses CoreX CUDA-compatible runtime API.
// 天数智芯 IXUCA BI-V150S 通过 CoreX SDK 提供标准 CUDA API 兼容层
// (cudaMalloc, cudaMemcpy, cudaStreamCreate 等均在 /usr/local/corex/include/
// 中实现)

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "insight/c_api/device_ext.h"
#include "insight/c_api/dtype.h"

#include "../registry/ixuca_registry.h"

#ifdef _MSC_VER
#include <stdlib.h>
#define strdup _strdup
#endif

// ========================================================================
// Thread-local error storage
// ========================================================================

static thread_local std::string gpu_last_error_str;

extern "C" void ixuca_set_last_error_ext(const char *msg) {
  gpu_last_error_str = msg ? msg : "";
}

extern "C" const char *ixuca_get_last_error_ext(void) {
  if (!gpu_last_error_str.empty()) {
    return gpu_last_error_str.c_str();
  }
  const char *cuda_err = cudaGetErrorString(cudaGetLastError());
  return cuda_err ? cuda_err : "IXUCA backend: no error";
}

// ========================================================================
// Device Lifecycle
// ========================================================================

static C_Status ix_initialize(void) {
  cudaError_t err = cudaFree(nullptr);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_finalize(void) {
  cudaError_t err = cudaDeviceReset();
  if (err != cudaSuccess && err != cudaErrorNoDevice) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_init_device(C_Device device) {
  if (!device)
    return C_FAILED;
  cudaError_t err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  err = cudaFree(nullptr);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_deinit_device(C_Device device) { return C_SUCCESS; }

// ========================================================================
// Device Management
// ========================================================================

static C_Status ix_set_device(C_Device device) {
  if (!device)
    return C_FAILED;
  cudaError_t err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_get_device(C_Device device) {
  if (!device)
    return C_FAILED;
  int id;
  cudaError_t err = cudaGetDevice(&id);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  device->id = id;
  return C_SUCCESS;
}

static C_Status ix_synchronize_device(C_Device device) {
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_get_device_count(size_t *count) {
  if (!count)
    return C_FAILED;
  int c;
  cudaError_t err = cudaGetDeviceCount(&c);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  *count = static_cast<size_t>(c);
  return C_SUCCESS;
}

static C_Status ix_get_device_list(size_t *devices) {
  if (!devices)
    return C_FAILED;
  size_t count;
  C_Status s = ix_get_device_count(&count);
  if (s != C_SUCCESS)
    return s;
  for (size_t i = 0; i < count; ++i)
    devices[i] = i;
  return C_SUCCESS;
}

// ========================================================================
// Memory Management
// ========================================================================

static C_Status ix_device_memory_allocate(C_Device device, void **ptr,
                                          size_t size) {
  if (!ptr)
    return C_FAILED;
  if (size == 0) {
    *ptr = nullptr;
    return C_SUCCESS;
  }
  cudaSetDevice(device->id);
  cudaFree(nullptr);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaGetLastError();
  cudaError_t err = cudaMalloc(ptr, size);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  ixuca_set_last_error_ext("");
  return C_SUCCESS;
}

static C_Status ix_device_memory_deallocate(C_Device device, void *ptr,
                                            size_t size) {
  if (!ptr)
    return C_SUCCESS;
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_host_memory_allocate(C_Device device, void **ptr,
                                        size_t size) {
  if (!ptr)
    return C_FAILED;
  if (size == 0) {
    *ptr = nullptr;
    return C_SUCCESS;
  }
  cudaError_t err = cudaMallocHost(ptr, size);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_host_memory_deallocate(C_Device device, void *ptr,
                                          size_t size) {
  if (!ptr)
    return C_SUCCESS;
  cudaError_t err = cudaFreeHost(ptr);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_memory_copy_h2d(C_Device device, void *dst, const void *src,
                                   size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_memory_copy_d2h(C_Device device, void *dst, const void *src,
                                   size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_memory_copy_d2d(C_Device device, void *dst, const void *src,
                                   size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_memory_copy_p2p(C_Device dst_device, C_Device src_device,
                                   void *dst, const void *src, size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err =
      cudaMemcpyPeer(dst, dst_device->id, src, src_device->id, size);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_device_memory_set(C_Device device, void *ptr,
                                     unsigned char value, size_t size) {
  if (!ptr || size == 0)
    return C_SUCCESS;
  cudaError_t err = cudaMemset(ptr, value, size);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_device_memory_stats(C_Device device, size_t *total_memory,
                                       size_t *free_memory) {
  if (!total_memory || !free_memory)
    return C_FAILED;
  cudaError_t err = cudaMemGetInfo(free_memory, total_memory);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

// ========================================================================
// Async Memory
// ========================================================================

static C_Status ix_async_memory_copy_h2d(C_Device device, C_Stream stream,
                                         void *dst, const void *src,
                                         size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                                    reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_async_memory_copy_d2h(C_Device device, C_Stream stream,
                                         void *dst, const void *src,
                                         size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                                    reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_async_memory_copy_d2d(C_Device device, C_Stream stream,
                                         void *dst, const void *src,
                                         size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice,
                                    reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_async_memory_copy_p2p(C_Device dst_device,
                                         C_Device src_device, C_Stream stream,
                                         void *dst, const void *src,
                                         size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err =
      cudaMemcpyPeerAsync(dst, dst_device->id, src, src_device->id, size,
                          reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

// ========================================================================
// Stream Management
// ========================================================================

static C_Status ix_create_stream(C_Device device, C_Stream *stream) {
  if (!stream)
    return C_FAILED;
  cudaStream_t s;
  cudaError_t err = cudaStreamCreate(&s);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  *stream = reinterpret_cast<C_Stream>(s);
  return C_SUCCESS;
}

static C_Status ix_destroy_stream(C_Device device, C_Stream stream) {
  cudaError_t err = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_query_stream(C_Device device, C_Stream stream) {
  cudaError_t err = cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream));
  if (err == cudaSuccess)
    return C_SUCCESS;
  if (err == cudaErrorNotReady)
    return C_WARNING;
  ixuca_set_last_error_ext(cudaGetErrorString(err));
  return C_FAILED;
}

static C_Status ix_synchronize_stream(C_Device device, C_Stream stream) {
  cudaError_t err =
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_stream_add_callback(C_Device device, C_Stream stream,
                                       void (*callback)(C_Device, C_Stream,
                                                        void *, C_Status *),
                                       void *user_data) {
  cudaError_t err = cudaLaunchHostFunc(
      reinterpret_cast<cudaStream_t>(stream), [](void *data) {}, user_data);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_stream_wait_event(C_Device device, C_Stream stream,
                                     C_Event event) {
  cudaError_t err =
      cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream),
                          reinterpret_cast<cudaEvent_t>(event), 0);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

// ========================================================================
// Event Management
// ========================================================================

static C_Status ix_create_event(C_Device device, C_Event *event) {
  if (!event)
    return C_FAILED;
  cudaEvent_t e;
  cudaError_t err = cudaEventCreate(&e);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  *event = reinterpret_cast<C_Event>(e);
  return C_SUCCESS;
}

static C_Status ix_destroy_event(C_Device device, C_Event event) {
  cudaError_t err = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_record_event(C_Device device, C_Stream stream,
                                C_Event event) {
  cudaError_t err = cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                                    reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_query_event(C_Device device, C_Event event) {
  cudaError_t err = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
  if (err == cudaSuccess)
    return C_SUCCESS;
  if (err == cudaErrorNotReady)
    return C_WARNING;
  ixuca_set_last_error_ext(cudaGetErrorString(err));
  return C_FAILED;
}

static C_Status ix_synchronize_event(C_Device device, C_Event event) {
  cudaError_t err = cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event));
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status ix_elapsed_time(C_Event start, C_Event end, float *ms) {
  if (!ms)
    return C_FAILED;
  cudaError_t err =
      cudaEventElapsedTime(ms, reinterpret_cast<cudaEvent_t>(start),
                           reinterpret_cast<cudaEvent_t>(end));
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

// ========================================================================
// Device Information
// ========================================================================

static C_Status ix_get_compute_capability(C_Device device, size_t *capability) {
  if (!capability || !device)
    return C_FAILED;
  int major, minor;
  cudaError_t err = cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, device->id);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device->id);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  *capability = static_cast<size_t>(major * 10 + minor);
  return C_SUCCESS;
}

static C_Status ix_get_runtime_version(C_Device device, size_t *version) {
  if (!version)
    return C_FAILED;
  int v;
  cudaError_t err = cudaRuntimeGetVersion(&v);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  *version = static_cast<size_t>(v);
  return C_SUCCESS;
}

static C_Status ix_get_driver_version(C_Device device, size_t *version) {
  if (!version)
    return C_FAILED;
  int v;
  cudaError_t err = cudaDriverGetVersion(&v);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  *version = static_cast<size_t>(v);
  return C_SUCCESS;
}

static C_Status ix_get_multi_process(C_Device device, size_t *multi_process) {
  if (!multi_process || !device)
    return C_FAILED;
  int mp;
  cudaError_t err =
      cudaDeviceGetAttribute(&mp, cudaDevAttrMultiProcessorCount, device->id);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  *multi_process = static_cast<size_t>(mp);
  return C_SUCCESS;
}

static C_Status ix_get_max_threads_per_mp(C_Device device, size_t *threads) {
  if (!threads || !device)
    return C_FAILED;
  int t;
  cudaError_t err = cudaDeviceGetAttribute(
      &t, cudaDevAttrMaxThreadsPerMultiProcessor, device->id);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  *threads = static_cast<size_t>(t);
  return C_SUCCESS;
}

static C_Status ix_get_max_threads_per_block(C_Device device, size_t *threads) {
  if (!threads || !device)
    return C_FAILED;
  int t;
  cudaError_t err =
      cudaDeviceGetAttribute(&t, cudaDevAttrMaxThreadsPerBlock, device->id);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  *threads = static_cast<size_t>(t);
  return C_SUCCESS;
}

static C_Status ix_get_max_grid_dim_size(C_Device device, size_t dims[3]) {
  if (!dims || !device)
    return C_FAILED;
  int x, y, z;
  cudaError_t err;
  err = cudaDeviceGetAttribute(&x, cudaDevAttrMaxGridDimX, device->id);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  err = cudaDeviceGetAttribute(&y, cudaDevAttrMaxGridDimY, device->id);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  err = cudaDeviceGetAttribute(&z, cudaDevAttrMaxGridDimZ, device->id);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  dims[0] = static_cast<size_t>(x);
  dims[1] = static_cast<size_t>(y);
  dims[2] = static_cast<size_t>(z);
  return C_SUCCESS;
}

static C_Status ix_get_device_name(C_Device device, char *buf,
                                   size_t buf_size) {
  if (!buf || buf_size == 0 || !device)
    return C_FAILED;
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device->id);
  if (err != cudaSuccess) {
    ixuca_set_last_error_ext(cudaGetErrorString(err));
    return C_FAILED;
  }
  std::strncpy(buf, prop.name, buf_size - 1);
  buf[buf_size - 1] = '\0';
  return C_SUCCESS;
}

// ========================================================================
// Profiler (std::chrono-based, same as CUDA backend)
// ========================================================================

struct IXUCAProfilerEvent {
  std::string name;
  size_t calls = 0;
  double total_ms = 0.0;
  double min_ms = 1e18;
  double max_ms = 0.0;
};

struct IXUCAProfiler {
  std::string name;
  int device_id;
  std::chrono::high_resolution_clock::time_point event_start;
  std::string current_event_name;
  bool in_event = false;
  bool running = false;
  std::unordered_map<std::string, IXUCAProfilerEvent> events;
  std::vector<C_ProfilerEvent> report_cache;
};

static C_Status ix_profiler_create(C_Profiler *prof, const char *name,
                                   int device_id) {
  if (!prof)
    return C_FAILED;
  auto *p = new (std::nothrow) IXUCAProfiler();
  if (!p)
    return C_FAILED;
  if (name)
    p->name = name;
  p->device_id = device_id;
  *prof = reinterpret_cast<C_Profiler>(p);
  return C_SUCCESS;
}

static C_Status ix_profiler_destroy(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  delete reinterpret_cast<IXUCAProfiler *>(prof);
  return C_SUCCESS;
}

static C_Status ix_profiler_start(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  reinterpret_cast<IXUCAProfiler *>(prof)->running = true;
  return C_SUCCESS;
}

static C_Status ix_profiler_stop(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  reinterpret_cast<IXUCAProfiler *>(prof)->running = false;
  return C_SUCCESS;
}

static C_Status ix_profiler_reset(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  auto *p = reinterpret_cast<IXUCAProfiler *>(prof);
  p->events.clear();
  p->report_cache.clear();
  p->in_event = false;
  return C_SUCCESS;
}

static C_Status ix_profiler_begin_event(C_Profiler prof, const char *name) {
  if (!prof || !name)
    return C_FAILED;
  auto *p = reinterpret_cast<IXUCAProfiler *>(prof);
  if (!p->running)
    return C_SUCCESS;
  p->current_event_name = name;
  p->event_start = std::chrono::high_resolution_clock::now();
  p->in_event = true;
  return C_SUCCESS;
}

static C_Status ix_profiler_end_event(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  auto *p = reinterpret_cast<IXUCAProfiler *>(prof);
  if (!p->running || !p->in_event)
    return C_SUCCESS;
  auto end = std::chrono::high_resolution_clock::now();
  double ms =
      std::chrono::duration<double, std::milli>(end - p->event_start).count();
  p->in_event = false;
  auto &ev = p->events[p->current_event_name];
  ev.name = p->current_event_name;
  ev.calls++;
  ev.total_ms += ms;
  if (ms < ev.min_ms)
    ev.min_ms = ms;
  if (ms > ev.max_ms)
    ev.max_ms = ms;
  return C_SUCCESS;
}

static C_Status ix_profiler_get_events(C_Profiler prof,
                                       C_ProfilerEvent **events,
                                       size_t *count) {
  if (!prof || !events || !count)
    return C_FAILED;
  auto *p = reinterpret_cast<IXUCAProfiler *>(prof);
  for (auto &e : p->report_cache) {
    if (e.name) {
      free(const_cast<char *>(e.name));
      e.name = nullptr;
    }
  }
  p->report_cache.clear();
  for (auto &kv : p->events) {
    auto &src = kv.second;
    C_ProfilerEvent ev;
    ev.name = strdup(src.name.c_str());
    ev.calls = src.calls;
    ev.total_ms = static_cast<float>(src.total_ms);
    ev.min_ms = static_cast<float>(src.min_ms < 1e17 ? src.min_ms : 0.0f);
    ev.max_ms = static_cast<float>(src.max_ms);
    p->report_cache.push_back(ev);
  }
  *events = p->report_cache.data();
  *count = p->report_cache.size();
  return C_SUCCESS;
}

// ========================================================================
// InitPluginGPU — IXUCA Backend Entry Point
// ========================================================================

extern "C" {
INSIGHT_GPU_API C_Status InitPluginGPU(CustomRuntimeParams *params) {
  if (!params || !params->interface)
    return C_FAILED;
  INSIGHT_CHECK_CUSTOM_DEVICE_VERSION(params);

  C_DeviceInterface *iface = params->interface;
  iface->size = sizeof(C_DeviceInterface);

  iface->initialize = ix_initialize;
  iface->finalize = ix_finalize;
  iface->init_device = ix_init_device;
  iface->deinit_device = ix_deinit_device;

  iface->set_device = ix_set_device;
  iface->get_device = ix_get_device;
  iface->synchronize_device = ix_synchronize_device;
  iface->get_device_count = ix_get_device_count;
  iface->get_device_list = ix_get_device_list;

  iface->device_memory_allocate = ix_device_memory_allocate;
  iface->device_memory_deallocate = ix_device_memory_deallocate;
  iface->host_memory_allocate = ix_host_memory_allocate;
  iface->host_memory_deallocate = ix_host_memory_deallocate;
  iface->memory_copy_h2d = ix_memory_copy_h2d;
  iface->memory_copy_d2h = ix_memory_copy_d2h;
  iface->memory_copy_d2d = ix_memory_copy_d2d;
  iface->memory_copy_p2p = ix_memory_copy_p2p;
  iface->device_memory_set = ix_device_memory_set;
  iface->device_memory_stats = ix_device_memory_stats;

  iface->async_memory_copy_h2d = ix_async_memory_copy_h2d;
  iface->async_memory_copy_d2h = ix_async_memory_copy_d2h;
  iface->async_memory_copy_d2d = ix_async_memory_copy_d2d;
  iface->async_memory_copy_p2p = ix_async_memory_copy_p2p;

  iface->create_stream = ix_create_stream;
  iface->destroy_stream = ix_destroy_stream;
  iface->query_stream = ix_query_stream;
  iface->synchronize_stream = ix_synchronize_stream;
  iface->stream_add_callback = ix_stream_add_callback;
  iface->stream_wait_event = ix_stream_wait_event;

  iface->create_event = ix_create_event;
  iface->destroy_event = ix_destroy_event;
  iface->record_event = ix_record_event;
  iface->query_event = ix_query_event;
  iface->synchronize_event = ix_synchronize_event;
  iface->elapsed_time = ix_elapsed_time;

  iface->get_compute_capability = ix_get_compute_capability;
  iface->get_runtime_version = ix_get_runtime_version;
  iface->get_driver_version = ix_get_driver_version;
  iface->get_multi_process = ix_get_multi_process;
  iface->get_max_threads_per_mp = ix_get_max_threads_per_mp;
  iface->get_max_threads_per_block = ix_get_max_threads_per_block;
  iface->get_max_grid_dim_size = ix_get_max_grid_dim_size;
  iface->get_device_name = ix_get_device_name;

  iface->profiler_create = ix_profiler_create;
  iface->profiler_destroy = ix_profiler_destroy;
  iface->profiler_start = ix_profiler_start;
  iface->profiler_stop = ix_profiler_stop;
  iface->profiler_reset = ix_profiler_reset;
  iface->profiler_begin_event = ix_profiler_begin_event;
  iface->profiler_end_event = ix_profiler_end_event;
  iface->profiler_get_events = ix_profiler_get_events;

  // Error
  iface->get_last_error = ixuca_get_last_error_ext;

  // Reserved
  for (int i = 0; i < 8; ++i) {
    iface->reserved[i] = nullptr;
  }

  // Device identity
  params->device_type = const_cast<char *>("ixuca");
  params->sub_device_type = const_cast<char *>("v1.0");

  if (params->register_kernel) {
    ixuca_sync_kernels(params->register_kernel);
  }

  return C_SUCCESS;
}
}
