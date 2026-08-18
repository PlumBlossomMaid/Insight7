// bindings/lua/insight_lua.cpp
//
// Lua/LuaJIT bindings for the Insight7 scientific computing framework.
// Built with sol2 (header-only C++ <-> Lua binding library).
// API style: PaddlePaddle (ins.float32, ins.CPUPlace(), etc.)
//
// Usage:
//   local ins = require("insight")
//   local a = ins.zeros({2, 3}, ins.float32)
//   local b = ins.ones({2, 3}, ins.float32)
//   local c = a + b
//   print(c)

#define SOL_ALL_SAFETIES_ON 1
#define SOL_NO_EXCEPTIONS 0
#include <sol/sol.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "insight/c_api/array.h"
#include "insight/c_api/profiler.h"
#include "insight/core/array.h"
#include "insight/core/dtype.h"
#include "insight/core/place.h"
#include "insight/core/shape.h"
#include "insight/core/slice.h"
#include "insight/init.h"
#include "insight/ops/cast.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/linalg.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/operator.h"
#include "insight/ops/random.h"
#include "insight/ops/reduction.h"
#include "insight/ops/signal.h"
#include "insight/ops/unary.h"
#ifdef INSIGHT_USE_MATPLOT
#include "insight/ops/plot.h"
#endif

#include <cmath>
#include <csignal>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Helper: convert Lua table {1,2,3} to std::vector<int64_t>
static std::vector<int64_t> table_to_shape(sol::table t) {
  std::vector<int64_t> v;
  for (size_t i = 1; i <= t.size(); i++) {
    v.push_back(t.get<int64_t>(i));
  }
  return v;
}

// Helper: convert Lua table of Arrays to std::vector<ins::Array>
static std::vector<ins::Array> table_to_arrays(sol::table t) {
  std::vector<ins::Array> v;
  for (size_t i = 1; i <= t.size(); i++) {
    v.push_back(t.get<ins::Array>(i));
  }
  return v;
}

// ── Strict Lua table → ins::Array conversion ──────────────────────────────

// Validate a Lua table element: must be number, boolean, or table.
// nil, string, function, userdata, thread → error.
static void validate_element(sol::object obj, const std::string &path) {
  auto t = obj.get_type();
  if (t == sol::type::lua_nil) {
    throw std::runtime_error("nil found at " + path +
                             ": nil values are not allowed");
  }
  if (t == sol::type::string) {
    throw std::runtime_error("string found at " + path +
                             ": only numbers and booleans are allowed");
  }
  if (t == sol::type::function) {
    throw std::runtime_error("function found at " + path +
                             ": only numbers and booleans are allowed");
  }
  if (t == sol::type::userdata || t == sol::type::lightuserdata) {
    throw std::runtime_error("userdata found at " + path +
                             ": only numbers and booleans are allowed");
  }
  if (t == sol::type::thread) {
    throw std::runtime_error("thread found at " + path +
                             ": only numbers and booleans are allowed");
  }
}

// Recursively infer shape from a nested Lua table.
// Throws on ragged nesting or illegal element types.
static void infer_lua_shape(sol::table t, std::vector<int64_t> &shape,
                            int depth, const std::string &path) {
  if (depth >= INSIGHT_MAX_NDIM) {
    throw std::runtime_error(
        "Table nesting exceeds ins::maximum supported dimensions (" +
        std::to_string(INSIGHT_MAX_NDIM) + ") at " + path);
  }
  int64_t len = static_cast<int64_t>(t.size());
  int64_t entry_count = 0;
  int64_t max_index = 0;
  bool has_invalid_key = false;
  t.for_each([&](sol::object key, sol::object) {
    if (!key.is<int64_t>()) {
      has_invalid_key = true;
      return;
    }
    int64_t index = key.as<int64_t>();
    if (index < 1) {
      has_invalid_key = true;
      return;
    }
    ++entry_count;
    max_index = std::max(max_index, index);
  });
  if (has_invalid_key || entry_count != len || max_index != len) {
    throw std::runtime_error("Lua table must contain contiguous array indices at " +
                             path);
  }
  if (depth >= static_cast<int>(shape.size())) {
    shape.push_back(len);
  } else if (shape[depth] != len) {
    throw std::runtime_error(
        "Ragged nested table at " + path + ": dimension " +
        std::to_string(depth) + " has inconsistent sizes (" +
        std::to_string(shape[depth]) + " vs " + std::to_string(len) + ")");
  }
  for (int64_t i = 1; i <= len; i++) {
    sol::object item = t[i];
    std::string child_path = path + "[" + std::to_string(i) + "]";
    validate_element(item, child_path);
    if (item.get_type() == sol::type::table) {
      infer_lua_shape(item.as<sol::table>(), shape, depth + 1, child_path);
    }
  }
}

// Flatten a validated nested Lua table into doubles.
static void flatten_lua_table(sol::table t, std::vector<double> &out) {
  for (size_t i = 1; i <= t.size(); i++) {
    sol::object item = t[i];
    if (item.get_type() == sol::type::table) {
      flatten_lua_table(item.as<sol::table>(), out);
    } else if (item.is<bool>()) {
      out.push_back(item.as<bool>() ? 1.0 : 0.0);
    } else {
      out.push_back(item.as<double>());
    }
  }
}

// Public API: convert a Lua table (nested or flat) to an ins::Array.
// Strict validation: only number/bool/table allowed. nil/string/etc → error.
static ins::Array from_lua_table(sol::table t,
                                 sol::optional<ins::Place> place) {
  ins::Place p = place.value_or(ins::get_device());
  if (t.size() == 0) {
    return ins::Array(ins::Shape({0}), ins::DType::F64, p);
  }
  std::vector<int64_t> shape;
  std::string path = "root";
  infer_lua_shape(t, shape, 0, path);
  std::vector<double> data;
  flatten_lua_table(t, data);
  int64_t expected = 1;
  for (auto d : shape)
    expected *= d;
  if (static_cast<int64_t>(data.size()) != expected) {
    throw std::runtime_error("ins::Shape/data size mismatch");
  }
  ins::Array result(ins::Shape(shape), ins::DType::F64, ins::CPUPlace());
  std::memcpy(result.data(), data.data(), data.size() * sizeof(double));
  if (p.kind() != ins::DeviceKind::CPU) {
    result = result.to(p);
  }
  return result;
}

// Helper: ins::Array __tostring — delegates to C++ ins::to_string()
static std::string array_tostring(const ins::Array &a) {
  if (!a.defined())
    return "<insight.Array (undefined)>";
  char *s = insight_array_tostring(a.layout_ptr());
  if (s) {
    std::string result(s);
    std::free(s);
    return result;
  }
  return "<insight.Array (error)>";
}

// Helper: ins::signal::write_bin wrapper (avoids sol2 default-param issues)
static void lua_write_bin(const std::string &file, const ins::Array &data,
                          sol::optional<bool> append) {
  ins::signal::write_bin(file, data, append.value_or(true));
}

// Helper: ins::signal::read_bin wrapper
static ins::Array lua_read_bin(const std::string &file,
                               sol::optional<int> dtype) {
  ins::DType dt = dtype.has_value() ? static_cast<ins::DType>(dtype.value())
                                    : ins::DType::U8;
  return ins::signal::read_bin(file, dt);
}

// Helper: ins::signal::unpack_bin wrapper (avoids sol2 default-param issues)
static ins::Array lua_unpack_bin(const ins::Array &binary, int dtype,
                                 sol::optional<std::string> endianness) {
  return ins::signal::unpack_bin(binary, static_cast<ins::DType>(dtype),
                                 endianness.value_or("L"));
}

// Helper: ins::signal::write_sigmf wrapper (avoids sol2 default-param issues)
static void lua_write_sigmf(const std::string &data_file,
                            const ins::Array &data,
                            sol::optional<bool> append) {
  ins::signal::write_sigmf(data_file, data, append.value_or(true));
}

// Convert a Lua 1-based slice spec string to C++ 0-based.
// Positive numbers are decremented by 1; negative numbers and colons unchanged.
// Example: "1:3" → "0:2", "2,1:-1" → "1,0:-1", "::-1" → "::-1"
static std::string lua_spec_to_cpp(const std::string &spec) {
  std::string out;
  for (size_t i = 0; i < spec.size(); i++) {
    char c = spec[i];
    if (c == ':' || c == ',') {
      out += c;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
      size_t start = i;
      if (c == '-')
        i++;
      while (i < spec.size() && spec[i] >= '0' && spec[i] <= '9')
        i++;
      std::string num = spec.substr(start, i - start);
      int val = std::stoi(num);
      if (val > 0) {
        out += std::to_string(val - 1);
      } else {
        out += num;
      }
      i--;
    } else {
      out += c;
    }
  }
  return out;
}

static int lua_axis_to_cpp(int axis) {
  if (axis == 0)
    throw std::runtime_error("Lua axes are 1-based: axis 0 is invalid");
  return axis > 0 ? axis - 1 : axis;
}

static std::optional<int> lua_optional_axis_to_cpp(sol::optional<int> axis) {
  if (!axis)
    return std::nullopt;
  return lua_axis_to_cpp(*axis);
}

static int lua_axis_or_default_to_cpp(sol::optional<int> axis,
                                      int lua_default_axis) {
  return lua_axis_to_cpp(axis.value_or(lua_default_axis));
}

static std::vector<int> lua_axes_to_cpp(const std::vector<int> &axes) {
  std::vector<int> converted;
  converted.reserve(axes.size());
  for (int axis : axes)
    converted.push_back(lua_axis_to_cpp(axis));
  return converted;
}

static std::vector<int>
lua_axes_or_default_to_cpp(sol::optional<std::vector<int>> axes,
                           std::vector<int> defaults) {
  return lua_axes_to_cpp(axes.value_or(std::move(defaults)));
}

static std::vector<int> lua_axes_table_to_cpp(sol::table axes) {
  std::vector<int> converted;
  for (auto &kv : axes)
    converted.push_back(lua_axis_to_cpp(kv.second.as<int>()));
  return converted;
}

// ============================================================================
// Module entry point
// ============================================================================

#if defined(_WIN32) || defined(_WIN64)
#define INSIGHT_LUA_EXPORT __declspec(dllexport)
#else
#define INSIGHT_LUA_EXPORT __attribute__((visibility("default")))
#endif

extern "C" INSIGHT_LUA_EXPORT int luaopen__insight(lua_State *L) {
  sol::state_view lua(L);
  sol::table m = lua.create_table();

  // Auto-initialize with smart backend discovery
  if (!ins::is_initialized()) {
    // Add this .so's directory to LD_LIBRARY_PATH so dlopen finds backends
#ifndef _WIN32
    Dl_info dlinfo;
    if (dladdr((void *)luaopen__insight, &dlinfo) && dlinfo.dli_fname) {
      std::string path(dlinfo.dli_fname);
      auto pos = path.find_last_of('/');
      if (pos != std::string::npos) {
        std::string dir = path.substr(0, pos + 1);
        const char *ld = getenv("LD_LIBRARY_PATH");
        std::string ld_str = ld ? ld : "";
        if (ld_str.find(dir) == std::string::npos) {
          std::string new_ld = dir + ":" + ld_str;
          setenv("LD_LIBRARY_PATH", new_ld.c_str(), 1);
        }
      }
    }
#endif
    try {
      ins::init();
    } catch (...) {
      // Backend not available — user must call init() manually
    }
  }

  // Manual init — supports no args (auto-discover) or table of backend names
  m["init"] = [](sol::optional<sol::table> backends) {
    if (!backends.has_value()) {
      ins::init(); // auto-discover: CPU + first GPU if available
    } else {
      std::vector<std::string> be;
      for (size_t i = 1; i <= backends->size(); i++) {
        be.push_back(backends->get<std::string>(i));
      }
      ins::init(be);
    }
  };

  // Load additional backend after init
  m["load_backend"] = [](const std::string &backend) -> bool {
    try {
      ins::load_backend(backend);
      return true;
    } catch (...) {
      return false;
    }
  };

  // Check if a device kind is available (e.g., "cpu", "gpu")
  m["has_device"] = [](const std::string &kind) -> bool {
    if (kind == "cpu")
      return ins::has_device(ins::DeviceKind::CPU);
    if (kind == "gpu")
      return ins::has_device(ins::DeviceKind::GPU);
    return false;
  };

  // ===== Device information =====
  m["device_name"] = [](sol::optional<std::string> kind,
                        sol::optional<int> device_id) {
    std::string k = kind.value_or("cpu");
    if (k == "gpu")
      return ins::device_name(ins::DeviceKind::GPU, device_id.value_or(0));
    if (k == "cpu")
      return ins::device_name(ins::DeviceKind::CPU, device_id.value_or(0));
    throw std::runtime_error("device_name() expects 'cpu' or 'gpu'");
  };
  m["active_gpu_backend_name"] = []() { return ins::active_gpu_backend_name(); };
  m["active_gpu_backend_version"] = []() { return ins::active_gpu_backend_version(); };
  m["gpu_version"] = []() { return ins::gpu_runtime_version(); };
  m["driver_version"] = []() { return ins::driver_version(); };
  m["compute_capability"] = [](sol::optional<int> device_id) {
    return ins::compute_capability(device_id.value_or(0));
  };
  m["device_memory"] = [](sol::optional<int> device_id) {
    auto info = ins::device_memory(device_id.value_or(0));
    return std::make_tuple(info.total, info.free);
  };
  m["device_memory_info"] = [](int kind, sol::optional<int> device_id) {
    auto info = ins::device_memory_info(static_cast<ins::DeviceKind>(kind),
                                        device_id.value_or(0));
    return std::make_tuple(info.total, info.free);
  };
  m["gpu_count"] = []() {
    return static_cast<int>(ins::device_count(ins::DeviceKind::GPU));
  };
  m["device_count"] = [](int kind) {
    return static_cast<int>(
        ins::device_count(static_cast<ins::DeviceKind>(kind)));
  };

  // ===== ins::DType shortcuts (Paddle-style: ins.float32, ins.int64, ...)
  // =====
  m["float32"] = ins::DType::F32;
  m["float64"] = ins::DType::F64;
  m["float16"] = ins::DType::F16;
  m["bfloat16"] = ins::DType::BF16;
  m["int8"] = ins::DType::I8;
  m["int16"] = ins::DType::I16;
  m["int32"] = ins::DType::I32;
  m["int64"] = ins::DType::I64;
  m["uint8"] = ins::DType::U8;
  m["uint16"] = ins::DType::U16;
  m["uint32"] = ins::DType::U32;
  m["uint64"] = ins::DType::U64;
  m["bool"] = ins::DType::BOOL;
  m["complex64"] = ins::DType::C32;
  m["complex128"] = ins::DType::C64;

  // ===== ins::Place constructors =====
  sol::usertype<ins::Place> place_type =
      m.new_usertype<ins::Place>("Place", sol::constructors<ins::Place()>());
  place_type["is_cpu"] = &ins::Place::is_cpu;
  place_type["is_gpu"] = &ins::Place::is_gpu;
  place_type["device_id"] = &ins::Place::device_id;

  m["CPUPlace"] = []() { return ins::CPUPlace(); };
  m["GPUPlace"] = [](sol::optional<int> id, sol::this_state ts) -> ins::Place {
    try {
      return ins::GPUPlace(id.value_or(0));
    } catch (const std::exception &e) {
      luaL_error(ts, "%s", e.what());
      return ins::CPUPlace();
    }
  };

  m["set_device"] = [](const ins::Place &p) { ins::set_device(p); };
  m["get_device"] = []() { return ins::get_device(); };

  // ===== ins::Shape helper =====
  m["Shape"] = [](sol::table dims) { return ins::Shape(table_to_shape(dims)); };

  // ===== ins::Array usertype =====
  sol::usertype<ins::Array> array_type = m.new_usertype<ins::Array>(
      "Array",
      sol::constructors<ins::Array(), ins::Array(const ins::Shape &, ins::DType,
                                                 const ins::Place &)>());

  // Array(table) constructor: ins.Array{3,4,5} → creates Array from table
  // Must be set on the usertype table's metatable (not instances)
  {
    sol::stack::push(lua, m);
    lua_getfield(L, -1, "Array");
    if (!lua_isnil(L, -1)) {
      // Get or create the metatable for the Array table
      if (!lua_getmetatable(L, -1)) {
        lua_newtable(L);
        lua_setmetatable(L, -2);
        lua_getmetatable(L, -1);
      }
      // Now the metatable is on top of the stack
      lua_pushcfunction(L, [](lua_State *L) -> int {
        // Stack: table(self), args...
        sol::table t(L, 2); // first arg after self
        sol::optional<ins::Place> place;
        if (lua_gettop(L) > 2) {
          place = sol::stack::get<ins::Place>(L, 3);
        }
        try {
          ins::Array result = from_lua_table(t, place);
          sol::stack::push(L, std::move(result));
          return 1;
        } catch (const std::exception &e) {
          return luaL_error(L, "%s", e.what());
        }
      });
      lua_setfield(L, -2, "__call");
      lua_pop(L, 1); // pop metatable
    }
    lua_pop(L, 2); // pop Array table and module table
  }

  // Properties
  array_type["shape"] =
      sol::property([](const ins::Array &a, sol::this_state L) {
        sol::table t = sol::table::create(L.lua_state(), a.shape().ndim());
        for (int i = 0; i < a.shape().ndim(); i++) {
          t[i + 1] = a.shape().dim(i);
        }
        return t;
      });
  array_type["dtype"] = sol::property(&ins::Array::dtype);
  array_type["place"] = sol::property(&ins::Array::place);
  array_type["numel"] = sol::property(&ins::Array::numel);
  array_type["nbytes"] = sol::property(&ins::Array::nbytes);
  array_type["ndim"] =
      sol::property([](const ins::Array &a) { return a.shape().ndim(); });
  array_type["is_contiguous"] = sol::property(&ins::Array::is_contiguous);
  array_type["defined"] = sol::property(&ins::Array::defined);

  // Scalar extraction
  array_type["item"] = [](const ins::Array &a,
                          sol::this_state ts) -> sol::object {
    if (a.numel() != 1) {
      return sol::make_object(ts.lua_state(), sol::lua_nil);
    }
    ins::Array cpu =
        (a.place().kind() == ins::DeviceKind::CPU) ? a : a.to(ins::CPUPlace());
    double val = 0;
    switch (cpu.dtype()) {
    case ins::DType::F64:
      val = cpu.data<double>()[0];
      break;
    case ins::DType::F32:
      val = (double)cpu.data<float>()[0];
      break;
    case ins::DType::I64:
      val = (double)cpu.data<int64_t>()[0];
      break;
    case ins::DType::I32:
      val = (double)cpu.data<int32_t>()[0];
      break;
    case ins::DType::BOOL:
      val = (double)cpu.data<bool>()[0];
      break;
    default:
      return sol::make_object(ts.lua_state(), sol::lua_nil);
    }
    return sol::make_object(ts.lua_state(), val);
  };
  array_type["get"] = [](const ins::Array &a, int64_t idx) -> double {
    ins::Array cpu =
        (a.place().kind() == ins::DeviceKind::CPU) ? a : a.to(ins::CPUPlace());
    if (idx < 0 || idx >= cpu.numel())
      return 0.0;
    switch (cpu.dtype()) {
    case ins::DType::F64:
      return cpu.data<double>()[idx];
    case ins::DType::F32:
      return (double)cpu.data<float>()[idx];
    case ins::DType::I64:
      return (double)cpu.data<int64_t>()[idx];
    case ins::DType::I32:
      return (double)cpu.data<int32_t>()[idx];
    case ins::DType::I16:
      return (double)cpu.data<int16_t>()[idx];
    case ins::DType::I8:
      return (double)cpu.data<int8_t>()[idx];
    case ins::DType::U8:
      return (double)cpu.data<uint8_t>()[idx];
    case ins::DType::BOOL:
      return cpu.data<bool>()[idx] ? 1.0 : 0.0;
    default:
      return 0.0;
    }
  };

  // Bulk extraction: flatten to Lua table
  array_type["table"] = [](const ins::Array &a,
                           sol::this_state ts) -> sol::table {
    sol::state_view lv(ts);
    ins::Array cpu =
        (a.place().kind() == ins::DeviceKind::CPU) ? a : a.to(ins::CPUPlace());
    sol::table t = lv.create_table(cpu.numel());
    int64_t n = cpu.numel();
    switch (cpu.dtype()) {
    case ins::DType::F64: {
      const double *d = cpu.data<double>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = d[i];
      break;
    }
    case ins::DType::F32: {
      const float *d = cpu.data<float>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = (double)d[i];
      break;
    }
    case ins::DType::I64: {
      const int64_t *d = cpu.data<int64_t>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = (double)d[i];
      break;
    }
    case ins::DType::I32: {
      const int32_t *d = cpu.data<int32_t>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = (double)d[i];
      break;
    }
    case ins::DType::I16: {
      const int16_t *d = cpu.data<int16_t>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = (double)d[i];
      break;
    }
    case ins::DType::I8: {
      const int8_t *d = cpu.data<int8_t>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = (double)d[i];
      break;
    }
    case ins::DType::U64: {
      const uint64_t *d = cpu.data<uint64_t>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = (double)d[i];
      break;
    }
    case ins::DType::U32: {
      const uint32_t *d = cpu.data<uint32_t>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = (double)d[i];
      break;
    }
    case ins::DType::U16: {
      const uint16_t *d = cpu.data<uint16_t>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = (double)d[i];
      break;
    }
    case ins::DType::U8: {
      const uint8_t *d = cpu.data<uint8_t>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = (double)d[i];
      break;
    }
    case ins::DType::BOOL: {
      const bool *d = cpu.data<bool>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = d[i] ? 1.0 : 0.0;
      break;
    }
    case ins::DType::C64: {
      const std::complex<double> *d = cpu.data<std::complex<double>>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = d[i].real();
      break;
    }
    case ins::DType::C32: {
      const std::complex<float> *d = cpu.data<std::complex<float>>();
      for (int64_t i = 0; i < n; i++)
        t[i + 1] = (double)d[i].real();
      break;
    }
    default:
      break;
    }
    return t;
  };

  // View ops
  array_type["contiguous"] = &ins::Array::contiguous;
  array_type["reshape"] = [](const ins::Array &a, sol::table new_shape,
                             sol::this_state L) {
    try {
      return a.reshape(ins::Shape(table_to_shape(new_shape)));
    } catch (const std::exception &e) {
      return luaL_error(L, "%s", e.what()), ins::Array();
    }
  };
  array_type["transpose"] =
      sol::overload([](const ins::Array &a) { return a.transpose(); },
                    [](const ins::Array &a, std::vector<int> perm) {
                      return a.transpose(lua_axes_to_cpp(perm));
                    });
  array_type["squeeze"] = [](const ins::Array &a, sol::optional<int> axis) {
    return a.squeeze(lua_optional_axis_to_cpp(axis));
  };
  array_type["unsqueeze"] = [](const ins::Array &a, int dim) {
    return a.unsqueeze(lua_axis_to_cpp(dim));
  };

  // Device/type conversion
  // NOTE: No to(int) overload — DType values are numbers in Lua and would
  // conflict. Users must pass Place objects: ins.CPUPlace(), ins.GPUPlace(0).
  array_type["to"] = sol::overload(
      [](const ins::Array &a, const ins::Place &p,
         sol::this_state ts) -> ins::Array {
        try {
          return a.to(p);
        } catch (const std::exception &e) {
          luaL_error(ts, "%s", e.what());
          return ins::Array();
        }
      },
      [](const ins::Array &a, ins::DType dt) { return a.to(dt); },
      [](const ins::Array &a, const ins::Place &p, ins::DType dt,
         sol::this_state ts) -> ins::Array {
        try {
          return a.to(p, dt);
        } catch (const std::exception &e) {
          luaL_error(ts, "%s", e.what());
          return ins::Array();
        }
      });
  array_type["copy"] = &ins::Array::copy;

  // In-place mutation
  array_type["fill_"] = &ins::Array::fill_;
  array_type["copy_from_"] = &ins::Array::copy_from_;

  // String slicing: a[":,1:-1"] (Lua 1-based → auto-convert to 0-based)
  // Integer indexing: a[1] → first element, a[3] → third element
  array_type[sol::meta_function::index] = sol::overload(
      [](const ins::Array &a, const std::string &spec) {
        return a[lua_spec_to_cpp(spec)];
      },
      [](const ins::Array &a, int64_t idx) {
        // 1-based integer indexing → 0-based
        if (idx == 0)
          throw std::runtime_error(
              "Lua arrays are 1-based: index 0 is invalid");
        if (idx > 0)
          idx -= 1;
        return a.at({idx});
      },
      [](const ins::Array &a, const ins::Slice &s) { return a[s]; });

  // Arithmetic metamethods
  array_type[sol::meta_function::addition] = sol::overload(
      [](const ins::Array &a, const ins::Array &b) { return a + b; },
      [](const ins::Array &a, double b) { return a + b; },
      [](double a, const ins::Array &b) { return a + b; });
  array_type[sol::meta_function::subtraction] = sol::overload(
      [](const ins::Array &a, const ins::Array &b) { return a - b; },
      [](const ins::Array &a, double b) { return a - b; },
      [](double a, const ins::Array &b) { return a - b; });
  array_type[sol::meta_function::multiplication] = sol::overload(
      [](const ins::Array &a, const ins::Array &b) { return a * b; },
      [](const ins::Array &a, double b) { return a * b; },
      [](double a, const ins::Array &b) { return a * b; });
  array_type[sol::meta_function::division] = sol::overload(
      [](const ins::Array &a, const ins::Array &b) { return a / b; },
      [](const ins::Array &a, double b) { return a / b; },
      [](double a, const ins::Array &b) { return a / b; });
  array_type[sol::meta_function::modulus] = sol::overload(
      [](const ins::Array &a, const ins::Array &b) { return a % b; },
      [](const ins::Array &a, double b) { return a % b; });
  array_type[sol::meta_function::unary_minus] = [](const ins::Array &a) {
    return -a;
  };
  array_type[sol::meta_function::bitwise_not] = [](const ins::Array &a) {
    return ~a;
  };

  // Comparison metamethods
  array_type[sol::meta_function::equal_to] =
      [](const ins::Array &a, const ins::Array &b) { return ins::equal(a, b); };
  array_type[sol::meta_function::less_than] =
      [](const ins::Array &a, const ins::Array &b) { return ins::less(a, b); };
  array_type[sol::meta_function::less_than_or_equal_to] =
      [](const ins::Array &a, const ins::Array &b) {
        return ins::less_equal(a, b);
      };

  // String representation metamethod (enables print(arr) and tostring(arr))
  array_type[sol::meta_function::to_string] = &array_tostring;

  // ====================================================================
  // Creation
  // ====================================================================
  m["from_table"] = [](sol::table t,
                       sol::optional<ins::Place> place) -> ins::Array {
    try {
      return from_lua_table(t, place);
    } catch (const std::exception &e) {
      luaL_error(t.lua_state(), "%s", e.what());
      return ins::Array(); // unreachable, luaL_error longjmps
    }
  };
  m["zeros"] = [](sol::table shape, sol::optional<ins::DType> dtype,
                  sol::optional<ins::Place> place) {
    return ins::zeros(ins::Shape(table_to_shape(shape)),
                      dtype.value_or(ins::DType::F32),
                      place.value_or(ins::get_device()));
  };
  m["ones"] = [](sol::table shape, sol::optional<ins::DType> dtype,
                 sol::optional<ins::Place> place) {
    return ins::ones(ins::Shape(table_to_shape(shape)),
                     dtype.value_or(ins::DType::F32),
                     place.value_or(ins::get_device()));
  };
  m["full"] = [](sol::table shape, double fill, sol::optional<ins::DType> dtype,
                 sol::optional<ins::Place> place) {
    return ins::full(ins::Shape(table_to_shape(shape)), fill,
                     dtype.value_or(ins::DType::F32),
                     place.value_or(ins::get_device()));
  };
  m["eye"] = [](int64_t n, sol::optional<int64_t> m_opt,
                sol::optional<int64_t> k, sol::optional<ins::DType> dtype,
                sol::optional<ins::Place> place) {
    return ins::eye(n, m_opt.value_or(-1), k.value_or(0),
                    dtype.value_or(ins::DType::F32),
                    place.value_or(ins::get_device()));
  };
  m["arange"] = sol::overload(
      [](double end, sol::optional<ins::DType> dtype,
         sol::optional<ins::Place> place) {
        return ins::arange(end, dtype.value_or(ins::DType::I64),
                           place.value_or(ins::get_device()));
      },
      [](double start, double end, sol::optional<double> step,
         sol::optional<ins::DType> dtype, sol::optional<ins::Place> place) {
        return ins::arange(start, end, step.value_or(1.0),
                           dtype.value_or(ins::DType::I64),
                           place.value_or(ins::get_device()));
      });
  m["linspace"] = [](double start, double stop, int64_t num,
                     sol::optional<ins::DType> dtype,
                     sol::optional<ins::Place> place) {
    return ins::linspace(start, stop, num, dtype.value_or(ins::DType::F32),
                         place.value_or(ins::get_device()));
  };
  m["zeros_like"] = &ins::zeros_like;
  m["ones_like"] = &ins::ones_like;

  // ====================================================================
  // Elementwise
  // ====================================================================
  m["add"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::add(a, b);
  };
  m["sub"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::sub(a, b);
  };
  m["mul"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::mul(a, b);
  };
  m["div"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::div(a, b);
  };
  m["pow"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::pow(a, b);
  };
  m["mod"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::mod(a, b);
  };
  m["maximum"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::maximum(a, b);
  };
  m["minimum"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::minimum(a, b);
  };
  m["equal"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::equal(a, b);
  };
  m["not_equal"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::not_equal(a, b);
  };
  m["greater"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::greater(a, b);
  };
  m["less"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::less(a, b);
  };
  m["greater_equal"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::greater_equal(a, b);
  };
  m["less_equal"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::less_equal(a, b);
  };
  m["logical_and"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::logical_and(a, b);
  };
  m["logical_or"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::logical_or(a, b);
  };
  m["logical_xor"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::logical_xor(a, b);
  };
  m["logical_not"] = [](const ins::Array &x) { return ins::logical_not(x); };

  // ====================================================================
  // Unary math
  // ====================================================================
  m["abs"] = [](const ins::Array &x) { return ins::abs(x); };
  m["negative"] = [](const ins::Array &x) { return ins::negative(x); };
  m["square"] = [](const ins::Array &x) { return ins::square(x); };
  m["sqrt"] = [](const ins::Array &x) { return ins::sqrt(x); };
  m["exp"] = [](const ins::Array &x) { return ins::exp(x); };
  m["log"] = [](const ins::Array &x) { return ins::log(x); };
  m["log2"] = [](const ins::Array &x) { return ins::log2(x); };
  m["log10"] = [](const ins::Array &x) { return ins::log10(x); };
  m["sin"] = [](const ins::Array &x) { return ins::sin(x); };
  m["cos"] = [](const ins::Array &x) { return ins::cos(x); };
  m["tan"] = [](const ins::Array &x) { return ins::tan(x); };
  m["asin"] = [](const ins::Array &x) { return ins::asin(x); };
  m["acos"] = [](const ins::Array &x) { return ins::acos(x); };
  m["atan"] = [](const ins::Array &x) { return ins::atan(x); };
  m["sinh"] = [](const ins::Array &x) { return ins::sinh(x); };
  m["cosh"] = [](const ins::Array &x) { return ins::cosh(x); };
  m["tanh"] = [](const ins::Array &x) { return ins::tanh(x); };
  m["floor"] = [](const ins::Array &x) { return ins::floor(x); };
  m["ceil"] = [](const ins::Array &x) { return ins::ceil(x); };
  m["round"] = [](const ins::Array &x) { return ins::rint(x); };
  m["sign"] = [](const ins::Array &x) { return ins::sign(x); };
  m["conj"] = &ins::conj;
  m["angle"] = &ins::angle;
  m["isnan"] = [](const ins::Array &x) { return ins::isnan(x); };
  m["isinf"] = [](const ins::Array &x) { return ins::isinf(x); };
  m["isfinite"] = [](const ins::Array &x) { return ins::isfinite(x); };
  m["where"] = [](const ins::Array &cond_arr, const ins::Array &x,
                  const ins::Array &y) { return ins::where(cond_arr, x, y); };
  m["exp2"] = [](const ins::Array &x) { return ins::exp2(x); };
  m["expm1"] = [](const ins::Array &x) { return ins::expm1(x); };
  m["log1p"] = [](const ins::Array &x) { return ins::log1p(x); };
  m["cbrt"] = [](const ins::Array &x) { return ins::cbrt(x); };
  m["reciprocal"] = [](const ins::Array &x) { return ins::reciprocal(x); };
  m["asinh"] = [](const ins::Array &x) { return ins::asinh(x); };
  m["acosh"] = [](const ins::Array &x) { return ins::acosh(x); };
  m["atanh"] = [](const ins::Array &x) { return ins::atanh(x); };
  m["trunc"] = [](const ins::Array &x) { return ins::trunc(x); };
  m["deg2rad"] = [](const ins::Array &x) { return ins::deg2rad(x); };
  m["rad2deg"] = [](const ins::Array &x) { return ins::rad2deg(x); };

  // ====================================================================
  // Complex
  // ====================================================================
  m["is_complex"] = [](const ins::Array &x) { return ins::is_complex(x); };
  m["has_complex_shape"] = [](const ins::Array &x) {
    return ins::has_complex_shape(x);
  };
  m["to_complex"] = sol::overload(
      [](const ins::Array &real) { return ins::to_complex(real); },
      [](const ins::Array &real, const ins::Array &imag) {
        return ins::to_complex(real, imag);
      });
  m["as_complex"] = [](const ins::Array &x) { return ins::as_complex(x); };
  m["as_real"] = [](const ins::Array &x) { return ins::as_real(x); };
  m["real"] = [](const ins::Array &z) { return ins::real(z); };
  m["imag"] = [](const ins::Array &z) { return ins::imag(z); };

  // ====================================================================
  // Reduction
  // ====================================================================
  m["sum"] = [](const ins::Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::sum(x, ax, keepdims.value_or(false));
  };
  m["mean"] = [](const ins::Array &x, sol::optional<int> axis,
                 sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::mean(x, ax, keepdims.value_or(false));
  };
  m["max"] = [](const ins::Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::max(x, ax, keepdims.value_or(false));
  };
  m["min"] = [](const ins::Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::min(x, ax, keepdims.value_or(false));
  };
  m["prod"] = [](const ins::Array &x, sol::optional<int> axis,
                 sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::prod(x, ax, keepdims.value_or(false));
  };
  m["argmax"] = [](const ins::Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::argmax(x, ax, keepdims.value_or(false));
  };
  m["argmin"] = [](const ins::Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::argmin(x, ax, keepdims.value_or(false));
  };
  m["any"] = [](const ins::Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::any(x, ax, keepdims.value_or(false));
  };
  m["all"] = [](const ins::Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::all(x, ax, keepdims.value_or(false));
  };
  m["var"] = [](const ins::Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims, sol::optional<int> ddof) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::var(x, ax, keepdims.value_or(false), ddof.value_or(0));
  };
  m["std"] = [](const ins::Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims, sol::optional<int> ddof) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::std(x, ax, keepdims.value_or(false), ddof.value_or(0));
  };
  m["cumsum"] = [](const ins::Array &x, int axis) {
    return ins::cumsum(x, lua_axis_to_cpp(axis));
  };
  m["cumprod"] = [](const ins::Array &x, int axis) {
    return ins::cumprod(x, lua_axis_to_cpp(axis));
  };
  m["cummax"] = [](const ins::Array &x, int axis) {
    return ins::cummax(x, lua_axis_to_cpp(axis));
  };
  m["cummin"] = [](const ins::Array &x, int axis) {
    return ins::cummin(x, lua_axis_to_cpp(axis));
  };
  m["sem"] = [](const ins::Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims, sol::optional<int> ddof) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::sem(x, ax, keepdims.value_or(false), ddof.value_or(0));
  };
  m["count_nonzero"] = [](const ins::Array &x, sol::optional<int> axis,
                          sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::count_nonzero(x, ax, keepdims.value_or(false));
  };
  m["median"] = [](const ins::Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::median(x, ax, keepdims.value_or(false));
  };
  m["quantile"] = sol::overload(
      [](const ins::Array &x, double q, sol::optional<int> axis,
         sol::optional<bool> keepdims) {
        std::optional<int> ax = lua_optional_axis_to_cpp(axis);
        return ins::quantile(x, q, ax, keepdims.value_or(false));
      },
      [](const ins::Array &x, const ins::Array &q, sol::optional<int> axis,
         sol::optional<bool> keepdims) {
        std::optional<int> ax = lua_optional_axis_to_cpp(axis);
        return ins::quantile(x, q, ax, keepdims.value_or(false));
      });
  m["percentile"] = [](const ins::Array &x, double q, sol::optional<int> axis,
                       sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::percentile(x, q, ax, keepdims.value_or(false));
  };
  m["nansum"] = [](const ins::Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::nansum(x, ax, keepdims.value_or(false));
  };
  m["nanmean"] = [](const ins::Array &x, sol::optional<int> axis,
                    sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::nanmean(x, ax, keepdims.value_or(false));
  };
  m["nanmax"] = [](const ins::Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::nanmax(x, ax, keepdims.value_or(false));
  };
  m["nanmin"] = [](const ins::Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::nanmin(x, ax, keepdims.value_or(false));
  };
  m["nanstd"] = [](const ins::Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims, sol::optional<int> ddof) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::nanstd(x, ax, keepdims.value_or(false), ddof.value_or(0));
  };
  m["nanvar"] = [](const ins::Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims, sol::optional<int> ddof) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::nanvar(x, ax, keepdims.value_or(false), ddof.value_or(0));
  };

  // ====================================================================
  // Manipulation
  // ====================================================================
  m["concat"] = [](sol::table arrays, sol::optional<int> axis) {
    return ins::concat(table_to_arrays(arrays),
                       lua_axis_or_default_to_cpp(axis, 1));
  };
  m["stack"] = [](sol::table arrays, sol::optional<int> axis) {
    return ins::stack(table_to_arrays(arrays),
                      lua_axis_or_default_to_cpp(axis, 1));
  };
  m["split"] = [](const ins::Array &x, int sections, sol::optional<int> axis) {
    return ins::split(x, sections, lua_axis_or_default_to_cpp(axis, 1));
  };
  m["tile"] = [](const ins::Array &x, const ins::Shape &reps) {
    return ins::tile(x, reps);
  };
  m["repeat"] = [](const ins::Array &x, int repeats, sol::optional<int> axis) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::repeat(x, repeats, ax);
  };
  m["flip"] = [](const ins::Array &x, sol::optional<int> axis) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::flip(x, ax);
  };
  m["pad"] = [](const ins::Array &x, std::vector<int64_t> pad_width,
                sol::optional<double> val) {
    return ins::pad(x, pad_width, val.value_or(0.0));
  };
  m["flatten"] = [](const ins::Array &x) { return ins::flatten(x); };
  m["ravel"] = [](const ins::Array &x) { return ins::ravel(x); };
  m["roll"] = [](const ins::Array &x, int shift, sol::optional<int> axis) {
    std::optional<int> ax = lua_optional_axis_to_cpp(axis);
    return ins::roll(x, shift, ax);
  };
  m["permute"] = [](const ins::Array &x, std::vector<int> axes) {
    return ins::permute(x, lua_axes_to_cpp(axes));
  };
  m["swapaxes"] = [](const ins::Array &x, int axis1, int axis2) {
    return ins::swapaxes(x, lua_axis_to_cpp(axis1), lua_axis_to_cpp(axis2));
  };
  m["moveaxis"] = [](const ins::Array &x, int source, int destination) {
    return ins::moveaxis(x, lua_axis_to_cpp(source),
                         lua_axis_to_cpp(destination));
  };
  m["fliplr"] = [](const ins::Array &x) { return ins::fliplr(x); };
  m["flipud"] = [](const ins::Array &x) { return ins::flipud(x); };
  m["rot90"] = [](const ins::Array &x, sol::optional<int> k,
                  sol::optional<std::vector<int>> axes) {
    return ins::rot90(x, k.value_or(1),
                      lua_axes_or_default_to_cpp(axes, {1, 2}));
  };
  m["diag"] = [](const ins::Array &x, sol::optional<int> k) {
    return ins::diag(x, k.value_or(0));
  };
  m["diagonal"] = [](const ins::Array &x, sol::optional<int> offset,
                     sol::optional<int> axis1, sol::optional<int> axis2) {
    return ins::diagonal(x, offset.value_or(0),
                         lua_axis_or_default_to_cpp(axis1, 1),
                         lua_axis_or_default_to_cpp(axis2, 2));
  };
  m["tril"] = [](const ins::Array &x, sol::optional<int> k) {
    return ins::tril(x, k.value_or(0));
  };
  m["triu"] = [](const ins::Array &x, sol::optional<int> k) {
    return ins::triu(x, k.value_or(0));
  };
  m["diff"] = [](const ins::Array &x, sol::optional<int> n,
                 sol::optional<int> axis) {
    return ins::diff(x, n.value_or(1), lua_axis_or_default_to_cpp(axis, -1));
  };

  // ====================================================================
  // Linear Algebra
  // ====================================================================
  m["matmul"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::matmul(a, b);
  };
  m["dot"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::dot(a, b);
  };
  m["outer"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::outer(a, b);
  };
  m["det"] = [](const ins::Array &x) { return ins::det(x); };
  m["inv"] = [](const ins::Array &x) { return ins::inv(x); };
  m["solve"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::solve(a, b);
  };
  m["svd"] = [](const ins::Array &x) { return ins::svd(x); };
  m["eigh"] = [](const ins::Array &x) { return ins::eigh(x); };
  m["cholesky"] = [](const ins::Array &x, sol::optional<bool> lower) {
    return ins::cholesky(x, lower.value_or(true));
  };
  m["qr"] = [](const ins::Array &x) { return ins::qr(x); };
  m["norm"] = [](const ins::Array &x, sol::optional<double> ord) {
    return ins::norm(x, ord.value_or(2.0));
  };
  m["trace"] = [](const ins::Array &x) { return ins::trace(x); };
  m["lstsq"] = [](const ins::Array &a, const ins::Array &b) {
    return ins::lstsq(a, b);
  };
  m["cond"] = [](const ins::Array &x, sol::optional<double> p) {
    return ins::cond(x, p.value_or(2.0));
  };
  m["matrix_rank"] = [](const ins::Array &x) { return ins::matrix_rank(x); };
  m["matrix_power"] = [](const ins::Array &x, int n) {
    return ins::matrix_power(x, n);
  };
  m["slogdet"] = [](const ins::Array &x) { return ins::slogdet(x); };
  m["eigvalsh"] = [](const ins::Array &x, sol::optional<std::string> uplo) {
    return ins::eigvalsh(x, uplo.value_or("L"));
  };

  // ====================================================================
  // FFT
  // ====================================================================
  m["fft"] = [](const ins::Array &x, sol::optional<int> n,
                sol::optional<int> axis, sol::optional<std::string> norm_mode) {
    return ins::fft::fft(x, n.value_or(-1),
                         lua_axis_or_default_to_cpp(axis, -1),
                         norm_mode.value_or("backward"));
  };
  m["ifft"] = [](const ins::Array &x, sol::optional<int> n,
                 sol::optional<int> axis,
                 sol::optional<std::string> norm_mode) {
    return ins::fft::ifft(x, n.value_or(-1),
                          lua_axis_or_default_to_cpp(axis, -1),
                          norm_mode.value_or("backward"));
  };
  m["rfft"] = [](sol::this_state L, const ins::Array &x, sol::optional<int> n,
                 sol::optional<int> axis,
                 sol::optional<std::string> norm_mode) -> sol::object {
    try {
      auto result = ins::fft::rfft(x, n.value_or(-1),
                                   lua_axis_or_default_to_cpp(axis, -1),
                                   norm_mode.value_or("backward"));
      return sol::make_object(L, std::move(result));
    } catch (const std::exception &e) {
      luaL_error(L, "rfft: %s", e.what());
      return sol::nil;
    }
  };
  m["irfft"] = [](const ins::Array &x, sol::optional<int> n,
                  sol::optional<int> axis,
                  sol::optional<std::string> norm_mode) {
    return ins::fft::irfft(x, n.value_or(-1),
                           lua_axis_or_default_to_cpp(axis, -1),
                           norm_mode.value_or("backward"));
  };
  m["fftfreq"] = [](int64_t n, sol::optional<double> d) {
    return ins::fft::fftfreq(n, d.value_or(1.0));
  };
  m["rfftfreq"] = [](int64_t n, sol::optional<double> d) {
    return ins::fft::rfftfreq(n, d.value_or(1.0));
  };
  m["fftshift"] = [](const ins::Array &x, sol::optional<int> axis) {
    return ins::fft::fftshift(x, lua_axis_or_default_to_cpp(axis, -1));
  };
  m["ifftshift"] = [](const ins::Array &x, sol::optional<int> axis) {
    return ins::fft::ifftshift(x, lua_axis_or_default_to_cpp(axis, -1));
  };
  m["next_fast_len"] = [](int target) {
    return ins::fft::next_fast_len(target);
  };
  m["hfft"] = [](const ins::Array &x, sol::optional<int> n,
                 sol::optional<int> axis,
                 sol::optional<std::string> norm_mode) {
    return ins::fft::hfft(x, n.value_or(-1),
                          lua_axis_or_default_to_cpp(axis, -1),
                          norm_mode.value_or("backward"));
  };
  m["ihfft"] = [](const ins::Array &x, sol::optional<int> n,
                  sol::optional<int> axis,
                  sol::optional<std::string> norm_mode) {
    return ins::fft::ihfft(x, n.value_or(-1),
                           lua_axis_or_default_to_cpp(axis, -1),
                           norm_mode.value_or("backward"));
  };
  m["rfft2"] = [](const ins::Array &x, sol::optional<std::vector<int64_t>> s,
                  sol::optional<std::vector<int>> axes,
                  sol::optional<std::string> norm_mode) {
    return ins::fft::rfft2(x, s.value_or(std::vector<int64_t>{}),
                           lua_axes_or_default_to_cpp(axes, {-2, -1}),
                           norm_mode.value_or("backward"));
  };
  m["irfft2"] = [](const ins::Array &x, sol::optional<std::vector<int64_t>> s,
                   sol::optional<std::vector<int>> axes,
                   sol::optional<std::string> norm_mode) {
    return ins::fft::irfft2(x, s.value_or(std::vector<int64_t>{}),
                            lua_axes_or_default_to_cpp(axes, {-2, -1}),
                            norm_mode.value_or("backward"));
  };
  m["fft2"] = [](const ins::Array &x, sol::optional<sol::table> s,
                 sol::optional<sol::table> axes,
                 sol::optional<std::string> norm_mode) {
    std::vector<int64_t> s_vec;
    if (s) {
      for (auto &kv : *s)
        s_vec.push_back(kv.second.as<int64_t>());
    }
    std::vector<int> axes_vec =
        axes ? lua_axes_table_to_cpp(*axes) : std::vector<int>{-2, -1};
    return ins::fft::fft2(x, s_vec, axes_vec, norm_mode.value_or("backward"));
  };
  m["ifft2"] = [](const ins::Array &x, sol::optional<sol::table> s,
                  sol::optional<sol::table> axes,
                  sol::optional<std::string> norm_mode) {
    std::vector<int64_t> s_vec;
    if (s) {
      for (auto &kv : *s)
        s_vec.push_back(kv.second.as<int64_t>());
    }
    std::vector<int> axes_vec =
        axes ? lua_axes_table_to_cpp(*axes) : std::vector<int>{-2, -1};
    return ins::fft::ifft2(x, s_vec, axes_vec, norm_mode.value_or("backward"));
  };
  m["rfftn"] = [](const ins::Array &x, sol::optional<std::vector<int64_t>> s,
                  sol::optional<std::vector<int>> axes,
                  sol::optional<std::string> norm_mode) {
    return ins::fft::rfftn(x, s.value_or(std::vector<int64_t>{}),
                           lua_axes_or_default_to_cpp(axes, {}),
                           norm_mode.value_or("backward"));
  };
  m["irfftn"] = [](const ins::Array &x, sol::optional<std::vector<int64_t>> s,
                   sol::optional<std::vector<int>> axes,
                   sol::optional<std::string> norm_mode) {
    return ins::fft::irfftn(x, s.value_or(std::vector<int64_t>{}),
                            lua_axes_or_default_to_cpp(axes, {}),
                            norm_mode.value_or("backward"));
  };

  // ====================================================================
  // Signal (ins.signal.*)
  // ====================================================================
  {
    sol::table sig = m.create_named("signal");

    // --- ChirpMethod enum ---
    sol::table cm = sig.create_named("ChirpMethod");
    cm["linear"] = ins::signal::ChirpMethod::Linear;
    cm["quadratic"] = ins::signal::ChirpMethod::Quadratic;
    cm["logarithmic"] = ins::signal::ChirpMethod::Logarithmic;
    cm["hyperbolic"] = ins::signal::ChirpMethod::Hyperbolic;

    // --- Windows ---
    sig["general_cosine"] = [](int64_t M, std::vector<double> a,
                               sol::optional<bool> sym) {
      return ins::signal::general_cosine(M, a, sym.value_or(true));
    };
    sig["get_window"] = sol::overload(
        [](const std::string &w, int64_t Nx, sol::optional<bool> fftbins) {
          return ins::signal::get_window(w, Nx, fftbins.value_or(true));
        },
        [](const std::string &w, double param, int64_t Nx,
           sol::optional<bool> fftbins) {
          return ins::signal::get_window(w, param, Nx, fftbins.value_or(true));
        });
    sig["boxcar"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::boxcar(M, sym.value_or(true));
    };
    sig["triang"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::triang(M, sym.value_or(true));
    };
    sig["parzen"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::parzen(M, sym.value_or(true));
    };
    sig["bohman"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::bohman(M, sym.value_or(true));
    };
    sig["bartlett"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::bartlett(M, sym.value_or(true));
    };
    sig["cosine"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::cosine(M, sym.value_or(true));
    };
    sig["exponential"] = [](int64_t M, sol::optional<double> center,
                            sol::optional<double> tau,
                            sol::optional<bool> sym) {
      return ins::signal::exponential(M, center.value_or(-1.0),
                                      tau.value_or(1.0), sym.value_or(true));
    };
    sig["blackman"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::blackman(M, sym.value_or(true));
    };
    sig["nuttall"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::nuttall(M, sym.value_or(true));
    };
    sig["blackmanharris"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::blackmanharris(M, sym.value_or(true));
    };
    sig["flattop"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::flattop(M, sym.value_or(true));
    };
    sig["hann"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::hann(M, sym.value_or(true));
    };
    sig["general_hamming"] = [](int64_t M, double alpha,
                                sol::optional<bool> sym) {
      return ins::signal::general_hamming(M, alpha, sym.value_or(true));
    };
    sig["hamming"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::hamming(M, sym.value_or(true));
    };
    sig["tukey"] = [](int64_t M, sol::optional<double> alpha,
                      sol::optional<bool> sym) {
      return ins::signal::tukey(M, alpha.value_or(0.5), sym.value_or(true));
    };
    sig["barthann"] = [](int64_t M, sol::optional<bool> sym) {
      return ins::signal::barthann(M, sym.value_or(true));
    };
    sig["kaiser"] = [](int64_t M, double beta_val, sol::optional<bool> sym) {
      return ins::signal::kaiser(M, beta_val, sym.value_or(true));
    };
    sig["gaussian"] = [](int64_t M, double std_val, sol::optional<bool> sym) {
      return ins::signal::gaussian(M, std_val, sym.value_or(true));
    };
    sig["general_gaussian"] = [](int64_t M, double p, double sig_val,
                                 sol::optional<bool> sym) {
      return ins::signal::general_gaussian(M, p, sig_val, sym.value_or(true));
    };
    sig["chebwin"] = [](int64_t M, double at, sol::optional<bool> sym) {
      return ins::signal::chebwin(M, at, sym.value_or(true));
    };
    sig["taylor"] = [](int64_t M, sol::optional<int64_t> nbar,
                       sol::optional<double> sll, sol::optional<bool> norm_mode,
                       sol::optional<bool> sym) {
      return ins::signal::taylor(M, nbar.value_or(4), sll.value_or(-30.0),
                                 norm_mode.value_or(true), sym.value_or(true));
    };
    sig["qmf"] = &ins::signal::qmf;

    // --- Waveforms ---
    sig["sawtooth"] = [](const ins::Array &t, sol::optional<double> width) {
      return ins::signal::sawtooth(t, width.value_or(1.0));
    };
    sig["square_wf"] = [](const ins::Array &t, sol::optional<double> duty) {
      return ins::signal::square(t, duty.value_or(0.5));
    };
    sig["square"] = sig["square_wf"]; // alias for ins::signal::square wave
    sig["gausspulse"] = [](const ins::Array &t, sol::optional<double> fc,
                           sol::optional<double> bw, sol::optional<double> bwr,
                           sol::optional<double> tpr) {
      return ins::signal::gausspulse(t, fc.value_or(1000), bw.value_or(0.5),
                                     bwr.value_or(-6), tpr.value_or(-60));
    };
    sig["chirp"] = [](const ins::Array &t, double f0, double t1, double f1,
                      sol::optional<ins::signal::ChirpMethod> method,
                      sol::optional<double> phi,
                      sol::optional<bool> vertex_zero) {
      return ins::signal::chirp(
          t, f0, t1, f1, method.value_or(ins::signal::ChirpMethod::Linear),
          phi.value_or(0.0), vertex_zero.value_or(true));
    };
    sig["unit_impulse"] = [](sol::table shape, sol::optional<int64_t> idx,
                             sol::optional<ins::DType> dtype,
                             sol::optional<ins::Place> place) {
      return ins::signal::unit_impulse(
          ins::Shape(table_to_shape(shape)), idx.value_or(-1),
          dtype.value_or(ins::DType::F64), place.value_or(ins::get_device()));
    };

    // --- B-Splines ---
    sig["gauss_spline"] = &ins::signal::gauss_spline;
    sig["cubic"] = &ins::signal::cubic;
    sig["quadratic"] = &ins::signal::quadratic;

    // --- Filter Design ---
    sig["kaiser_beta"] = &ins::signal::kaiser_beta;
    sig["kaiser_atten"] = &ins::signal::kaiser_atten;
    sig["firwin"] = [](int64_t numtaps, std::vector<double> cutoff,
                       sol::optional<std::string> window,
                       sol::optional<std::string> pass_zero,
                       sol::optional<bool> scale) {
      return ins::signal::firwin(numtaps, cutoff, window.value_or("hamming"),
                                 pass_zero.value_or("lowpass"),
                                 scale.value_or(true));
    };
    sig["firwin2"] = [](int64_t numtaps, std::vector<double> freq,
                        std::vector<double> gain, sol::optional<int64_t> nfreqs,
                        sol::optional<std::string> window,
                        sol::optional<bool> antisymmetric) {
      return ins::signal::firwin2(numtaps, freq, gain, nfreqs.value_or(0),
                                  window.value_or("hamming"),
                                  antisymmetric.value_or(false));
    };
    sig["cmplx_sort"] = &ins::signal::cmplx_sort;

    // --- Convolution ---
    sig["fftconvolve"] = [](const ins::Array &in1, const ins::Array &in2,
                            sol::optional<std::string> mode) {
      return ins::signal::fftconvolve(in1, in2, mode.value_or("full"));
    };
    sig["correlate"] = [](const ins::Array &in1, const ins::Array &in2,
                          sol::optional<std::string> mode) {
      return ins::signal::correlate(in1, in2, mode.value_or("full"));
    };
    sig["convolve2d"] = [](const ins::Array &in1, const ins::Array &in2,
                           sol::optional<std::string> mode) {
      return ins::signal::convolve2d(in1, in2, mode.value_or("full"));
    };
    sig["correlate2d"] = [](const ins::Array &in1, const ins::Array &in2,
                            sol::optional<std::string> mode) {
      return ins::signal::correlate2d(in1, in2, mode.value_or("full"));
    };
    sig["choose_conv_method"] = [](const ins::Array &in1, const ins::Array &in2,
                                   sol::optional<std::string> mode) {
      return ins::signal::choose_conv_method(in1, in2, mode.value_or("full"));
    };
    sig["correlation_lags"] = [](int64_t in1_len, int64_t in2_len,
                                 sol::optional<std::string> mode) {
      return ins::signal::correlation_lags(in1_len, in2_len,
                                           mode.value_or("full"));
    };

    // --- Filtering ---
    sig["hilbert"] = [](const ins::Array &x, sol::optional<int64_t> N) {
      return ins::signal::hilbert(x, N.value_or(-1));
    };
    sig["hilbert2"] = [](const ins::Array &x, sol::optional<int64_t> N) {
      return ins::signal::hilbert2(x, N.value_or(-1));
    };
    sig["detrend"] = [](const ins::Array &data, sol::optional<int> axis,
                        sol::optional<std::string> type) {
      return ins::signal::detrend(data, lua_axis_or_default_to_cpp(axis, -1),
                                  type.value_or("linear"));
    };
    sig["wiener"] = [](const ins::Array &im,
                       sol::optional<std::vector<int64_t>> mysize,
                       sol::optional<double> noise) {
      return ins::signal::wiener(im, mysize.value_or(std::vector<int64_t>{}),
                                 noise.value_or(-1.0));
    };
    sig["firfilter"] = [](const ins::Array &b, const ins::Array &x,
                          sol::optional<int> axis) {
      return ins::signal::firfilter(b, x, lua_axis_or_default_to_cpp(axis, -1));
    };
    sig["lfilter"] = [](const ins::Array &b, const ins::Array &a,
                        const ins::Array &x, sol::optional<int> axis) {
      return ins::signal::lfilter(b, a, x,
                                  lua_axis_or_default_to_cpp(axis, -1));
    };
    sig["lfilter_zi"] = &ins::signal::lfilter_zi;
    sig["filtfilt"] = [](const ins::Array &b, const ins::Array &a,
                         const ins::Array &x, sol::optional<int> axis) {
      return ins::signal::filtfilt(b, a, x,
                                   lua_axis_or_default_to_cpp(axis, -1));
    };
    sig["decimate"] = [](const ins::Array &x, int64_t q,
                         sol::optional<int> axis,
                         sol::optional<bool> zero_phase) {
      return ins::signal::decimate(x, q, lua_axis_or_default_to_cpp(axis, -1),
                                   zero_phase.value_or(true));
    };
    sig["resample"] = [](const ins::Array &x, int64_t num,
                         sol::optional<int> axis) {
      return ins::signal::resample(x, num,
                                   lua_axis_or_default_to_cpp(axis, -1));
    };
    sig["resample_poly"] = [](const ins::Array &x, int64_t up, int64_t down,
                              sol::optional<int> axis) {
      return ins::signal::resample_poly(x, up, down,
                                        lua_axis_or_default_to_cpp(axis, -1));
    };
    sig["freq_shift"] = &ins::signal::freq_shift;
    sig["sosfilt"] = &ins::signal::sosfilt;
    sig["upfirdn"] = &ins::signal::upfirdn;
    sig["channelize_poly"] = &ins::signal::channelize_poly;

    // --- Spectral Analysis ---
    sig["welch"] = [](const ins::Array &x, sol::optional<double> fs,
                      sol::optional<std::string> window,
                      sol::optional<int64_t> nperseg,
                      sol::optional<int64_t> noverlap,
                      sol::optional<int64_t> nfft,
                      sol::optional<std::string> detrend,
                      sol::optional<bool> return_onesided,
                      sol::optional<std::string> scaling) {
      return ins::signal::welch(
          x, fs.value_or(1.0), window.value_or("hann"), nperseg.value_or(256),
          noverlap.value_or(0), nfft.value_or(0), detrend.value_or("constant"),
          return_onesided.value_or(true), scaling.value_or("density"));
    };
    sig["periodogram"] = [](const ins::Array &x, sol::optional<double> fs,
                            sol::optional<std::string> window,
                            sol::optional<int64_t> nfft,
                            sol::optional<std::string> detrend,
                            sol::optional<bool> return_onesided,
                            sol::optional<std::string> scaling) {
      return ins::signal::periodogram(
          x, fs.value_or(1.0), window.value_or("boxcar"), nfft.value_or(0),
          detrend.value_or("constant"), return_onesided.value_or(true),
          scaling.value_or("density"));
    };
    sig["csd"] = [](const ins::Array &x, const ins::Array &y,
                    sol::optional<double> fs, sol::optional<std::string> window,
                    sol::optional<int64_t> nperseg,
                    sol::optional<int64_t> noverlap,
                    sol::optional<int64_t> nfft,
                    sol::optional<std::string> detrend,
                    sol::optional<bool> return_onesided,
                    sol::optional<std::string> scaling) {
      return ins::signal::csd(x, y, fs.value_or(1.0), window.value_or("hann"),
                              nperseg.value_or(256), noverlap.value_or(0),
                              nfft.value_or(0), detrend.value_or("constant"),
                              return_onesided.value_or(true),
                              scaling.value_or("density"));
    };
    sig["coherence"] =
        [](const ins::Array &x, const ins::Array &y, sol::optional<double> fs,
           sol::optional<std::string> window, sol::optional<int64_t> nperseg,
           sol::optional<int64_t> noverlap, sol::optional<int64_t> nfft,
           sol::optional<std::string> detrend) {
          return ins::signal::coherence(
              x, y, fs.value_or(1.0), window.value_or("hann"),
              nperseg.value_or(256), noverlap.value_or(0), nfft.value_or(0),
              detrend.value_or("constant"));
        };
    sig["spectrogram"] = [](const ins::Array &x, sol::optional<double> fs,
                            sol::optional<std::string> window,
                            sol::optional<int64_t> nperseg,
                            sol::optional<int64_t> noverlap,
                            sol::optional<int64_t> nfft,
                            sol::optional<std::string> detrend,
                            sol::optional<bool> return_onesided,
                            sol::optional<std::string> mode) {
      return ins::signal::spectrogram(
          x, fs.value_or(1.0), window.value_or("hann"), nperseg.value_or(256),
          noverlap.value_or(0), nfft.value_or(0), detrend.value_or("constant"),
          return_onesided.value_or(true), mode.value_or("psd"));
    };
    sig["stft"] =
        [](const ins::Array &x, sol::optional<double> fs,
           sol::optional<std::string> window, sol::optional<int64_t> nperseg,
           sol::optional<int64_t> noverlap, sol::optional<int64_t> nfft) {
          return ins::signal::stft(x, fs.value_or(1.0), window.value_or("hann"),
                                   nperseg.value_or(256), noverlap.value_or(0),
                                   nfft.value_or(0));
        };
    sig["istft"] = [](const ins::Array &Zxx, sol::optional<double> fs,
                      sol::optional<std::string> window,
                      sol::optional<int64_t> nperseg,
                      sol::optional<int64_t> noverlap,
                      sol::optional<int64_t> nfft) {
      return ins::signal::istft(Zxx, fs.value_or(1.0), window.value_or("hann"),
                                nperseg.value_or(0), noverlap.value_or(0),
                                nfft.value_or(0));
    };
    sig["vectorstrength"] = &ins::signal::vectorstrength;
    sig["lombscargle"] = &ins::signal::lombscargle;

    // --- Wavelets ---
    sig["morlet"] = [](int64_t M, sol::optional<double> w,
                       sol::optional<double> s, sol::optional<bool> complete) {
      return ins::signal::morlet(M, w.value_or(5.0), s.value_or(1.0),
                                 complete.value_or(true));
    };
    sig["ricker"] = &ins::signal::ricker;
    sig["morlet2"] = [](int64_t M, double s, sol::optional<double> w) {
      return ins::signal::morlet2(M, s, w.value_or(5.0));
    };
    sig["cwt"] = [](const ins::Array &data, sol::function wavelet,
                    std::vector<double> widths) {
      auto wavelet_fn = [&wavelet](int64_t points, double a) -> ins::Array {
        sol::function_result r = wavelet(points, a);
        return r.get<ins::Array>();
      };
      return ins::signal::cwt(data, wavelet_fn, widths);
    };

    // --- Acoustics ---
    sig["mel2hz"] = &ins::signal::mel2hz;
    sig["hz2mel"] = &ins::signal::hz2mel;
    sig["mel_frequencies"] = [](int64_t n_mels, sol::optional<double> f_low,
                                sol::optional<double> f_high) {
      return ins::signal::mel_frequencies(n_mels, f_low.value_or(0.0),
                                          f_high.value_or(11000.0));
    };
    sig["hz2bark"] = &ins::signal::hz2bark;
    sig["bark2hz"] = &ins::signal::bark2hz;

    // --- Demod ---
    sig["fm_demod"] = [](const ins::Array &x, sol::optional<int> axis) {
      return ins::signal::fm_demod(x, lua_axis_or_default_to_cpp(axis, -1));
    };

    // --- Peak Finding ---
    sig["argrelextrema"] = &ins::signal::argrelextrema;
    sig["argrelmax"] = &ins::signal::argrelmax;
    sig["argrelmin"] = &ins::signal::argrelmin;

    // --- Radar ---
    sig["pulse_compression"] =
        [](sol::this_state L, const ins::Array &x, const ins::Array &tpl,
           sol::optional<bool> normalize, sol::optional<std::string> window,
           sol::optional<int64_t> nfft) -> sol::object {
      try {
        auto result = ins::signal::pulse_compression(
            x, tpl, normalize.value_or(false), window.value_or(""),
            nfft.value_or(0));
        return sol::make_object(L, std::move(result));
      } catch (const std::exception &e) {
        sol::state_view lv(L);
        lv.script("error('pulse_compression: " + std::string(e.what()) + "')");
        return sol::nil;
      }
    };
    sig["pulse_doppler"] = [](const ins::Array &x,
                              sol::optional<std::string> window,
                              sol::optional<int64_t> nfft) {
      return ins::signal::pulse_doppler(x, window.value_or(""),
                                        nfft.value_or(0));
    };
    sig["cfar_alpha"] = &ins::signal::cfar_alpha;
    sig["ca_cfar"] = [](sol::this_state L, const ins::Array &data,
                        sol::table guard_tbl, sol::table ref_tbl,
                        sol::optional<double> pfa) {
      std::vector<int> guard, ref;
      for (auto &kv : guard_tbl)
        guard.push_back(kv.second.as<int>());
      for (auto &kv : ref_tbl)
        ref.push_back(kv.second.as<int>());
      auto result = ins::signal::ca_cfar(data, guard, ref, pfa.value_or(1e-3));
      sol::state_view lua(L);
      sol::table ret = lua.create_table();
      ret[1] = result.first;
      ret[2] = result.second;
      return ret;
    };
    sig["mvdr"] = [](const ins::Array &x, const ins::Array &sv,
                     sol::optional<bool> calc_cov) {
      return ins::signal::mvdr(x, sv, calc_cov.value_or(true));
    };
    sig["ambgfun"] = [](const ins::Array &x, double fs, double prf,
                        sol::optional<ins::Array> y) {
      return ins::signal::ambgfun(x, fs, prf, y.value_or(ins::Array()));
    };

    // --- Signal I/O ---
    sig["read_bin"] = &lua_read_bin;
    sig["write_bin"] = &lua_write_bin;
    sig["unpack_bin"] = &lua_unpack_bin;
    sig["pack_bin"] = &ins::signal::pack_bin;
    sig["read_sigmf"] = &ins::signal::read_sigmf;
    sig["write_sigmf"] = &lua_write_sigmf;

    // --- Estimation (KalmanFilter) ---
    sol::usertype<ins::signal::KalmanFilter> kf_type =
        sig.new_usertype<ins::signal::KalmanFilter>("KalmanFilter");
    kf_type["x"] = &ins::signal::KalmanFilter::x;
    kf_type["P"] = &ins::signal::KalmanFilter::P;
    kf_type["z"] = &ins::signal::KalmanFilter::z;
    kf_type["R"] = &ins::signal::KalmanFilter::R;
    kf_type["Q"] = &ins::signal::KalmanFilter::Q;
    kf_type["F"] = &ins::signal::KalmanFilter::F;
    kf_type["H"] = &ins::signal::KalmanFilter::H;
    kf_type["dim_x"] = &ins::signal::KalmanFilter::dim_x;
    kf_type["dim_z"] = &ins::signal::KalmanFilter::dim_z;
    kf_type["points"] = &ins::signal::KalmanFilter::points;
    kf_type["predict"] = &ins::signal::KalmanFilter::predict;
    kf_type["update"] = &ins::signal::KalmanFilter::update;
    // Factory function for constructing KalmanFilter with optional args
    sig["KalmanFilter_new"] = [](int dim_x, int dim_z, sol::optional<int> dim_u,
                                 sol::optional<int> points,
                                 sol::optional<ins::DType> dtype) {
      return ins::signal::KalmanFilter(dim_x, dim_z, dim_u.value_or(0),
                                       points.value_or(1),
                                       dtype.value_or(ins::DType::F64));
    };

    // Top-level aliases
    sig["convolve"] = &ins::convolve;
    sig["unwrap"] = [](const ins::Array &p, sol::optional<int> axis,
                       sol::optional<double> discont,
                       sol::optional<double> period) {
      return ins::unwrap(p, lua_axis_or_default_to_cpp(axis, -1),
                         discont.value_or(M_PI), period.value_or(2 * M_PI));
    };
    sig["sinc"] = &ins::sinc;
  }

  // Top-level aliases (scipy-compatible)
  m["convolve"] = &ins::convolve;
  m["unwrap"] = [](const ins::Array &p, sol::optional<int> axis,
                   sol::optional<double> discont,
                   sol::optional<double> period) {
    return ins::unwrap(p, lua_axis_or_default_to_cpp(axis, -1),
                       discont.value_or(M_PI), period.value_or(2 * M_PI));
  };
  m["sinc"] = &ins::sinc;

  // ====================================================================
  // Plot (ins.plot.*, requires INSIGHT_USE_MATPLOT)
  // ====================================================================
#ifdef INSIGHT_USE_MATPLOT
  {
    // Ignore SIGPIPE to prevent process crash when gnuplot is unavailable.
    // matplotplusplus spawns gnuplot via popen; if gnuplot is not installed
    // the pipe breaks and SIGPIPE would kill the process.
    std::signal(SIGPIPE, SIG_IGN);

    sol::table plt = m.create_named("plot");

    plt["figure"] = [](sol::optional<int> number) {
      try {
        ins::plot::figure(number.value_or(-1));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["subplot"] = [](int r, int c, int i) {
      try {
        ins::plot::subplot(r, c, i);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["hold"] = [](bool on) {
      try {
        ins::plot::hold(on);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["clf"] = []() {
      try {
        ins::plot::clf();
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["show"] = []() {
      try {
        ins::plot::show();
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["save"] = [](const std::string &fn) {
      try {
        ins::plot::save(fn);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["plot"] = sol::overload(
        [](const ins::Array &y, sol::optional<std::string> fmt) {
          try {
            ins::plot::plot(y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, const ins::Array &y,
           sol::optional<std::string> fmt) {
          try {
            ins::plot::plot(x, y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["plot3"] = [](const ins::Array &x, const ins::Array &y,
                      const ins::Array &z, sol::optional<std::string> fmt) {
      try {
        ins::plot::plot3(x, y, z, fmt.value_or(""));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["stairs"] = sol::overload(
        [](const ins::Array &y, sol::optional<std::string> fmt) {
          try {
            ins::plot::stairs(y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, const ins::Array &y,
           sol::optional<std::string> fmt) {
          try {
            ins::plot::stairs(x, y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["errorbar"] = [](const ins::Array &x, const ins::Array &y,
                         const ins::Array &err,
                         sol::optional<std::string> fmt) {
      try {
        ins::plot::errorbar(x, y, err, fmt.value_or(""));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["area"] = sol::overload(
        [](const ins::Array &y) {
          try {
            ins::plot::area(y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, const ins::Array &y) {
          try {
            ins::plot::area(x, y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["fill"] = [](const ins::Array &x, const ins::Array &y,
                     sol::optional<std::string> color) {
      try {
        ins::plot::fill(x, y, color.value_or("b"));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["line"] = [](const ins::Array &x, const ins::Array &y,
                     sol::optional<std::string> fmt) {
      try {
        ins::plot::line(x, y, fmt.value_or(""));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["semilogx"] = sol::overload(
        [](const ins::Array &y, sol::optional<std::string> fmt) {
          try {
            ins::plot::semilogx(y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, const ins::Array &y,
           sol::optional<std::string> fmt) {
          try {
            ins::plot::semilogx(x, y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["semilogy"] = sol::overload(
        [](const ins::Array &y, sol::optional<std::string> fmt) {
          try {
            ins::plot::semilogy(y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, const ins::Array &y,
           sol::optional<std::string> fmt) {
          try {
            ins::plot::semilogy(x, y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["loglog"] = sol::overload(
        [](const ins::Array &y, sol::optional<std::string> fmt) {
          try {
            ins::plot::loglog(y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, const ins::Array &y,
           sol::optional<std::string> fmt) {
          try {
            ins::plot::loglog(x, y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });

    plt["scatter"] = [](const ins::Array &x, const ins::Array &y,
                        sol::optional<double> size) {
      try {
        ins::plot::scatter(x, y, size.value_or(20.0));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["scatter3"] = [](const ins::Array &x, const ins::Array &y,
                         const ins::Array &z, sol::optional<double> size) {
      try {
        ins::plot::scatter3(x, y, z, size.value_or(20.0));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["binscatter"] = [](const ins::Array &x, const ins::Array &y) {
      try {
        ins::plot::binscatter(x, y);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["bar"] = sol::overload(
        [](const ins::Array &y, sol::optional<double> w) {
          try {
            ins::plot::bar(y, w.value_or(0.8));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, const ins::Array &y, sol::optional<double> w) {
          try {
            ins::plot::bar(x, y, w.value_or(0.8));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["barh"] = sol::overload(
        [](const ins::Array &y, sol::optional<double> w) {
          try {
            ins::plot::barh(y, w.value_or(0.8));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, const ins::Array &y, sol::optional<double> w) {
          try {
            ins::plot::barh(x, y, w.value_or(0.8));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["barstacked"] = sol::overload(
        [](const ins::Array &Y) {
          try {
            ins::plot::barstacked(Y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, const ins::Array &Y) {
          try {
            ins::plot::barstacked(x, Y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["pareto"] = sol::overload(
        [](const ins::Array &y) {
          try {
            ins::plot::pareto(y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, const ins::Array &y) {
          try {
            ins::plot::pareto(x, y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });

    plt["hist"] = [](const ins::Array &data, sol::optional<int> bins) {
      try {
        ins::plot::hist(data, bins.value_or(10));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["hist2"] = [](const ins::Array &x, const ins::Array &y,
                      sol::optional<int> bins) {
      try {
        ins::plot::hist2(x, y, bins.value_or(10));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["boxplot"] = sol::overload(
        [](const ins::Array &data) {
          try {
            ins::plot::boxplot(data);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &data, std::vector<std::string> groups) {
          try {
            ins::plot::boxplot(data, groups);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["heatmap"] = [](const ins::Array &data) {
      try {
        ins::plot::heatmap(data);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["stem"] = sol::overload(
        [](const ins::Array &y, sol::optional<std::string> fmt) {
          try {
            ins::plot::stem(y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, const ins::Array &y,
           sol::optional<std::string> fmt) {
          try {
            ins::plot::stem(x, y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["stem3"] = [](const ins::Array &x, const ins::Array &y,
                      const ins::Array &z, sol::optional<std::string> fmt) {
      try {
        ins::plot::stem3(x, y, z, fmt.value_or(""));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["contour"] = [](const ins::Array &X, const ins::Array &Y,
                        const ins::Array &Z, sol::optional<int> levels) {
      try {
        ins::plot::contour(X, Y, Z, levels.value_or(10));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["contourf"] = [](const ins::Array &X, const ins::Array &Y,
                         const ins::Array &Z, sol::optional<int> levels) {
      try {
        ins::plot::contourf(X, Y, Z, levels.value_or(10));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["pcolor"] = [](const ins::Array &C) {
      try {
        ins::plot::pcolor(C);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["surf"] = [](const ins::Array &X, const ins::Array &Y,
                     const ins::Array &Z) {
      try {
        ins::plot::surf(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["surfc"] = [](const ins::Array &X, const ins::Array &Y,
                      const ins::Array &Z) {
      try {
        ins::plot::surfc(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["mesh"] = [](const ins::Array &X, const ins::Array &Y,
                     const ins::Array &Z) {
      try {
        ins::plot::mesh(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["meshc"] = [](const ins::Array &X, const ins::Array &Y,
                      const ins::Array &Z) {
      try {
        ins::plot::meshc(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["meshz"] = [](const ins::Array &X, const ins::Array &Y,
                      const ins::Array &Z) {
      try {
        ins::plot::meshz(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["waterfall"] = [](const ins::Array &X, const ins::Array &Y,
                          const ins::Array &Z) {
      try {
        ins::plot::waterfall(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["ribbon"] = sol::overload(
        [](const ins::Array &y) {
          try {
            ins::plot::ribbon(y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, const ins::Array &y) {
          try {
            ins::plot::ribbon(x, y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["fence"] = [](const ins::Array &X, const ins::Array &Y,
                      const ins::Array &Z) {
      try {
        ins::plot::fence(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["quiver"] = [](const ins::Array &X, const ins::Array &Y,
                       const ins::Array &U, const ins::Array &V,
                       sol::optional<double> scale) {
      try {
        ins::plot::quiver(X, Y, U, V, scale.value_or(1.0));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["quiver3"] = [](const ins::Array &X, const ins::Array &Y,
                        const ins::Array &Z, const ins::Array &U,
                        const ins::Array &V, const ins::Array &W) {
      try {
        ins::plot::quiver3(X, Y, Z, U, V, W);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["feather"] = [](const ins::Array &u, const ins::Array &v) {
      try {
        ins::plot::feather(u, v);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["polarplot"] = [](const ins::Array &theta, const ins::Array &rho,
                          sol::optional<std::string> fmt) {
      try {
        ins::plot::polarplot(theta, rho, fmt.value_or(""));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["polarscatter"] = [](const ins::Array &theta, const ins::Array &rho,
                             sol::optional<double> size) {
      try {
        ins::plot::polarscatter(theta, rho, size.value_or(20.0));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["polarhistogram"] = [](const ins::Array &data,
                               sol::optional<int> bins) {
      try {
        ins::plot::polarhistogram(data, bins.value_or(10));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["compass"] = [](const ins::Array &u, const ins::Array &v) {
      try {
        ins::plot::compass(u, v);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["ezpolar"] = [](const std::string &expr) {
      try {
        ins::plot::ezpolar(expr);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["pie"] = sol::overload(
        [](const ins::Array &x) {
          try {
            ins::plot::pie(x);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const ins::Array &x, std::vector<std::string> labels) {
          try {
            ins::plot::pie(x, labels);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });

    plt["imshow"] = [](const ins::Array &data) {
      try {
        ins::plot::imshow(data);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["image"] = [](const ins::Array &data) {
      try {
        ins::plot::image(data);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["imagesc"] = [](const ins::Array &data) {
      try {
        ins::plot::imagesc(data);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["graph"] = [](const std::vector<int> &s, const std::vector<int> &t) {
      try {
        ins::plot::graph(s, t);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["digraph"] = [](const std::vector<int> &s, const std::vector<int> &t) {
      try {
        ins::plot::digraph(s, t);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["parallelplot"] = [](const ins::Array &data) {
      try {
        ins::plot::parallelplot(data);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["plotmatrix"] = [](const ins::Array &X) {
      try {
        ins::plot::plotmatrix(X);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["wordcloud"] = [](const std::vector<std::string> &w,
                          const std::vector<double> &s) {
      try {
        ins::plot::wordcloud(w, s);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["rgbplot"] = [](const ins::Array &data) {
      try {
        ins::plot::rgbplot(data);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["title"] = [](const std::string &text) {
      try {
        ins::plot::title(text);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["subtitle"] = [](const std::string &text) {
      try {
        ins::plot::subtitle(text);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["xlabel"] = [](const std::string &text) {
      try {
        ins::plot::xlabel(text);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["ylabel"] = [](const std::string &text) {
      try {
        ins::plot::ylabel(text);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["zlabel"] = [](const std::string &text) {
      try {
        ins::plot::zlabel(text);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["legend"] = [](const std::vector<std::string> &labels) {
      try {
        ins::plot::legend(labels);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["xlim"] = [](double xmin, double xmax) {
      try {
        ins::plot::xlim(xmin, xmax);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["ylim"] = [](double ymin, double ymax) {
      try {
        ins::plot::ylim(ymin, ymax);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["zlim"] = [](double zmin, double zmax) {
      try {
        ins::plot::zlim(zmin, zmax);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["axis_equal"] = []() {
      try {
        ins::plot::axis_equal();
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["axis_tight"] = []() {
      try {
        ins::plot::axis_tight();
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["axis_off"] = []() {
      try {
        ins::plot::axis_off();
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["axis_ij"] = []() {
      try {
        ins::plot::axis_ij();
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["grid"] = [](bool on) {
      try {
        ins::plot::grid(on);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["xticks"] = [](const std::vector<double> &t) {
      try {
        ins::plot::xticks(t);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["yticks"] = [](const std::vector<double> &t) {
      try {
        ins::plot::yticks(t);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["zticks"] = [](const std::vector<double> &t) {
      try {
        ins::plot::zticks(t);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["xticklabels"] = [](const std::vector<std::string> &l) {
      try {
        ins::plot::xticklabels(l);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["yticklabels"] = [](const std::vector<std::string> &l) {
      try {
        ins::plot::yticklabels(l);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["xtickangle"] = [](double angle_val) {
      try {
        ins::plot::xtickangle(angle_val);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["ytickangle"] = [](double angle_val) {
      try {
        ins::plot::ytickangle(angle_val);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["colormap"] = [](ins::plot::Colormap cmap) {
      try {
        ins::plot::colormap(cmap);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["colorbar"] = [](sol::optional<bool> on) {
      try {
        ins::plot::colorbar(on.value_or(true));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["text"] = [](double x, double y, const std::string &s) {
      try {
        ins::plot::text(x, y, s);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["arrow"] = [](double x1, double y1, double x2, double y2) {
      try {
        ins::plot::arrow(x1, y1, x2, y2);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["rectangle"] = [](double x, double y, double w, double h) {
      try {
        ins::plot::rectangle(x, y, w, h);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["ellipse"] = [](double cx, double cy, double rx, double ry) {
      try {
        ins::plot::ellipse(cx, cy, rx, ry);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["view"] = [](double az, double el) {
      try {
        ins::plot::view(az, el);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
  }
#endif // INSIGHT_USE_MATPLOT

  // ====================================================================
  // Random
  // ====================================================================
  m["rand"] = [](sol::table shape, sol::optional<ins::DType> dtype,
                 sol::optional<ins::Place> place) {
    return ins::rand(ins::Shape(table_to_shape(shape)),
                     dtype.value_or(ins::DType::F32),
                     place.value_or(ins::get_device()));
  };
  m["randn"] = [](sol::table shape, sol::optional<ins::DType> dtype,
                  sol::optional<ins::Place> place) {
    return ins::randn(ins::Shape(table_to_shape(shape)),
                      dtype.value_or(ins::DType::F32),
                      place.value_or(ins::get_device()));
  };
  m["randint"] = [](int64_t low, int64_t high, sol::table shape,
                    sol::optional<ins::DType> dtype,
                    sol::optional<ins::Place> place) {
    return ins::randint(low, high, ins::Shape(table_to_shape(shape)),
                        dtype.value_or(ins::DType::I64),
                        place.value_or(ins::get_device()));
  };
  m["randperm"] = [](int64_t n, sol::optional<ins::DType> dtype,
                     sol::optional<ins::Place> place) {
    return ins::randperm(n, dtype.value_or(ins::DType::I64),
                         place.value_or(ins::get_device()));
  };
  m["seed"] = [](uint64_t s) { return ins::seed(s); };
  m["get_seed"] = []() { return ins::get_seed(); };
  m["rand_like"] = [](const ins::Array &x) { return ins::rand_like(x); };
  m["randn_like"] = [](const ins::Array &x) { return ins::randn_like(x); };
  m["exponential"] = [](double scale, sol::table shape,
                        sol::optional<ins::DType> dtype,
                        sol::optional<ins::Place> place) {
    return ins::exponential(scale, ins::Shape(table_to_shape(shape)),
                            dtype.value_or(ins::DType::F32),
                            place.value_or(ins::get_device()));
  };
  m["gamma"] = [](double shape_param, double rate, sol::table shape,
                  sol::optional<ins::DType> dtype,
                  sol::optional<ins::Place> place) {
    return ins::gamma(shape_param, rate, ins::Shape(table_to_shape(shape)),
                      dtype.value_or(ins::DType::F32),
                      place.value_or(ins::get_device()));
  };
  m["beta"] = [](double a, double b, sol::table shape,
                 sol::optional<ins::DType> dtype,
                 sol::optional<ins::Place> place) {
    return ins::beta(a, b, ins::Shape(table_to_shape(shape)),
                     dtype.value_or(ins::DType::F32),
                     place.value_or(ins::get_device()));
  };
  m["binomial"] = [](int64_t n, double p, sol::table shape,
                     sol::optional<ins::DType> dtype,
                     sol::optional<ins::Place> place) {
    return ins::binomial(n, p, ins::Shape(table_to_shape(shape)),
                         dtype.value_or(ins::DType::I64),
                         place.value_or(ins::get_device()));
  };
  m["poisson"] = [](double lam, sol::table shape,
                    sol::optional<ins::DType> dtype,
                    sol::optional<ins::Place> place) {
    return ins::poisson(lam, ins::Shape(table_to_shape(shape)),
                        dtype.value_or(ins::DType::I64),
                        place.value_or(ins::get_device()));
  };

  // ====================================================================
  // Cast
  // ====================================================================
  m["cast"] = [](const ins::Array &input, ins::DType target_dtype,
                 sol::optional<bool> copy) {
    return ins::cast(input, target_dtype, copy.value_or(true));
  };

  // ====================================================================
  // Indexing
  // ====================================================================
  sol::usertype<ins::UniqueResult> ur_type = m.new_usertype<ins::UniqueResult>(
      "UniqueResult", sol::constructors<ins::UniqueResult()>());
  ur_type["unique"] = &ins::UniqueResult::unique;
  ur_type["indices"] = &ins::UniqueResult::indices;
  ur_type["inverse"] = &ins::UniqueResult::inverse;
  ur_type["counts"] = &ins::UniqueResult::counts;

  m["take"] = &ins::take;
  m["nonzero"] = &ins::nonzero;
  m["flatnonzero"] = &ins::flatnonzero;

  // Fast slice: integer params, no string parsing overhead
  // dim: 1-based, start: 1-based inclusive, stop: 1-based exclusive
  m["slice"] = [](const ins::Array &a, int dim, int64_t start, int64_t stop) {
    return a.slice(lua_axis_to_cpp(dim), start - 1, stop - 1);
  };
  m["masked_select"] = &ins::masked_select;
  m["argsort"] = [](const ins::Array &x, sol::optional<int> axis,
                    sol::optional<bool> descending) {
    return ins::argsort(x, lua_axis_or_default_to_cpp(axis, -1),
                        descending.value_or(false));
  };
  m["searchsorted"] = [](const ins::Array &x, const ins::Array &v,
                         sol::optional<std::string> side) {
    return ins::searchsorted(x, v, side.value_or("left"));
  };
  m["sort"] = [](const ins::Array &x, sol::optional<int> axis,
                 sol::optional<bool> descending) {
    return ins::sort(x, lua_axis_or_default_to_cpp(axis, -1),
                     descending.value_or(false));
  };
  m["unique"] = [](const ins::Array &x, sol::optional<bool> return_indices,
                   sol::optional<bool> return_inverse,
                   sol::optional<bool> return_counts) {
    return ins::unique(x, return_indices.value_or(false),
                       return_inverse.value_or(false),
                       return_counts.value_or(false));
  };
  m["topk"] = [](const ins::Array &x, int64_t k, sol::optional<int> axis,
                 sol::optional<bool> largest, sol::optional<bool> sorted) {
    return ins::topk(x, k, lua_axis_or_default_to_cpp(axis, -1),
                     largest.value_or(true), sorted.value_or(true));
  };
  m["gather"] = [](const ins::Array &x, int dim, const ins::Array &index) {
    return ins::gather(x, lua_axis_to_cpp(dim), index);
  };
  m["scatter"] = [](const ins::Array &x, int dim, const ins::Array &index,
                    const ins::Array &src) {
    return ins::scatter(x, lua_axis_to_cpp(dim), index, src);
  };
  m["scatter_add"] = [](const ins::Array &x, int dim, const ins::Array &index,
                        const ins::Array &src) {
    return ins::scatter_add(x, lua_axis_to_cpp(dim), index, src);
  };
  m["scatter_reduce"] = [](const ins::Array &x, int dim,
                           const ins::Array &index, const ins::Array &src,
                           sol::optional<std::string> reduce) {
    return ins::scatter_reduce(x, lua_axis_to_cpp(dim), index, src,
                               reduce.value_or("replace"));
  };
  m["interp"] = [](const ins::Array &x, const ins::Array &xp,
                   const ins::Array &fp, sol::optional<double> left,
                   sol::optional<double> right) {
    std::optional<double> l =
        left ? std::optional<double>(*left) : std::nullopt;
    std::optional<double> r =
        right ? std::optional<double>(*right) : std::nullopt;
    return ins::interp(x, xp, fp, l, r);
  };
  m["indices"] = [](const ins::Shape &shape, sol::optional<bool> sparse) {
    return ins::indices(shape, sparse.value_or(false));
  };
  m["ix_"] = [](sol::table arrays) {
    return ins::ix_(table_to_arrays(arrays));
  };

  // ================================================================
  // Profiler / Timer
  // ================================================================

  m["timer_create"] = [](int device_type, int device_id) -> void * {
    InsightPlace place;
    place.device_type = static_cast<int32_t>(device_type);
    place.device_id = static_cast<int32_t>(device_id);
    InsightTimer timer = nullptr;
    C_Status status = insight_timer_create(&place, &timer);
    if (status != C_SUCCESS) {
      return nullptr;
    }
    return reinterpret_cast<void *>(timer);
  };

  m["timer_destroy"] = [](void *handle) {
    if (handle != nullptr) {
      insight_timer_destroy(reinterpret_cast<InsightTimer>(handle));
    }
  };

  m["timer_start"] = [](void *handle) {
    insight_timer_start(reinterpret_cast<InsightTimer>(handle));
  };

  m["timer_stop"] = [](void *handle) {
    insight_timer_stop(reinterpret_cast<InsightTimer>(handle));
  };

  m["timer_elapsed_ms"] = [](void *handle) -> double {
    float ms = 0.0f;
    insight_timer_elapsed_ms(reinterpret_cast<InsightTimer>(handle), &ms);
    return static_cast<double>(ms);
  };

  // ── Profiler bindings ──────────────────────────────────────────────

  m["profiler_create"] = [](int device_type, int device_id,
                            sol::optional<const char *> name) -> void * {
    try {
      InsightPlace place;
      place.device_type = static_cast<int32_t>(device_type);
      place.device_id = static_cast<int32_t>(device_id);
      C_Profiler prof = nullptr;
      C_Status status =
          insight_profiler_create(&place, name.value_or(nullptr), &prof);
      if (status != C_SUCCESS) {
        return nullptr;
      }
      return reinterpret_cast<void *>(prof);
    } catch (...) {
      return nullptr;
    }
  };

  m["profiler_destroy"] = [](void *handle) {
    try {
      if (handle != nullptr) {
        insight_profiler_destroy(reinterpret_cast<C_Profiler>(handle));
      }
    } catch (...) {
    }
  };

  m["profiler_start"] = [](void *handle) {
    try {
      insight_profiler_start(reinterpret_cast<C_Profiler>(handle));
    } catch (...) {
    }
  };

  m["profiler_stop"] = [](void *handle) {
    try {
      insight_profiler_stop(reinterpret_cast<C_Profiler>(handle));
    } catch (...) {
    }
  };

  m["profiler_reset"] = [](void *handle) {
    try {
      insight_profiler_reset(reinterpret_cast<C_Profiler>(handle));
    } catch (...) {
    }
  };

  m["profiler_begin_event"] = [](void *handle, const char *name) {
    try {
      insight_profiler_begin_event(reinterpret_cast<C_Profiler>(handle), name);
    } catch (...) {
    }
  };

  m["profiler_end_event"] = [](void *handle) {
    try {
      insight_profiler_end_event(reinterpret_cast<C_Profiler>(handle));
    } catch (...) {
    }
  };

  m["profiler_get_events"] = [](void *handle,
                                sol::this_state ts) -> sol::table {
    try {
      if (!handle) {
        sol::state_view lv(ts);
        return lv.create_table(0);
      }
      C_ProfilerEvent *events = nullptr;
      size_t count = 0;
      C_Status status = insight_profiler_get_events(
          reinterpret_cast<C_Profiler>(handle), &events, &count);
      sol::state_view lv(ts);
      sol::table result = lv.create_table(static_cast<int>(count));
      if (status == C_SUCCESS && events != nullptr) {
        for (size_t i = 0; i < count; i++) {
          sol::table ev = lv.create_table();
          ev["name"] = events[i].name ? std::string(events[i].name) : "";
          ev["calls"] = static_cast<int>(events[i].calls);
          ev["total_ms"] = events[i].total_ms;
          ev["min_ms"] = events[i].min_ms;
          ev["max_ms"] = events[i].max_ms;
          result[i + 1] = ev;
        }
      }
      return result;
    } catch (...) {
      sol::state_view lv(ts);
      return lv.create_table(0);
    }
  };

  // Push module table onto stack
  sol::stack::push(lua, m);
  return 1;
}
