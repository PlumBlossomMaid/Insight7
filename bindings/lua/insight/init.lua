--- Insight7 Lua bindings — Core module.
-- Provides array creation, arithmetic, reduction, linear algebra,
-- FFT, signal processing, and more.
--
-- @module insight
-- @author PlumBlossomMaid
-- @version 1.0.0
-- @license MIT
--
-- @usage
--   local ins = require("insight")
--   ins.init({"cpu"})
--   local a = ins.zeros({2, 3}, ins.float32)
--   local b = ins.ones({2, 3}, ins.float32)
--   local c = a + b
--   local s = ins.sum(c)
--   local m = ins.matmul(a, ins.transpose(b))
--
-- DType Shortcuts (Paddle-style):
--
--     ins.float32, ins.float64, ins.int32, ins.int64, ins.bool, ...
--
-- Place Constructors:
--
--     ins.CPUPlace()       -- CPU device
--     ins.GPUPlace(0)      -- GPU device 0

-- Try to load the native C++ module (_insight.so)
-- Pre-load ALL backend .so/.dll files (CPU + GPU) from various possible locations
local function _find_and_load_backends()
  local dirs = {}
  -- 1. Parent of this script's directory (dev/source layout)
  local _this_dir = debug.getinfo(1, "S").source:match("@?(.*/)")
  if not _this_dir then
    -- Windows: try backslash pattern
    _this_dir = debug.getinfo(1, "S").source:match("@?(.*[/\\])")
  end
  if _this_dir then
    local _parent = _this_dir:match("(.*[/\\])[^/\\]+[/\\]$") or _this_dir
    table.insert(dirs, _parent)
  end
  -- 2. Same directory as _insight.so (package.cpath)
  for path in package.cpath:gmatch("[^;]+") do
    local dir = path:gsub("%?.*$", "")
    if dir ~= "" then
      table.insert(dirs, dir)
    end
  end
  -- 3. Resolve _insight.so symlink → actual lib directory (Linux/macOS only)
  local is_windows = package.config:sub(1, 1) == "\\"
  if not is_windows then
    for path in package.cpath:gmatch("[^;]+") do
      local candidate = path:gsub("%?", "_insight")
      local f = io.open(candidate, "r")
      if f then
        f:close()
        local pipe = io.popen('readlink -f "' .. candidate .. '" 2>/dev/null')
        if pipe then
          local target = pipe:read("*l")
          pipe:close()
          if target and target ~= "" and target ~= candidate then
            local dir = target:match("(.*/)") or ""
            if dir ~= "" then
              table.insert(dirs, dir)
            end
          end
        end
      end
    end
  end
  -- 4. LD_LIBRARY_PATH (Linux/macOS) or PATH (Windows) directories
  local _env_path
  local _sep
  if is_windows then
    _env_path = os.getenv("PATH") or ""
    _sep = ";"
  else
    _env_path = os.getenv("LD_LIBRARY_PATH") or ""
    _sep = ":"
  end
  for dir in _env_path:gmatch("[^" .. _sep .. "]+") do
    if dir ~= "" then
      table.insert(dirs, dir)
    end
  end

  -- Scan all candidate dirs for backend .so/.dll and pre-load them
  -- Uses package.loadlib (standard Lua 5.1+) with * prefix for RTLD_GLOBAL
  -- Falls back to ffi.C.dlopen/LoadLibraryA (LuaJIT) if available
  local ok_ffi, ffi_lib = pcall(require, "ffi")
  local use_ffi = ok_ffi and ffi_lib ~= nil
  if use_ffi then
    if is_windows then
      pcall(
        ffi_lib.cdef,
        [[
        void *LoadLibraryA(const char *filename);
        void *GetProcAddress(void *module, const char *name);
      ]]
      )
    else
      pcall(
        ffi_lib.cdef,
        [[
        void *dlopen(const char *filename, int flag);
      ]]
      )
    end
  end

  local seen = {}
  for _, dir in ipairs(dirs) do
    -- Cross-platform glob: use platform-appropriate command
    local cmd
    if is_windows then
      cmd = 'dir /b "' .. dir .. '\\libinsight_*_backend.dll" "' .. dir .. '\\insight_*_backend.dll" 2>nul'
    else
      cmd = 'ls "' .. dir .. '"/libinsight_*_backend.so "' .. dir .. '"/insight_*_backend.dll 2>/dev/null'
    end
    local pipe = io.popen(cmd)
    if pipe then
      for line in pipe:lines() do
        -- On Windows, dir /b returns bare names; prepend directory
        if is_windows and not line:match("[/\\]") then
          line = dir .. "\\" .. line
        end
        if not seen[line] then
          seen[line] = true
          local f2 = io.open(line, "r")
          if f2 then
            f2:close()
            if use_ffi then
              if is_windows then
                ffi_lib.C.LoadLibraryA(line)
              else
                -- LuaJIT: use ffi for reliable RTLD_GLOBAL
                ffi_lib.C.dlopen(line, 258) -- RTLD_NOW | RTLD_GLOBAL
              end
            else
              -- Lua 5.3+: package.loadlib with * prefix loads with RTLD_GLOBAL
              -- even if function doesn't exist, library stays loaded
              pcall(package.loadlib, line, "*luaopen_dummy")
            end
          end
        end
      end
      pipe:close()
    end
  end
end

_find_and_load_backends()

local ok, native = pcall(require, "_insight")
if not ok then
  error(
    "Failed to load Insight native module: "
      .. tostring(native)
      .. "\nMake sure _insight.so is in your LUA_CPATH or LD_LIBRARY_PATH."
  )
end

-- Try to load Penlight for enhanced utilities
local has_penlight, utils = pcall(require, "pl.utils")
local has_tablex, tablex = pcall(require, "pl.tablex")
local has_stringx, stringx = pcall(require, "pl.stringx")

--- @class insight
local M = {}

-- Lua 5.1 / LuaJIT have no table.unpack (5.2+); keep wrap() portable.
local _unpack = table.unpack or unpack

-- Save native reference for internal use
M._native = native

-- Set up __newindex on Array metatable for assignment syntax
-- a["spec"] = value and a[int] = value
do
  local tmp = native.zeros({ 1 }, native.float64, native.CPUPlace())
  local mt = getmetatable(tmp)
  if mt then
    local orig_newindex = mt.__newindex
    mt.__newindex = function(self, key, value)
      if type(key) == "number" then
        if key == 0 then
          error("Lua arrays are 1-based: index 0 is invalid")
        end
        -- self[key] goes through the __index integer overload, which
        -- converts 1-based → 0-based and returns a view for fill_/copy_from_
        local view = self[key]
        if type(value) == "number" then
          view:fill_(value)
        else
          view:copy_from_(value)
        end
      elseif type(key) == "string" then
        local view = self[key] -- uses __index for string slicing
        if type(value) == "number" then
          view:fill_(value)
        else
          view:copy_from_(value)
        end
      elseif orig_newindex then
        orig_newindex(self, key, value)
      end
    end
  end
end

-- Re-export all native functions and values
for k, v in pairs(native) do
  M[k] = v
end

-- Lua 5.3 strictly distinguishes int/float. sol2 rejects 1e7 (float) for
-- int64_t params. Provide a helper to auto-convert in wrapper functions.
function M._toint(v)
  if type(v) == "number" then
    return math.floor(v)
  end
  return v
end

--- Auto-convert float table values to integers (for shape tables).
function M._toint_table(t)
  if type(t) ~= "table" then
    return t
  end
  local out = {}
  for i, v in ipairs(t) do
    out[i] = math.floor(v)
  end
  return out
end

--- Create an Array from a nested Lua table.
-- Strict validation: only number, boolean, and table allowed.
-- nil, string, function, userdata → error. Max 10 dimensions.
-- @tparam table t Nested table of numbers/booleans
-- @treturn Array The created array (float64)
function M.from_table(t)
  local function validate(obj, path, depth)
    if depth > 10 then
      error("Table nesting exceeds 10 dimensions at " .. path, 3)
    end
    local tp = type(obj)
    if tp == "nil" then
      error("nil found at " .. path, 3)
    elseif tp == "string" then
      error("string found at " .. path, 3)
    elseif tp == "function" then
      error("function found at " .. path, 3)
    elseif tp == "userdata" or tp == "lightuserdata" then
      error("userdata found at " .. path, 3)
    elseif tp == "thread" then
      error("thread found at " .. path, 3)
    elseif tp ~= "number" and tp ~= "boolean" and tp ~= "table" then
      error("unexpected type '" .. tp .. "' at " .. path, 3)
    end
  end

  local function infer_shape(tbl, shape, depth, path)
    validate(tbl, path, depth)
    if type(tbl) ~= "table" then
      return
    end
    local len = #tbl
    if depth > #shape then
      shape[#shape + 1] = len
    elseif shape[depth] ~= len then
      error(
        "Ragged table at "
          .. path
          .. ": dimension "
          .. depth
          .. " inconsistent ("
          .. shape[depth]
          .. " vs "
          .. len
          .. ")",
        3
      )
    end
    for i = 1, len do
      validate(tbl[i], path .. "[" .. i .. "]", depth)
      if type(tbl[i]) == "table" then
        infer_shape(tbl[i], shape, depth + 1, path .. "[" .. i .. "]")
      end
    end
  end

  if type(t) ~= "table" then
    error("from_table expects a table, got " .. type(t), 2)
  end
  if #t == 0 then
    return M.zeros({ 0 }, M.float64)
  end
  infer_shape(t, {}, 1, "root")
  return M._native.from_table(t)
end

--- Get the shape of an array as a Lua table.
-- @tparam Array arr The input array
-- @treturn table Shape as {dim1, dim2, ...}
function M.shape_table(arr)
  local s = arr.shape
  local t = {}
  for i = 1, #s do
    t[i] = s[i]
  end
  return t
end

--- Pretty-print an array (enhanced with Penlight if available).
-- @tparam Array arr The input array
-- @treturn string Human-readable string representation
function M.pretty(arr)
  if has_stringx then
    return stringx.format(
      "<Array shape=(%s) dtype=%s place=%s>",
      table.concat(M.shape_table(arr), ", "),
      tostring(arr.dtype),
      arr.place and tostring(arr.place) or "cpu"
    )
  else
    return tostring(arr)
  end
end

--- Convenience alias: get number of dimensions.
-- @tparam Array arr The input array
-- @treturn int Number of dimensions
function M.ndim(arr)
  return arr.ndim
end

--- Convenience alias: get total number of elements.
-- @tparam Array arr The input array
-- @treturn int Total element count
function M.size(arr)
  return arr.numel
end

-- ============================================================================
-- Argument parsing helpers
-- ============================================================================

--- Parse arguments supporting both positional and named-table calling styles.
--
-- Usage:
--   local args = ins.args({"x", "y", "axis"}, ...)
--
-- Supports:
--   func(x, y, 1)           -- positional
--   func{x, y, axis=1}      -- mixed positional + named
--   func{x=x, y=y, axis=1}  -- fully named
--
-- @tparam table names Ordered list of argument names.
-- @param ... The actual call arguments.
-- @treturn table A table mapping each name to its value.
function M.args(names, ...)
  local nargs = select("#", ...)
  if nargs == 1 and type(select(1, ...)) == "table" then
    local t = select(1, ...)
    -- Check if this is a positional array table (has numeric keys only)
    -- or a named argument table
    local has_names = false
    for k, _ in pairs(t) do
      if type(k) ~= "number" then
        has_names = true
        break
      end
    end
    if has_names then
      local result = {}
      for i, name in ipairs(names) do
        result[name] = t[name]
        if result[name] == nil and t[i] ~= nil then
          result[name] = t[i]
        end
      end
      return result
    end
  end
  -- Positional arguments
  local result = {}
  for i, name in ipairs(names) do
    result[name] = select(i, ...)
  end
  return result
end

--- Create a dual-mode wrapper function that supports positional and table args.
--
-- @tparam table arg_names Ordered argument names.
-- @tparam function fn The underlying function to call.
-- @treturn function A wrapper supporting both calling conventions.
--
-- Example:
--   local my_add = ins.wrap({"a", "b"}, function(a, b) return native.add(a, b) end)
--   my_add(x, y)       -- positional
--   my_add{a=x, b=y}   -- named table
function M.wrap(arg_names, fn)
  return function(...)
    if select("#", ...) == 1 and type(select(1, ...)) == "table" then
      local t = select(1, ...)
      local has_names = false
      for k, _ in pairs(t) do
        if type(k) ~= "number" then
          has_names = true
          break
        end
      end
      if has_names then
        local pos = {}
        for i, name in ipairs(arg_names) do
          pos[i] = t[name]
          if pos[i] == nil then
            pos[i] = t[i]
          end
        end
        return fn(_unpack(pos, 1, #arg_names))
      end
    end
    return fn(...)
  end
end

-- ============================================================================
-- Load sub-modules
-- ============================================================================

-- Signal submodule
local ok_signal, signal_mod = pcall(require, "insight.signal")
if ok_signal then
  M.signal = signal_mod
end

-- Non-signal sub-modules (organized by C++ source structure)
local submodules = {
  "elementwise",
  "unary",
  "reduction",
  "manipulation",
  "linalg",
  "fft",
  "complex",
  "random",
  "cast",
  "indexing",
}

for _, mod_name in ipairs(submodules) do
  local ok_mod, mod = pcall(require, "insight." .. mod_name)
  if ok_mod then
    for k, v in pairs(mod) do
      if M[k] == nil then
        M[k] = v
      end
    end
  end
end

-- ============================================================================
-- Type documentation
-- ============================================================================

--- Array type documentation.
-- Arrays are the core data container in Insight7.
--
-- **Properties:**
--
--     arr.shape         — Shape object (table of dimension sizes)
--     arr.dtype         — DType enum value
--     arr.place         — Place object (CPUPlace or GPUPlace)
--     arr.numel         — Total number of elements
--     arr.nbytes        — Total bytes of data
--     arr.ndim          — Number of dimensions
--     arr.is_contiguous — Whether data is contiguous in memory
--     arr.defined       — Whether the array is valid
--
-- **Methods:**
--
--     arr:contiguous()      — Return contiguous copy if needed
--     arr:reshape(shape)    — Return view with new shape
--     arr:transpose()       — Return transposed view
--     arr:squeeze(axis)     — Remove size-1 dimensions
--     arr:unsqueeze(axis)   — Insert size-1 dimension
--     arr:to(place)         — Transfer to device
--     arr:copy()            — Deep copy
--     arr:get(index)        — Get scalar value at index
--     arr:item()            — Get scalar value (0-d array only)
--
-- **Indexing (1-based):**
--
--     arr["1:3, :"]         — String-based slicing
--     arr[1]                — Integer indexing
--
-- **Metamethods:**
--
--     arr + arr, arr - arr, arr * arr, arr / arr
--     -arr (negation), ~arr (bitwise NOT)
--     arr == arr, arr < arr, arr <= arr
-- @see Array
M._array_docs = "See Array usertype documentation above"

--- Place types for device placement.
-- @see CPUPlace
-- @see GPUPlace
-- @usage
--   local cpu = ins.CPUPlace()
--   local gpu = ins.GPUPlace(0)
--   local arr_gpu = arr:to(gpu)
--   local arr_cpu = arr_gpu:to(cpu)
M._place_docs = "CPUPlace() for CPU, GPUPlace(id) for GPU"

--- DType shortcuts for array creation and casting.
-- Available types:
--
--     ins.float32, ins.float64, ins.float16, ins.bfloat16,
--     ins.int8, ins.int16, ins.int32, ins.int64,
--     ins.uint8, ins.uint16, ins.uint32, ins.uint64,
--     ins.bool, ins.complex64, ins.complex128
--
-- @see DType
-- @usage
--   local a = ins.zeros({2, 3}, ins.float64)
--   local b = ins.cast(a, ins.int32)
M._dtype_docs = "See DType documentation above"

-- ========================================================================
-- Timer class (Penlight class)
-- ========================================================================

-- Try to load Penlight class system
local has_class, pl_class = pcall(require, "pl.class")

if has_class then
  -- Penlight-based Timer class
  local Timer = pl_class:new()

  function Timer:_init(device_type, device_id)
    if device_type == nil or device_id == nil then
      error("Timer: device_type and device_id are required. " .. "Usage: Timer(0, 0) for CPU, Timer(1, 0) for GPU:0")
    end
    if type(device_type) ~= "number" or type(device_id) ~= "number" then
      error("Timer: device_type and device_id must be numbers")
    end
    self._handle = M.timer_create(device_type, device_id)
    if self._handle == nil then
      error("Timer: failed to create timer (device backend not available)")
    end
    self._started = false
  end

  function Timer:start()
    M.timer_start(self._handle)
    self._started = true
  end

  function Timer:stop()
    M.timer_stop(self._handle)
    self._started = false
  end

  function Timer:elapsed()
    return M.timer_elapsed_ms(self._handle)
  end

  function Timer:reset()
    -- Destroy old handle and create new one
    -- (The timer handle stays valid, just reuse it)
    -- For CPU/GPU, reset doesn't need special handling
    self._started = false
  end

  function Timer:__gc()
    if self._handle then
      M.timer_destroy(self._handle)
      self._handle = nil
    end
  end

  -- Register the Timer class in the insight module
  M.Timer = Timer
else
  -- Fallback: table-based Timer without Penlight
  --- Create a new Timer.
  -- @function Timer
  -- @tparam number device_type 0 for CPU, 1 for GPU
  -- @tparam number device_id Device ID (0 for first device)
  -- @treturn table Timer object with start/stop/elapsed methods
  function M.Timer(device_type, device_id)
    if device_type == nil or device_id == nil then
      error("Timer: device_type and device_id are required")
    end
    local handle = M.timer_create(device_type, device_id)
    if handle == nil then
      error("Timer: failed to create timer (device backend not available)")
    end
    return {
      _handle = handle,
      _started = false,
      start = function(self)
        M.timer_start(self._handle)
        self._started = true
      end,
      stop = function(self)
        M.timer_stop(self._handle)
        self._started = false
      end,
      elapsed = function(self)
        return M.timer_elapsed_ms(self._handle)
      end,
      __gc = function(self)
        if self._handle then
          M.timer_destroy(self._handle)
          self._handle = nil
        end
      end,
    }
  end
end

-- ========================================================================
-- Profiler class
-- ========================================================================

if has_class then
  -- Penlight-based Profiler class
  local Profiler = pl_class:new()

  function Profiler:_init(device_kind, device_id)
    device_kind = device_kind or 0
    device_id = device_id or 0
    self._handle = M.profiler_create(device_kind, device_id)
    if self._handle == nil then
      error("Profiler: failed to create profiler (device backend not available)")
    end
  end

  function Profiler:start()
    M.profiler_start(self._handle)
  end

  function Profiler:stop()
    M.profiler_stop(self._handle)
  end

  function Profiler:reset()
    M.profiler_reset(self._handle)
  end

  function Profiler:begin_event(name)
    M.profiler_begin_event(self._handle, name)
  end

  function Profiler:end_event()
    M.profiler_end_event(self._handle)
  end

  function Profiler:get_events()
    return M.profiler_get_events(self._handle)
  end

  function Profiler:report()
    local events = self:get_events()
    if #events == 0 then
      print("  [Profiler] no events recorded")
      return
    end
    print("")
    print(string.format("  %-24s %7s %12s %10s %10s", "Event", "Calls", "Total(ms)", "Avg(ms)", "Max(ms)"))
    print("  " .. string.rep("─", 63))
    for _, ev in ipairs(events) do
      local avg = ev.total_ms / ev.calls
      print(string.format("  %-24s %7d %12.3f %10.4f %10.4f", ev.name, ev.calls, ev.total_ms, avg, ev.max_ms))
    end
    print("")
  end

  function Profiler:__gc()
    if self._handle then
      M.profiler_destroy(self._handle)
      self._handle = nil
    end
  end

  M.Profiler = Profiler
else
  -- Fallback: table-based Profiler without Penlight
  --- Create a new Profiler.
  -- @function Profiler
  -- @tparam[opt] number device_kind 0 for CPU (default), 1 for GPU
  -- @tparam[opt] number device_id Device ID (default 0)
  -- @treturn table Profiler object with start/stop/reset/begin_event/end_event/get_events/report methods
  function M.Profiler(device_kind, device_id)
    device_kind = device_kind or 0
    device_id = device_id or 0
    local handle = M.profiler_create(device_kind, device_id)
    if handle == nil then
      error("Profiler: failed to create profiler (device backend not available)")
    end
    return {
      _handle = handle,
      start = function(self)
        M.profiler_start(self._handle)
      end,
      stop = function(self)
        M.profiler_stop(self._handle)
      end,
      reset = function(self)
        M.profiler_reset(self._handle)
      end,
      begin_event = function(self, name)
        M.profiler_begin_event(self._handle, name)
      end,
      end_event = function(self)
        M.profiler_end_event(self._handle)
      end,
      get_events = function(self)
        return M.profiler_get_events(self._handle)
      end,
      report = function(self)
        local events = self:get_events()
        if #events == 0 then
          print("  [Profiler] no events recorded")
          return
        end
        print("")
        print(string.format("  %-24s %7s %12s %10s %10s", "Event", "Calls", "Total(ms)", "Avg(ms)", "Max(ms)"))
        print("  " .. string.rep("─", 63))
        for _, ev in ipairs(events) do
          local avg = ev.total_ms / ev.calls
          print(string.format("  %-24s %7d %12.3f %10.4f %10.4f", ev.name, ev.calls, ev.total_ms, avg, ev.max_ms))
        end
        print("")
      end,
      __gc = function(self)
        if self._handle then
          M.profiler_destroy(self._handle)
          self._handle = nil
        end
      end,
    }
  end
end

return M
