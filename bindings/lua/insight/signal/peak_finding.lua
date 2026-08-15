--- Peak finding functions.
-- Provides functions to find local extrema (maxima and minima)
-- in signal arrays.
-- @module insight.signal.peak_finding

local native = require("_insight")
local sig = native.signal

local M = {}

local _wrap = require("insight._wrap")

local function lua_axis_to_cpp(axis)
  if axis == nil then
    return 0
  end
  if axis == 0 then
    error("Lua axes are 1-based: axis 0 is invalid", 3)
  end
  return axis > 0 and axis - 1 or axis
end

--- Find indices of relative extrema.
-- @tparam Array x Input signal array.
-- @string comparator Comparison direction: "greater" for maxima, "less" for minima.
-- @int[opt=1] axis Axis along which to find extrema.
-- @int[opt=1] order Number of points on each side to compare.
-- @string[opt="clip"] mode Edge handling mode.
-- @treturn userdata Vector of index Arrays.
M.argrelextrema = _wrap({ "x", "comparator", "axis", "order", "mode" }, function(x, comparator, axis, order, mode)
  return sig.argrelextrema(x, comparator, lua_axis_to_cpp(axis), order or 1, mode or "clip")
end)

--- Find indices of local maxima.
-- @tparam Array x Input signal array.
-- @int[opt=1] axis Axis along which to find maxima.
-- @int[opt=1] order Number of points on each side to compare.
-- @string[opt="clip"] mode Edge handling mode.
-- @treturn userdata Vector of index Arrays.
M.argrelmax = _wrap({ "x", "axis", "order", "mode" }, function(x, axis, order, mode)
  return sig.argrelmax(x, lua_axis_to_cpp(axis), order or 1, mode or "clip")
end)

--- Find indices of local minima.
-- @tparam Array x Input signal array.
-- @int[opt=1] axis Axis along which to find minima.
-- @int[opt=1] order Number of points on each side to compare.
-- @string[opt="clip"] mode Edge handling mode.
-- @treturn userdata Vector of index Arrays.
M.argrelmin = _wrap({ "x", "axis", "order", "mode" }, function(x, axis, order, mode)
  return sig.argrelmin(x, lua_axis_to_cpp(axis), order or 1, mode or "clip")
end)

return M
