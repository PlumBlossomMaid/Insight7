--- Wavelet functions.
-- Provides Morlet, Ricker (Mexican hat), and Morlet2 wavelet functions.
-- @module insight.signal.wavelets

local native = require("_insight")
local sig = native.signal

local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Compute a Morlet wavelet.
-- @int M Length of the wavelet (number of points).
-- @number[opt=5.0] w Frequency parameter (omega0).
-- @number[opt=1.0] s Scaling factor.
-- @bool[opt=true] complete Whether to use the complete version.
-- @treturn Array Complex Morlet wavelet.
M.morlet = rule { { name = "M", type = argrule.any, opt = true }, { name = "w", type = argrule.any, opt = true }, { name = "s", type = argrule.any, opt = true }, { name = "complete", type = argrule.any, opt = true } }(function(M_val, w, s, complete)
  return sig.morlet(M_val, w, s, complete)
end)

--- Compute a Ricker (Mexican hat) wavelet.
-- @int points Number of points.
-- @number a Width parameter.
-- @treturn Array Ricker wavelet.
M.ricker = rule { { name = "points", type = argrule.any, opt = true }, { name = "a", type = argrule.any, opt = true } }(function(points, a)
  return sig.ricker(points, a)
end)

--- Compute a Morlet2 wavelet.
-- @int M Length of the wavelet (number of points).
-- @number s Scale parameter.
-- @number[opt=5.0] w Frequency parameter (omega0).
-- @treturn Array Complex Morlet2 wavelet.
M.morlet2 = rule { { name = "M", type = argrule.any, opt = true }, { name = "s", type = argrule.any, opt = true }, { name = "w", type = argrule.any, opt = true } }(function(M_val, s, w)
  return sig.morlet2(M_val, s, w)
end)

return M
