--- Convolution and correlation functions.
-- Provides FFT-based convolution, cross-correlation,
-- and 2-D convolution/correlation operations.
-- @module insight.signal.convolution

local native = require("_insight")
local sig = native.signal

local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Convolve two N-dimensional arrays using FFT.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @string[opt="full"] mode Convolution mode: 'full', 'valid', or 'same'.
-- @treturn Array Convolved array.
M.fftconvolve = rule({
  { name = "a", type = argrule.any, opt = true },
  { name = "b", type = argrule.any, opt = true },
  { name = "mode", type = argrule.any, opt = true },
})(function(a, b, mode)
  return sig.fftconvolve(a, b, mode or "full")
end)

--- Cross-correlate two N-dimensional arrays.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @string[opt="full"] mode Correlation mode: 'full', 'valid', or 'same'.
-- @treturn Array Cross-correlation result.
M.correlate = rule({
  { name = "a", type = argrule.any, opt = true },
  { name = "b", type = argrule.any, opt = true },
  { name = "mode", type = argrule.any, opt = true },
})(function(a, b, mode)
  return sig.correlate(a, b, mode or "full")
end)

--- 2-D convolution of two arrays.
-- @tparam Array a First input 2-D array.
-- @tparam Array b Second input 2-D array.
-- @string[opt="full"] mode Convolution mode: 'full', 'valid', or 'same'.
-- @string[opt="fill"] boundary Boundary handling: 'fill', 'wrap', 'symm'.
-- @number[opt=0] fillvalue Value for fill boundary.
-- @treturn Array 2-D convolved array.
M.convolve2d = rule({
  { name = "a", type = argrule.any, opt = true },
  { name = "b", type = argrule.any, opt = true },
  { name = "mode", type = argrule.any, opt = true },
  { name = "boundary", type = argrule.any, opt = true },
  { name = "fillvalue", type = argrule.any, opt = true },
})(function(a, b, mode, boundary, fillvalue)
  return sig.convolve2d(a, b, mode or "full", boundary or "fill", fillvalue or 0)
end)

--- 2-D cross-correlation of two arrays.
-- @tparam Array a First input 2-D array.
-- @tparam Array b Second input 2-D array.
-- @string[opt="full"] mode Correlation mode: 'full', 'valid', or 'same'.
-- @string[opt="fill"] boundary Boundary handling: 'fill', 'wrap', 'symm'.
-- @number[opt=0] fillvalue Value for fill boundary.
-- @treturn Array 2-D cross-correlation result.
M.correlate2d = rule({
  { name = "a", type = argrule.any, opt = true },
  { name = "b", type = argrule.any, opt = true },
  { name = "mode", type = argrule.any, opt = true },
  { name = "boundary", type = argrule.any, opt = true },
  { name = "fillvalue", type = argrule.any, opt = true },
})(function(a, b, mode, boundary, fillvalue)
  return sig.correlate2d(a, b, mode or "full", boundary or "fill", fillvalue or 0)
end)

--- Choose the best convolution method (direct vs FFT).
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @string mode Convolution mode.
-- @treturn string Best method: 'direct' or 'fft'.
M.choose_conv_method = rule({
  { name = "a", type = argrule.any, opt = true },
  { name = "b", type = argrule.any, opt = true },
  { name = "mode", type = argrule.any, opt = true },
})(function(a, b, mode)
  return sig.choose_conv_method(a, b, mode)
end)

--- Compute the lag values for cross-correlation.
-- @int input_length Length of the first input.
-- @int kernel_length Length of the second input (kernel).
-- @string[opt="full"] mode Correlation mode.
-- @treturn Array Lag values.
M.correlation_lags = rule({
  { name = "input_length", type = argrule.any, opt = true },
  { name = "kernel_length", type = argrule.any, opt = true },
  { name = "mode", type = argrule.any, opt = true },
})(function(input_length, kernel_length, mode)
  return sig.correlation_lags(input_length, kernel_length, mode or "full")
end)

return M
