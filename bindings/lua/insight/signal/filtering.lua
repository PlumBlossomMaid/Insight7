--- Digital filtering functions.
-- Provides FIR/IIR filtering, Hilbert transform, detrending,
-- resampling, and Wiener filtering.
-- @module insight.signal.filtering

local native = require("_insight")
local sig = native.signal

local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Compute the analytic signal using the Hilbert transform.
-- @tparam Array x Input real-valued signal array.
-- @int[opt=-1] N Number of FFT points (-1 = signal length).
-- @treturn Array Complex-valued analytic signal.
M.hilbert = rule { { name = "x", type = argrule.any, opt = true }, { name = "N", type = argrule.any, opt = true } }(function(x, N)
  return sig.hilbert(x, N or -1)
end)

--- Compute the 2-D analytic signal using the Hilbert transform.
-- @tparam Array x Input 2-D real-valued signal array.
-- @int[opt=-1] N Number of FFT points per axis.
-- @treturn Array 2-D complex-valued analytic signal.
M.hilbert2 = rule { { name = "x", type = argrule.any, opt = true }, { name = "N", type = argrule.any, opt = true } }(function(x, N)
  return sig.hilbert2(x, N or -1)
end)

--- Remove a trend from the data along the given axis.
-- @tparam Array data Input array.
-- @int[opt=-1] axis Axis along which to detrend.
-- @string[opt="linear"] type Trend type: 'linear' or 'constant'.
-- @treturn Array Detrended array.
M.detrend = rule { { name = "data", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true }, { name = "type", type = argrule.any, opt = true } }(function(data, axis, type)
  return sig.detrend(data, axis or -1, type or "linear")
end)

--- Apply a Wiener filter to an N-dimensional array.
-- @tparam Array im Input array.
-- @tparam table[opt] mysize Size of the local neighbourhood.
-- @number[opt=-1] noise Noise power (-1 = estimated).
-- @treturn Array Wiener-filtered array.
M.wiener = rule { { name = "im", type = argrule.any, opt = true }, { name = "mysize", type = argrule.any, opt = true }, { name = "noise", type = argrule.any, opt = true } }(function(im, mysize, noise)
  local kwargs = {}
  if mysize then
    kwargs.mysize = mysize
  end
  if noise and noise ~= -1 then
    kwargs.noise = noise
  end
  return sig.wiener(im, kwargs)
end)

--- Apply an FIR filter to a signal.
-- @tparam Array b FIR filter coefficients (1-D array).
-- @tparam Array x Input signal array.
-- @int[opt=-1] axis Axis along which to filter.
-- @treturn Array Filtered signal.
M.firfilter = rule { { name = "b", type = argrule.any, opt = true }, { name = "x", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true } }(function(b, x, axis)
  return sig.firfilter(b, x, axis or -1)
end)

--- Apply an IIR or FIR filter using direct-form II transposed.
-- @tparam Array b Numerator coefficients.
-- @tparam Array a Denominator coefficients.
-- @tparam Array x Input signal array.
-- @int[opt=-1] axis Axis along which to filter.
-- @treturn Array Filtered signal.
M.lfilter = rule { { name = "b", type = argrule.any, opt = true }, { name = "a", type = argrule.any, opt = true }, { name = "x", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true } }(function(b, a, x, axis)
  return sig.lfilter(b, a, x, axis or -1)
end)

--- Compute the initial state for lfilter for step-response steady-state.
-- @tparam Array b Numerator coefficients.
-- @tparam Array a Denominator coefficients.
-- @treturn Array Initial state for lfilter.
M.lfilter_zi = rule { { name = "b", type = argrule.any, opt = true }, { name = "a", type = argrule.any, opt = true } }(function(b, a)
  return sig.lfilter_zi(b, a)
end)

--- Apply a forward-backward digital filter (zero-phase filtering).
-- @tparam Array b Numerator coefficients.
-- @tparam Array a Denominator coefficients.
-- @tparam Array x Input signal array.
-- @int[opt=-1] axis Axis along which to filter.
-- @treturn Array Zero-phase filtered signal.
M.filtfilt = rule { { name = "b", type = argrule.any, opt = true }, { name = "a", type = argrule.any, opt = true }, { name = "x", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true } }(function(b, a, x, axis)
  return sig.filtfilt(b, a, x, axis or -1)
end)

--- Downsample a signal after applying an anti-aliasing filter.
-- @tparam Array x Input signal array.
-- @int q Decimation factor.
-- @int[opt=-1] axis Axis along which to decimate.
-- @bool[opt=true] zero_phase Use forward-backward filtering.
-- @treturn Array Decimated signal.
M.decimate = rule { { name = "x", type = argrule.any, opt = true }, { name = "q", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true }, { name = "zero_phase", type = argrule.any, opt = true } }(function(x, q, axis, zero_phase)
  return sig.decimate(x, q, axis or -1, zero_phase == nil and true or zero_phase)
end)

--- Resample a signal to a given number of samples using FFT.
-- @tparam Array x Input signal array.
-- @int num Number of output samples.
-- @int[opt=-1] axis Axis along which to resample.
-- @treturn Array Resampled signal.
M.resample = rule { { name = "x", type = argrule.any, opt = true }, { name = "num", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true } }(function(x, num, axis)
  return sig.resample(x, num, axis or -1)
end)

--- Resample a signal using polyphase filtering.
-- @tparam Array x Input signal array.
-- @int up Upsampling factor.
-- @int down Downsampling factor.
-- @int[opt=-1] axis Axis along which to resample.
-- @treturn Array Resampled signal.
M.resample_poly = rule { { name = "x", type = argrule.any, opt = true }, { name = "up", type = argrule.any, opt = true }, { name = "down", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true } }(function(x, up, down, axis)
  return sig.resample_poly(x, up, down, axis or -1)
end)

--- Shift the frequency content of a signal.
-- @tparam Array x Input signal array.
-- @number freq Frequency shift in Hz.
-- @number fs Sampling frequency in Hz.
-- @treturn Array Frequency-shifted signal.
M.freq_shift = rule { { name = "x", type = argrule.any, opt = true }, { name = "freq", type = argrule.any, opt = true }, { name = "fs", type = argrule.any, opt = true } }(function(x, freq, fs)
  return sig.freq_shift(x, freq, fs)
end)

return M
