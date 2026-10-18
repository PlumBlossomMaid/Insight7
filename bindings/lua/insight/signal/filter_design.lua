--- Filter design functions.
-- Provides FIR filter design, Kaiser window parameter estimation,
-- and complex sorting utilities.
-- @module insight.signal.filter_design

local native = require("_insight")
local sig = native.signal

local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Compute the Kaiser beta parameter for a given attenuation.
-- @number attenuation Stopband attenuation in dB.
-- @number Kaiser beta parameter.
M.kaiser_beta = rule { { name = "attenuation", type = argrule.any, opt = true } }(function(attenuation)
  return sig.kaiser_beta(attenuation)
end)

--- Compute the Kaiser window attenuation for given parameters.
-- @int n Number of taps.
-- @number width Transition width (normalized).
-- @number Attenuation in dB.
M.kaiser_atten = rule { { name = "n", type = argrule.any, opt = true }, { name = "width", type = argrule.any, opt = true } }(function(n, width)
  return sig.kaiser_atten(n, width)
end)

--- Design a FIR filter using the window method.
-- @int numtaps Number of filter taps.
-- @number cutoff Cutoff frequency (normalized, 0 to 1).
-- @string[opt="hamming"] window Window function name: 'hamming', 'hann', 'blackman', etc.
-- @string[opt="lowpass"] pass_zero Filter type: 'lowpass', 'highpass', 'bandpass', 'bandstop', or boolean.
-- @bool[opt=true] scale If true, scale coefficients.
-- @treturn Array FIR filter coefficients.
M.firwin = rule { { name = "numtaps", type = argrule.any, opt = true }, { name = "cutoff", type = argrule.any, opt = true }, { name = "window", type = argrule.any, opt = true }, { name = "pass_zero", type = argrule.any, opt = true }, { name = "scale", type = argrule.any, opt = true } }(function(numtaps, cutoff, window, pass_zero, scale)
    return sig.firwin(numtaps, cutoff, window, pass_zero, scale)
  end
)

--- Design a FIR filter using frequency sampling.
-- @int numtaps Number of filter taps.
-- @tparam table freq Frequency breakpoints (normalized, 0 to 1).
-- @tparam table gain Desired gain at each frequency breakpoint.
-- @treturn Array FIR filter coefficients.
M.firwin2 = rule { { name = "numtaps", type = argrule.any, opt = true }, { name = "freq", type = argrule.any, opt = true }, { name = "gain", type = argrule.any, opt = true } }(function(numtaps, freq, gain)
  return sig.firwin2(numtaps, freq, gain)
end)

--- Sort complex numbers by real part, then imaginary part.
-- @tparam Array x Complex input array.
-- @treturn Array Sorted complex array.
M.cmplx_sort = rule { { name = "x", type = argrule.any, opt = true } }(function(x)
  return sig.cmplx_sort(x)
end)

return M
