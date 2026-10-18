--- Radar signal processing functions.
-- Provides pulse compression, Doppler processing, CFAR detection,
-- MVDR beamforming, and ambiguity function computation.
-- @module insight.signal.radar

local native = require("_insight")
local sig = native.signal

local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Perform pulse compression using matched filtering.
-- @tparam Array signal Input radar signal (1-D or 2-D).
-- @tparam Array reference Reference waveform (1-D).
-- @treturn Array Compressed pulse.
M.pulse_compression = rule { { name = "signal", type = argrule.any, opt = true }, { name = "reference", type = argrule.any, opt = true } }(function(signal, reference)
  return sig.pulse_compression(signal, reference)
end)

--- Perform pulse-Doppler processing.
-- @tparam Array data Input CPI data (2-D: pulses x range bins).
-- @string[opt=""] window Window name (e.g., "hann"). Empty for none.
-- @int[opt=0] nfft Number of FFT points for Doppler processing (0 = num_pulses).
-- @treturn Array Doppler-range map.
M.pulse_doppler = rule { { name = "data", type = argrule.any, opt = true }, { name = "window", type = argrule.any, opt = true }, { name = "nfft", type = argrule.any, opt = true } }(function(data, window, nfft)
  return sig.pulse_doppler(data, window or "", nfft or 0)
end)

--- Compute CFAR threshold alpha coefficient.
-- @number pfa Probability of false alarm.
-- @int N Number of reference cells.
-- @treturn number CFAR alpha threshold coefficient.
M.cfar_alpha = rule { { name = "pfa", type = argrule.any, opt = true }, { name = "N", type = argrule.any, opt = true } }(function(pfa, N)
  return sig.cfar_alpha(pfa, N)
end)

--- Apply Cell-Averaging CFAR detection.
-- @tparam Array data Input signal array.
-- @table guard_cells Guard cells per dimension, e.g. {2} or {2, 2}.
-- @table ref_cells Reference cells per dimension, e.g. {3} or {3, 3}.
-- @number[opt=1e-3] pfa Probability of false alarm.
-- @treturn table Pair of {threshold, detections} arrays.
M.ca_cfar = rule { { name = "data", type = argrule.any, opt = true }, { name = "guard_cells", type = argrule.any, opt = true }, { name = "ref_cells", type = argrule.any, opt = true }, { name = "pfa", type = argrule.any, opt = true } }(function(data, guard_cells, ref_cells, pfa)
  return sig.ca_cfar(data, guard_cells, ref_cells, pfa or 1e-3)
end)

--- Perform MVDR (Capon) beamforming.
-- @tparam Array x Input data matrix (sensors x samples).
-- @tparam Array steering Steering vector.
-- @treturn Array MVDR beamformer output.
M.mvdr = rule { { name = "x", type = argrule.any, opt = true }, { name = "steering", type = argrule.any, opt = true } }(function(x, steering)
  return sig.mvdr(x, steering)
end)

--- Compute the ambiguity function.
-- @tparam Array x Input signal (1-D, complex).
-- @number fs Sampling frequency in Hz.
-- @number prf Pulse repetition frequency in Hz.
-- @treturn Array Ambiguity function (2-D: delay x Doppler).
M.ambgfun = rule { { name = "x", type = argrule.any, opt = true }, { name = "fs", type = argrule.any, opt = true }, { name = "prf", type = argrule.any, opt = true } }(function(x, fs, prf)
  return sig.ambgfun(x, fs, prf)
end)

return M
