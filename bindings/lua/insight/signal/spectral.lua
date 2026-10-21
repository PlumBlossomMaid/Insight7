--- Spectral analysis functions.
-- Provides Welch's method, periodogram, cross-spectral density,
-- coherence, spectrogram, STFT, and vector strength.
-- @module insight.signal.spectral

local native = require("_insight")
local sig = native.signal

local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Estimate power spectral density using Welch's method.
-- @tparam Array x Input signal array.
-- @number[opt=1.0] fs Sampling frequency.
-- @string[opt="hann"] window Window type.
-- @int[opt] nperseg Length of each segment.
-- @int[opt] noverlap Number of overlapping points.
-- @int[opt] nfft FFT length.
-- @string[opt="density"] scaling 'density' or 'spectrum'.
-- @treturn table Table with fields 'f' (frequencies) and 'Pxx' (PSD).
M.welch = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "fs", type = argrule.any, opt = true },
  { name = "window", type = argrule.any, opt = true },
  { name = "nperseg", type = argrule.any, opt = true },
  { name = "noverlap", type = argrule.any, opt = true },
  { name = "nfft", type = argrule.any, opt = true },
  { name = "scaling", type = argrule.any, opt = true },
})(function(x, fs, window, nperseg, noverlap, nfft, scaling)
  return sig.welch(x, fs or 1.0, window or "hann", nperseg or 256, noverlap or 128, nfft or 256, scaling or "density")
end)

--- Estimate power spectral density using a periodogram.
-- @tparam Array x Input signal array.
-- @number[opt=1.0] fs Sampling frequency.
-- @string[opt="hann"] window Window type.
-- @int[opt] nfft FFT length.
-- @string[opt="density"] scaling 'density' or 'spectrum'.
-- @treturn table Table with fields 'f' (frequencies) and 'Pxx' (PSD).
M.periodogram = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "fs", type = argrule.any, opt = true },
  { name = "window", type = argrule.any, opt = true },
  { name = "nfft", type = argrule.any, opt = true },
  { name = "scaling", type = argrule.any, opt = true },
})(function(x, fs, window, nfft, scaling)
  return sig.periodogram(x, fs or 1.0, window or "hann", nfft or 256, scaling or "density")
end)

--- Estimate cross-spectral density using Welch's method.
-- @tparam Array x First input signal.
-- @tparam Array y Second input signal.
-- @number[opt=1.0] fs Sampling frequency.
-- @string[opt="hann"] window Window type.
-- @int[opt] nperseg Length of each segment.
-- @int[opt] noverlap Number of overlapping points.
-- @int[opt] nfft FFT length.
-- @treturn table Table with fields 'f' (frequencies) and 'Pxy' (CSD).
M.csd = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "y", type = argrule.any, opt = true },
  { name = "fs", type = argrule.any, opt = true },
  { name = "window", type = argrule.any, opt = true },
  { name = "nperseg", type = argrule.any, opt = true },
  { name = "noverlap", type = argrule.any, opt = true },
  { name = "nfft", type = argrule.any, opt = true },
})(function(x, y, fs, window, nperseg, noverlap, nfft)
  return sig.csd(x, y, fs or 1.0, window or "hann", nperseg or 256, noverlap or 128, nfft or 256)
end)

--- Estimate magnitude-squared coherence using Welch's method.
-- @tparam Array x First input signal.
-- @tparam Array y Second input signal.
-- @number[opt=1.0] fs Sampling frequency.
-- @string[opt="hann"] window Window type.
-- @int[opt] nperseg Length of each segment.
-- @int[opt] noverlap Number of overlapping points.
-- @int[opt] nfft FFT length.
-- @treturn table Table with fields 'f' (frequencies) and 'Cxy' (coherence).
M.coherence = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "y", type = argrule.any, opt = true },
  { name = "fs", type = argrule.any, opt = true },
  { name = "window", type = argrule.any, opt = true },
  { name = "nperseg", type = argrule.any, opt = true },
  { name = "noverlap", type = argrule.any, opt = true },
  { name = "nfft", type = argrule.any, opt = true },
})(function(x, y, fs, window, nperseg, noverlap, nfft)
  return sig.coherence(x, y, fs or 1.0, window or "hann", nperseg or 256, noverlap or 128, nfft or 256)
end)

--- Compute a spectrogram (SSTFT power spectral density).
-- @tparam Array x Input signal array.
-- @number[opt=1.0] fs Sampling frequency.
-- @string[opt="hann"] window Window type.
-- @int[opt] nperseg Length of each segment.
-- @int[opt] noverlap Number of overlapping points.
-- @int[opt] nfft FFT length.
-- @treturn table Table with fields 'f' (frequencies), 't' (times), 'Sxx' (spectrogram).
M.spectrogram = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "fs", type = argrule.any, opt = true },
  { name = "window", type = argrule.any, opt = true },
  { name = "nperseg", type = argrule.any, opt = true },
  { name = "noverlap", type = argrule.any, opt = true },
  { name = "nfft", type = argrule.any, opt = true },
})(function(x, fs, window, nperseg, noverlap, nfft)
  return sig.spectrogram(x, fs or 1.0, window or "hann", nperseg or 256, noverlap or 128, nfft or 256)
end)

--- Compute the Short-Time Fourier Transform (STFT).
-- @tparam Array x Input signal array.
-- @number[opt=1.0] fs Sampling frequency.
-- @string[opt="hann"] window Window type.
-- @int[opt] nperseg Length of each segment.
-- @int[opt] noverlap Number of overlapping points.
-- @int[opt] nfft FFT length.
-- @treturn table Table with fields 'f' (frequencies), 't' (times), 'Zxx' (STFT).
M.stft = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "fs", type = argrule.any, opt = true },
  { name = "window", type = argrule.any, opt = true },
  { name = "nperseg", type = argrule.any, opt = true },
  { name = "noverlap", type = argrule.any, opt = true },
  { name = "nfft", type = argrule.any, opt = true },
})(function(x, fs, window, nperseg, noverlap, nfft)
  return sig.stft(x, fs or 1.0, window or "hann", nperseg or 256, noverlap or 128, nfft or 256)
end)

--- Compute the vector strength (phase coherence).
-- @tparam Array signal Input signal array.
-- @number freq Frequency in Hz.
-- @number fs Sampling frequency in Hz.
-- @treturn table Table with fields 'strength' and 'angle'.
M.vectorstrength = rule({
  { name = "signal", type = argrule.any, opt = true },
  { name = "freq", type = argrule.any, opt = true },
  { name = "fs", type = argrule.any, opt = true },
})(function(signal, freq, fs)
  return sig.vectorstrength(signal, freq, fs)
end)

--- Compute the Lomb-Scargle periodogram.
-- @array x Sample times (1D).
-- @array y Sample values (1D, same length as x).
-- @array freqs Frequencies to evaluate (1D, in Hz).
-- @treturn Array Power spectral density at each frequency.
M.lombscargle = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "y", type = argrule.any, opt = true },
  { name = "freqs", type = argrule.any, opt = true },
})(function(x, y, freqs)
  return sig.lombscargle(x, y, freqs)
end)

return M
