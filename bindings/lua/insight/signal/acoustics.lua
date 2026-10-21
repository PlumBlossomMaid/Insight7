--- Acoustic scale functions.
-- Provides conversions between Hz, Mel, and Bark frequency scales.
-- @module insight.signal.acoustics

local native = require("_insight")
local sig = native.signal

local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

local from_table = native.from_table

--- Convert Mel scale to Hz.
-- Accepts a scalar number or an Array.
-- @param mel Frequency in Mel scale (number or Array).
-- @treturn number Frequency in Hz (scalar).
M.mel2hz = rule({ { name = "mel", type = argrule.any, opt = true } })(function(mel)
  local arr = (type(mel) == "number") and from_table({ mel }) or mel
  local r = sig.mel2hz(arr)
  if type(mel) == "number" then
    return r:item(0)
  end
  return r
end)

--- Convert Hz to Mel scale.
-- Accepts a scalar number or an Array.
-- @param hz Frequency in Hz (number or Array).
-- @treturn number Frequency in Mel scale (scalar).
M.hz2mel = rule({ { name = "hz", type = argrule.any, opt = true } })(function(hz)
  local arr = (type(hz) == "number") and from_table({ hz }) or hz
  local r = sig.hz2mel(arr)
  if type(hz) == "number" then
    return r:item(0)
  end
  return r
end)

--- Compute Mel-spaced frequencies.
-- @int num Number of frequencies to generate.
-- @number[opt=0] fmin Minimum frequency in Hz.
-- @number[opt] fmax Maximum frequency in Hz (default: fs/2).
-- @bool[opt=false] htk Use HTK formula.
-- @treturn Array Mel-spaced frequencies.
M.mel_frequencies = rule({
  { name = "num", type = argrule.any, opt = true },
  { name = "fmin", type = argrule.any, opt = true },
  { name = "fmax", type = argrule.any, opt = true },
  { name = "htk", type = argrule.any, opt = true },
})(function(num, fmin, fmax, htk)
  return sig.mel_frequencies(num, fmin or 0, fmax or 8000, htk or false)
end)

--- Convert Hz to Bark scale.
-- Accepts a scalar number or an Array.
-- @param hz Frequency in Hz (number or Array).
-- @treturn number Frequency in Bark scale (scalar).
M.hz2bark = rule({ { name = "hz", type = argrule.any, opt = true } })(function(hz)
  local arr = (type(hz) == "number") and from_table({ hz }) or hz
  local r = sig.hz2bark(arr)
  if type(hz) == "number" then
    return r:item(0)
  end
  return r
end)

--- Convert Bark scale to Hz.
-- Accepts a scalar number or an Array.
-- @param bark Frequency in Bark scale (number or Array).
-- @treturn number Frequency in Hz (scalar).
M.bark2hz = rule({ { name = "bark", type = argrule.any, opt = true } })(function(bark)
  local arr = (type(bark) == "number") and from_table({ bark }) or bark
  local r = sig.bark2hz(arr)
  if type(bark) == "number" then
    return r:item(0)
  end
  return r
end)

return M
