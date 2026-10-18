--- Signal I/O functions.
-- Provides binary file read/write, pack/unpack, and SigMF metadata support.
-- @module insight.signal.io

local native = require("_insight")
local sig = native.signal

local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Read a binary file into an array.
-- @string filepath Path to the binary file.
-- @string dtype Data type string (e.g., 'float32', 'int16').
-- @treturn Array Data read from the file.
M.read_bin = rule { { name = "filepath", type = argrule.any, opt = true }, { name = "dtype", type = argrule.any, opt = true } }(function(filepath, dtype)
  return sig.read_bin(filepath, dtype)
end)

--- Write an array to a binary file.
-- @string filepath Path to the output binary file.
-- @tparam Array x Input array.
M.write_bin = rule { { name = "filepath", type = argrule.any, opt = true }, { name = "x", type = argrule.any, opt = true } }(function(filepath, x)
  return sig.write_bin(filepath, x)
end)

--- Unpack binary data from raw bytes.
-- @tparam Array raw Raw byte array.
-- @int dtype Data type code (e.g., ins.float32, ins.int16).
-- @string[opt="L"] endianness Endianness: "L" (little) or "B" (big).
-- @treturn Array Unpacked data array.
M.unpack_bin = rule { { name = "raw", type = argrule.any, opt = true }, { name = "dtype", type = argrule.any, opt = true }, { name = "endianness", type = argrule.any, opt = true } }(function(raw, dtype, endianness)
  return sig.unpack_bin(raw, dtype, endianness)
end)

--- Pack array data into raw bytes.
-- @tparam Array x Input array.
-- @string dtype Target data type string.
-- @treturn Array Packed byte array.
M.pack_bin = rule { { name = "x", type = argrule.any, opt = true }, { name = "dtype", type = argrule.any, opt = true } }(function(x, dtype)
  return sig.pack_bin(x, dtype)
end)

--- Read a SigMF metadata file and its associated data.
-- @string metafile Path to the SigMF .json metadata file.
-- @treturn Array Data array read from the associated binary file.
M.read_sigmf = rule { { name = "metafile", type = argrule.any, opt = true } }(function(metafile)
  return sig.read_sigmf(metafile)
end)

--- Write data and metadata in SigMF format.
-- @tparam Array x Input data array.
-- @string metafile Path to the SigMF .sigmf-data output file.
-- @bool[opt=true] append Append mode.
M.write_sigmf = rule { { name = "x", type = argrule.any, opt = true }, { name = "metafile", type = argrule.any, opt = true }, { name = "append", type = argrule.any, opt = true } }(function(x, metafile, append)
  return sig.write_sigmf(metafile, x, append)
end)

return M
