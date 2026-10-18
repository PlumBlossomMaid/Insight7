--- Type casting operations.
--
-- Provides functions for converting arrays between different data types.
--
-- @module insight.cast

local native = require("_insight")
local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Cast an array to a different data type.
-- @tparam Array x Input array.
-- @tparam DType dtype Target data type (e.g. ins.float64).
-- @treturn Array Array with the new data type.
M.cast = rule { { name = "x", type = argrule.any, opt = true }, { name = "dtype", type = argrule.any, opt = true } }(function(x, dtype)
  return native.cast(x, dtype)
end)

return M
