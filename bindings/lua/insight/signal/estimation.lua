--- State estimation functions.
-- Provides the KalmanFilter class for linear dynamic system estimation.
-- @module insight.signal.estimation

local native = require("_insight")
local sig = native.signal

local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Create a KalmanFilter for linear dynamic system state estimation.
-- @int dim_x Number of state variables.
-- @int dim_z Number of measurement inputs.
-- @int[opt=0] dim_u Number of control inputs.
-- @int[opt=1] points Number of filter points.
-- @tparam[opt=ins.float64] DType dtype Data type.
-- @treturn KalmanFilter The constructed filter.
M.KalmanFilter = rule({
  { name = "dim_x", type = argrule.any, opt = true },
  { name = "dim_z", type = argrule.any, opt = true },
  { name = "dim_u", type = argrule.any, opt = true },
  { name = "points", type = argrule.any, opt = true },
  { name = "dtype", type = argrule.any, opt = true },
})(function(dim_x, dim_z, dim_u, points, dtype)
  return sig.KalmanFilter_new(dim_x, dim_z, dim_u, points, dtype)
end)

return M
