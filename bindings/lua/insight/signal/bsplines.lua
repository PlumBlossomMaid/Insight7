--- B-spline basis functions.
-- Provides Gaussian, cubic, and quadratic B-spline functions.
-- @module insight.signal.bsplines

local native = require("_insight")
local sig = native.signal

local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Return a Gaussian B-spline basis function.
-- @int n Order of the spline.
-- @tparam Array x Evaluation points.
-- @treturn Array B-spline values.
M.gauss_spline = rule({ { name = "n", type = argrule.any, opt = true }, { name = "x", type = argrule.any, opt = true } })(
  function(n, x)
    return sig.gauss_spline(x, n)
  end
)

--- Return a cubic B-spline.
-- @number x Evaluation points.
-- @treturn Array Cubic B-spline values.
M.cubic = rule({ { name = "x", type = argrule.any, opt = true } })(function(x)
  return sig.cubic(x)
end)

--- Return a quadratic B-spline.
-- @number x Evaluation points.
-- @treturn Array Quadratic B-spline values.
M.quadratic = rule({ { name = "x", type = argrule.any, opt = true } })(function(x)
  return sig.quadratic(x)
end)

return M
