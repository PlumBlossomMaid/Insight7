--- Elementwise binary operations.
--
-- Provides arithmetic, comparison, and logical operations
-- that are applied element-wise to arrays.
--
-- @module insight.elementwise

local native = require("_insight")
local M = {}

-- Dual-mode wrapper: supports both positional and named-table arguments.
-- func(a, b)  or  func{a=x, b=y}  or  func{x, y}
local rule = require("argrule.rule")
local argrule = require("argrule")

--- Element-wise addition.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Element-wise sum.
M.add = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.add(a, b)
  end
)

--- Element-wise subtraction.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Element-wise difference.
M.sub = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.sub(a, b)
  end
)

--- Element-wise multiplication.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Element-wise product.
M.mul = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.mul(a, b)
  end
)

--- Element-wise division.
-- @tparam Array a First input array (numerator).
-- @tparam Array b Second input array (denominator).
-- @treturn Array Element-wise quotient.
M.div = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.div(a, b)
  end
)

--- Element-wise exponentiation.
-- @tparam Array a Base array.
-- @tparam Array b Exponent array.
-- @treturn Array Element-wise power.
M.pow = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.pow(a, b)
  end
)

--- Element-wise modulus.
-- @tparam Array a First input array.
-- @tparam Array b Second input array (divisor).
-- @treturn Array Element-wise remainder.
M.mod = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.mod(a, b)
  end
)

--- Element-wise maximum of two arrays.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Element-wise maximum.
M.maximum = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.maximum(a, b)
  end
)

--- Element-wise minimum of two arrays.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Element-wise minimum.
M.minimum = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.minimum(a, b)
  end
)

--- Element-wise equality comparison.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Boolean array of element-wise equality.
M.equal = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.equal(a, b)
  end
)

--- Element-wise inequality comparison.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Boolean array of element-wise inequality.
M.not_equal = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.not_equal(a, b)
  end
)

--- Element-wise greater-than comparison.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Boolean array where a > b.
M.greater = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.greater(a, b)
  end
)

--- Element-wise less-than comparison.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Boolean array where a < b.
M.less = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.less(a, b)
  end
)

--- Element-wise greater-or-equal comparison.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Boolean array where a >= b.
M.greater_equal = rule({
  { name = "a", type = argrule.any, opt = true },
  { name = "b", type = argrule.any, opt = true },
})(function(a, b)
  return native.greater_equal(a, b)
end)

--- Element-wise less-or-equal comparison.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Boolean array where a <= b.
M.less_equal = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.less_equal(a, b)
  end
)

--- Element-wise logical AND.
-- @tparam Array a First boolean input array.
-- @tparam Array b Second boolean input array.
-- @treturn Array Boolean array of element-wise AND.
M.logical_and = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.logical_and(a, b)
  end
)

--- Element-wise logical OR.
-- @tparam Array a First boolean input array.
-- @tparam Array b Second boolean input array.
-- @treturn Array Boolean array of element-wise OR.
M.logical_or = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.logical_or(a, b)
  end
)

--- Element-wise logical XOR.
-- @tparam Array a First boolean input array.
-- @tparam Array b Second boolean input array.
-- @treturn Array Boolean array of element-wise XOR.
M.logical_xor = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.logical_xor(a, b)
  end
)

--- Element-wise logical NOT.
-- @tparam Array x Input boolean array.
-- @treturn Array Boolean array of element-wise NOT.
M.logical_not = rule({ { name = "x", type = argrule.any, opt = true } })(function(x)
  return native.logical_not(x)
end)

return M
