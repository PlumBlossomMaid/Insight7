--- Reduction operations.
--
-- Provides functions that reduce array dimensions by aggregating
-- values along one or more axes (sum, mean, max, min, etc.).
--
-- @module insight.reduction

local native = require("_insight")
local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Sum of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to sum. Defaults to all elements.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Sum result.
M.sum = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.sum(x, axis, keepdims or false)
end)

--- Mean of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute mean.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Mean result.
M.mean = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.mean(x, axis, keepdims or false)
end)

--- Maximum of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to find maximum.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Maximum result.
M.max = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.max(x, axis, keepdims or false)
end)

--- Minimum of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to find minimum.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Minimum result.
M.min = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.min(x, axis, keepdims or false)
end)

--- Product of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute product.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Product result.
M.prod = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.prod(x, axis, keepdims or false)
end)

--- Indices of maximum values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to find argmax.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Indices of maximum values.
M.argmax = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.argmax(x, axis, keepdims or false)
end)

--- Indices of minimum values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to find argmin.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Indices of minimum values.
M.argmin = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.argmin(x, axis, keepdims or false)
end)

--- Test whether any element is true.
-- @tparam Array x Input boolean array.
-- @int[opt] axis Axis along which to test.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Boolean result.
M.any = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.any(x, axis, keepdims or false)
end)

--- Test whether all elements are true.
-- @tparam Array x Input boolean array.
-- @int[opt] axis Axis along which to test.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Boolean result.
M.all = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.all(x, axis, keepdims or false)
end)

--- Variance of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute variance.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @int[opt=0] ddof Delta degrees of freedom.
-- @treturn Array Variance result.
M.var = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
  { name = "ddof", type = argrule.any, opt = true },
})(function(x, axis, keepdims, ddof)
  return native.var(x, axis, keepdims or false, ddof or 0)
end)

--- Standard deviation of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute std.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @int[opt=0] ddof Delta degrees of freedom.
-- @treturn Array Standard deviation result.
M.std = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
  { name = "ddof", type = argrule.any, opt = true },
})(function(x, axis, keepdims, ddof)
  return native.std(x, axis, keepdims or false, ddof or 0)
end)

--- Cumulative sum along an axis.
-- @tparam Array x Input array.
-- @int axis Axis along which to compute cumulative sum.
-- @treturn Array Cumulative sum result.
M.cumsum = rule({ { name = "x", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true } })(
  function(x, axis)
    return native.cumsum(x, axis)
  end
)

--- Cumulative product along an axis.
-- @tparam Array x Input array.
-- @int axis Axis along which to compute cumulative product.
-- @treturn Array Cumulative product result.
M.cumprod = rule({ { name = "x", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true } })(
  function(x, axis)
    return native.cumprod(x, axis)
  end
)

--- Cumulative maximum along an axis.
-- @tparam Array x Input array.
-- @int axis Axis along which to compute cumulative maximum.
-- @treturn Array Cumulative maximum result.
M.cummax = rule({ { name = "x", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true } })(
  function(x, axis)
    return native.cummax(x, axis)
  end
)

--- Cumulative minimum along an axis.
-- @tparam Array x Input array.
-- @int axis Axis along which to compute cumulative minimum.
-- @treturn Array Cumulative minimum result.
M.cummin = rule({ { name = "x", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true } })(
  function(x, axis)
    return native.cummin(x, axis)
  end
)

--- Standard error of the mean.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute SEM.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @int[opt=0] ddof Delta degrees of freedom.
-- @treturn Array Standard error of the mean.
M.sem = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
  { name = "ddof", type = argrule.any, opt = true },
})(function(x, axis, keepdims, ddof)
  return native.sem(x, axis, keepdims or false, ddof or 0)
end)

--- Count non-zero elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to count.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Count of non-zero elements.
M.count_nonzero = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.count_nonzero(x, axis, keepdims or false)
end)

--- Median of array elements.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute median.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Median result.
M.median = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.median(x, axis, keepdims or false)
end)

--- Compute the q-th quantile.
-- @tparam Array x Input array.
-- @number q Quantile to compute, between 0 and 1.
-- @int[opt] axis Axis along which to compute.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Quantile result.
M.quantile = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "q", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, q, axis, keepdims)
  return native.quantile(x, q, axis, keepdims or false)
end)

--- Compute the q-th percentile.
-- @tparam Array x Input array.
-- @number q Percentile to compute, between 0 and 100.
-- @int[opt] axis Axis along which to compute.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array Percentile result.
M.percentile = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "q", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, q, axis, keepdims)
  return native.percentile(x, q, axis, keepdims or false)
end)

--- Sum of array elements, ignoring NaN values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to sum.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array NaN-safe sum result.
M.nansum = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.nansum(x, axis, keepdims or false)
end)

--- Mean of array elements, ignoring NaN values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute mean.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array NaN-safe mean result.
M.nanmean = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.nanmean(x, axis, keepdims or false)
end)

--- Maximum of array elements, ignoring NaN values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to find maximum.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array NaN-safe maximum result.
M.nanmax = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.nanmax(x, axis, keepdims or false)
end)

--- Minimum of array elements, ignoring NaN values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to find minimum.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @treturn Array NaN-safe minimum result.
M.nanmin = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
})(function(x, axis, keepdims)
  return native.nanmin(x, axis, keepdims or false)
end)

--- Standard deviation, ignoring NaN values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute std.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @int[opt=0] ddof Delta degrees of freedom.
-- @treturn Array NaN-safe standard deviation.
M.nanstd = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
  { name = "ddof", type = argrule.any, opt = true },
})(function(x, axis, keepdims, ddof)
  return native.nanstd(x, axis, keepdims or false, ddof or 0)
end)

--- Variance, ignoring NaN values.
-- @tparam Array x Input array.
-- @int[opt] axis Axis along which to compute variance.
-- @bool[opt=false] keepdims If true, reduced axes are kept with size 1.
-- @int[opt=0] ddof Delta degrees of freedom.
-- @treturn Array NaN-safe variance.
M.nanvar = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "keepdims", type = argrule.any, opt = true },
  { name = "ddof", type = argrule.any, opt = true },
})(function(x, axis, keepdims, ddof)
  return native.nanvar(x, axis, keepdims or false, ddof or 0)
end)

return M
