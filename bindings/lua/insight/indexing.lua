--- Indexing and searching operations.
--
-- Provides functions for taking, sorting, gathering, scattering,
-- and searching elements in arrays.
--
-- @module insight.indexing

local native = require("_insight")
local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Take elements from an array along an axis by index.
-- @tparam Array x Input array.
-- @tparam Array indices Indices of elements to take.
-- @int[opt] axis Axis along which to take. If nil, operates on flattened array.
-- @treturn Array Gathered elements.
M.take = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "indices", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
})(function(x, indices, axis)
  return native.take(x, indices, axis)
end)

--- Return indices of non-zero elements.
-- @tparam Array x Input array.
-- @treturn Array Indices of non-zero elements.
M.nonzero = rule({ { name = "x", type = argrule.any, opt = true } })(function(x)
  return native.nonzero(x)
end)

--- Return indices of non-zero elements (flattened).
-- @tparam Array x Input array.
-- @treturn Array Flat indices of non-zero elements.
M.flatnonzero = rule({ { name = "x", type = argrule.any, opt = true } })(function(x)
  return native.flatnonzero(x)
end)

--- Return indices that would sort the array.
-- @tparam Array x Input array.
-- @int[opt=-1] axis Axis along which to sort. -1 is last axis.
-- @treturn Array Index array for sorting.
M.argsort = rule({ { name = "x", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true } })(
  function(x, axis)
    return native.argsort(x, axis or -1)
  end
)

--- Return a sorted copy of the array.
-- @tparam Array x Input array.
-- @int[opt=-1] axis Axis along which to sort. -1 is last axis.
-- @treturn Array Sorted array.
M.sort = rule({ { name = "x", type = argrule.any, opt = true }, { name = "axis", type = argrule.any, opt = true } })(
  function(x, axis)
    return native.sort(x, axis or -1)
  end
)

--- Return unique elements of the array.
-- @tparam Array x Input array.
-- @treturn UniqueResult Table with fields: values, indices, inverse, counts.
M.unique = rule({ { name = "x", type = argrule.any, opt = true } })(function(x)
  return native.unique(x)
end)

--- Return the k largest or smallest elements.
-- @tparam Array x Input array.
-- @int k Number of elements to return.
-- @int[opt=-1] axis Axis along which to search. -1 is last axis.
-- @bool[opt=true] largest If true, return largest; otherwise smallest.
-- @treturn Array Top-k values.
M.topk = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "k", type = argrule.any, opt = true },
  { name = "axis", type = argrule.any, opt = true },
  { name = "largest", type = argrule.any, opt = true },
})(function(x, k, axis, largest)
  return native.topk(x, k, axis or -1, largest == nil and true or largest)
end)

--- Gather values along an axis by index.
-- @tparam Array x Input array.
-- @int[opt=1] dim Axis along which to gather.
-- @tparam Array index Indices to gather.
-- @treturn Array Gathered values.
M.gather = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "dim", type = argrule.any, opt = true },
  { name = "index", type = argrule.any, opt = true },
})(function(x, dim, index)
  return native.gather(x, dim == nil and 1 or dim, index)
end)

--- Scatter values into an array by index.
-- @tparam Array x Input array.
-- @int[opt=1] dim Axis along which to scatter.
-- @tparam Array index Indices at which to scatter.
-- @tparam Array src Source values to scatter.
-- @treturn Array Updated array.
M.scatter = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "dim", type = argrule.any, opt = true },
  { name = "index", type = argrule.any, opt = true },
  { name = "src", type = argrule.any, opt = true },
})(function(x, dim, index, src)
  return native.scatter(x, dim == nil and 1 or dim, index, src)
end)

--- Scatter-add values into an array by index.
-- @tparam Array x Input array.
-- @int[opt=1] dim Axis along which to scatter-add.
-- @tparam Array index Indices at which to add.
-- @tparam Array src Source values to add.
-- @treturn Array Updated array.
M.scatter_add = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "dim", type = argrule.any, opt = true },
  { name = "index", type = argrule.any, opt = true },
  { name = "src", type = argrule.any, opt = true },
})(function(x, dim, index, src)
  return native.scatter_add(x, dim == nil and 1 or dim, index, src)
end)

--- Scatter with reduction into an array by index.
-- @tparam Array x Input array.
-- @int[opt=1] dim Axis along which to scatter-reduce.
-- @tparam Array index Indices at which to reduce.
-- @tparam Array src Source values.
-- @string reduce Reduction type ("add", "mul", "mean", "amax", "amin").
-- @treturn Array Updated array.
M.scatter_reduce = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "dim", type = argrule.any, opt = true },
  { name = "index", type = argrule.any, opt = true },
  { name = "src", type = argrule.any, opt = true },
  { name = "reduce", type = argrule.any, opt = true },
})(function(x, dim, index, src, reduce)
  return native.scatter_reduce(x, dim == nil and 1 or dim, index, src, reduce)
end)

--- 1-D linear interpolation.
-- @tparam Array x Points at which to evaluate.
-- @tparam Array xp Known x-coordinates (must be sorted).
-- @tparam Array fp Known y-coordinates.
-- @treturn Array Interpolated values.
M.interp = rule({
  { name = "x", type = argrule.any, opt = true },
  { name = "xp", type = argrule.any, opt = true },
  { name = "fp", type = argrule.any, opt = true },
})(function(x, xp, fp)
  return native.interp(x, xp, fp)
end)

--- Return an array of indices (meshgrid-like).
-- @tparam table dimensions Shape of the output grid.
-- @treturn Array Index arrays.
M.indices = rule({ { name = "dimensions", type = argrule.table, opt = true } })(function(dimensions)
  return native.indices(native.Shape(dimensions))
end)

--- Construct open mesh from sequences (for advanced indexing).
-- @tparam table args Sequence of 1-D arrays.
-- @treturn table List of open mesh arrays.
M.ix_ = rule({})(function(...)
  return native.ix_(...)
end)

return M
