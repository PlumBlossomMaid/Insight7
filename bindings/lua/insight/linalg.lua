--- Linear algebra operations.
--
-- Provides matrix multiplication, decompositions, eigenvalue
-- solvers, and other standard linear algebra routines.
--
-- @module insight.linalg

local native = require("_insight")
local M = {}

local rule = require("argrule.rule")
local argrule = require("argrule")

--- Matrix multiplication.
-- @tparam Array a First input matrix.
-- @tparam Array b Second input matrix.
-- @treturn Array Matrix product.
M.matmul = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.matmul(a, b)
  end
)

--- Dot product of two arrays.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Dot product result.
M.dot = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.dot(a, b)
  end
)

--- Outer product of two 1-D arrays.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Outer product matrix.
M.outer = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.outer(a, b)
  end
)

--- Determinant of a square matrix.
-- @tparam Array x Input square matrix.
-- @number Determinant value.
M.det = rule({ { name = "x", type = argrule.any, opt = true } })(function(x)
  return native.det(x)
end)

--- Matrix inverse.
-- @tparam Array x Input square matrix.
-- @treturn Array Inverse matrix.
M.inv = rule({ { name = "x", type = argrule.any, opt = true } })(function(x)
  return native.inv(x)
end)

--- Solve linear system Ax = b.
-- @tparam Array a Coefficient matrix.
-- @tparam Array b Right-hand side.
-- @treturn Array Solution x.
M.solve = rule({ { name = "a", type = argrule.any, opt = true }, { name = "b", type = argrule.any, opt = true } })(
  function(a, b)
    return native.solve(a, b)
  end
)

--- Singular value decomposition.
-- @tparam Array x Input matrix.
-- @treturn Array U, S, Vt (or compact equivalents).
M.svd = rule({ { name = "x", type = argrule.any, opt = true } })(function(x)
  return native.svd(x)
end)

--- Eigenvalue decomposition for symmetric/Hermitian matrices.
-- @tparam Array x Symmetric input matrix.
-- @string[opt="L"] uplo Upper or lower triangular ("U" or "L").
-- @treturn Array Eigenvalues, eigenvectors.
M.eigh = rule({ { name = "x", type = argrule.any, opt = true }, { name = "uplo", type = argrule.any, opt = true } })(
  function(x, uplo)
    return native.eigh(x, uplo or "L")
  end
)

--- Cholesky decomposition.
-- @tparam Array x Symmetric positive-definite matrix.
-- @bool[opt=true] lower If true, return lower triangular factor.
-- @treturn Array Cholesky factor.
M.cholesky = rule({ { name = "x", type = argrule.any, opt = true }, { name = "lower", type = argrule.any, opt = true } })(
  function(x, lower)
    return native.cholesky(x, lower == nil and true or lower)
  end
)

--- QR decomposition.
-- @tparam Array x Input matrix.
-- @string[opt="reduced"] mode "reduced" or "complete".
-- @treturn Array Q, R matrices.
M.qr = rule({ { name = "x", type = argrule.any, opt = true }, { name = "mode", type = argrule.any, opt = true } })(
  function(x, mode)
    return native.qr(x, mode or "reduced")
  end
)

--- Matrix or vector norm.
-- @tparam Array x Input array.
-- @number[opt=2] ord Order of the norm.
-- @number Norm value.
M.norm = rule({ { name = "x", type = argrule.any, opt = true }, { name = "ord", type = argrule.any, opt = true } })(
  function(x, ord)
    return native.norm(x, ord or 2.0)
  end
)

--- Sum of diagonal elements (trace).
-- @tparam Array x Input square matrix.
-- @number Trace value.
M.trace = rule({ { name = "x", type = argrule.any, opt = true } })(function(x)
  return native.trace(x)
end)

--- Least-squares solution to a linear system.
-- @tparam Array a Coefficient matrix.
-- @tparam Array b Target values.
-- @number[opt=-1] rcond Cut-off ratio for small singular values.
-- @treturn Array Least-squares solution.
M.lstsq = rule({
  { name = "a", type = argrule.any, opt = true },
  { name = "b", type = argrule.any, opt = true },
  { name = "rcond", type = argrule.any, opt = true },
})(function(a, b, rcond)
  return native.lstsq(a, b, rcond or -1.0)
end)

--- Condition number of a matrix.
-- @tparam Array x Input matrix.
-- @number[opt=2] p Order of the norm.
-- @number Condition number.
M.cond = rule({ { name = "x", type = argrule.any, opt = true }, { name = "p", type = argrule.any, opt = true } })(
  function(x, p)
    return native.cond(x, p or 2.0)
  end
)

--- Rank of a matrix.
-- @tparam Array x Input matrix.
-- @number[opt=-1] tol Tolerance threshold for singular values.
-- @int Matrix rank.
M.matrix_rank = rule({ { name = "x", type = argrule.any, opt = true }, { name = "tol", type = argrule.any, opt = true } })(
  function(x, tol)
    return native.matrix_rank(x, tol or -1.0)
  end
)

return M
