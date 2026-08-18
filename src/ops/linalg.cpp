// src/ops/linalg.cpp
/**
 * @file linalg.cpp
 * @brief Linear algebra operations.
 *
 * Provides matrix multiplication, decompositions, eigenvalue computations,
 * equation solving, and other linear algebra routines.
 */

#include "insight/ops/linalg.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include "insight/ops/broadcast.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/reduction.h"
#include "insight/utils/promotion.h"
#include <cmath>
#include <string>
#include <vector>

namespace ins {

// ============================================================================
// Helper functions
// ============================================================================

static DType get_working_dtype(DType dtype) {
  if (is_integer(dtype) || dtype == DType::BOOL) {
    return DType::F64;
  }
  if (dtype == DType::F32) {
    return DType::F32;
  }
  return DType::F64;
}

static Array ensure_2d(const Array &x, const char *func_name) {
  INS_CHECK(x.shape().ndim() == 2, func_name, ": input must be 2D");
  return x;
}

static Array ensure_2d_square(const Array &x, const char *func_name) {
  INS_CHECK(x.shape().ndim() == 2, func_name, ": input must be 2D");
  INS_CHECK(x.shape().dim(0) == x.shape().dim(1), func_name,
            ": input must be square matrix");
  return x;
}

static Array ensure_1d(const Array &x, const char *func_name) {
  INS_CHECK(x.shape().ndim() == 1, func_name, ": input must be 1D");
  return x;
}

static Array to_working_dtype(const Array &x, DType target_dtype) {
  if (x.dtype() == target_dtype) {
    return x;
  }
  return x.to(target_dtype);
}

static Array to_original_dtype(const Array &x, DType original_dtype) {
  if (x.dtype() == original_dtype) {
    return x;
  }
  return x.to(original_dtype);
}

static OpSchema linalg_schema(const char *name, size_t input_count,
                              size_t output_count = 1) {
  std::vector<OpArgumentSchema> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    inputs.push_back({"x" + std::to_string(i), OpArgumentKind::Array,
                      OpArgumentAccess::ReadOnly});
  }

  std::vector<OpArgumentSchema> outputs;
  outputs.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    outputs.push_back({output_count == 1 ? "out" : "out" + std::to_string(i),
                       OpArgumentKind::Array, OpArgumentAccess::WriteOnly});
  }

  return OpSchema(name, OpKind::Linalg, inputs, outputs)
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

static ArrayIterator linalg_iterator(const char *name,
                                     const std::vector<Array> &inputs,
                                     const std::vector<Array> &outputs) {
  OpSchema schema = linalg_schema(name, inputs.size(), outputs.size());
  return schema.make_transform_iterator(inputs, outputs);
}

static ArrayIterator linalg_iterator(const char *name,
                                     const std::vector<Array> &inputs,
                                     const Array &output) {
  return linalg_iterator(name, inputs, std::vector<Array>{output});
}

static void launch_linalg(const char *name, const Place &place, DType dtype,
                          const std::vector<Array> &inputs,
                          const std::vector<Array> &outputs,
                          std::vector<OpRegistry::Arg> scalars = {}) {
  ArrayIterator iter = linalg_iterator(name, inputs, outputs);
  auto arrays = iter.arrays();

  std::vector<OpRegistry::Arg> launch_inputs;
  launch_inputs.reserve(outputs.size() + inputs.size() + scalars.size());
  for (size_t i = inputs.size(); i < arrays.size(); ++i) {
    launch_inputs.push_back(OpRegistry::array_arg(arrays[i].layout_ptr()));
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    launch_inputs.push_back(OpRegistry::array_arg(arrays[i].layout_ptr()));
  }
  launch_inputs.insert(launch_inputs.end(), scalars.begin(), scalars.end());

  std::vector<OpRegistry::Arg> launch_outputs;
  launch_outputs.reserve(outputs.size());
  for (size_t i = inputs.size(); i < arrays.size(); ++i) {
    launch_outputs.push_back(OpRegistry::array_arg(arrays[i].layout_ptr()));
  }

  OpRegistry::launch_schema(name, place, dtype, launch_inputs, launch_outputs);
}

static void launch_linalg(const char *name, const Place &place, DType dtype,
                          const std::vector<Array> &inputs,
                          const Array &output,
                          std::vector<OpRegistry::Arg> scalars = {}) {
  launch_linalg(name, place, dtype, inputs, std::vector<Array>{output}, scalars);
}

// ============================================================================
// matmul - Matrix Multiplication
// ============================================================================
Array matmul(const Array &a, const Array &b) {
  INS_CHECK(a.defined() && b.defined(), "matmul: input is undefined");

  int ndim_a = a.shape().ndim();
  int ndim_b = b.shape().ndim();

  // Determine output shape and working dtype
  DType working_dtype = get_working_dtype(a.dtype());
  Array a_work = to_working_dtype(a, working_dtype);
  Array b_work = to_working_dtype(b, working_dtype);

  if (a_work.place() != b_work.place()) {
    b_work = b_work.to(a_work.place());
  }

  if (!a_work.is_contiguous()) {
    a_work = a_work.contiguous();
  }
  if (!b_work.is_contiguous()) {
    b_work = b_work.contiguous();
  }

  Shape out_shape;

  // Case 1: vector * vector -> scalar
  if (ndim_a == 1 && ndim_b == 1) {
    INS_CHECK(a.numel() == b.numel(), "matmul: vector length mismatch");
    out_shape = Shape({});
  }
  // Case 2: matrix * vector -> vector
  else if (ndim_a == 2 && ndim_b == 1) {
    INS_CHECK(a.shape().dim(1) == b.numel(), "matmul: shape mismatch");
    out_shape = Shape({a.shape().dim(0)});
  }
  // Case 3: vector * matrix -> vector
  else if (ndim_a == 1 && ndim_b == 2) {
    INS_CHECK(a.numel() == b.shape().dim(0), "matmul: shape mismatch");
    out_shape = Shape({b.shape().dim(1)});
  }
  // Case 4: matrix * matrix -> matrix
  else if (ndim_a == 2 && ndim_b == 2) {
    INS_CHECK(a.shape().dim(1) == b.shape().dim(0), "matmul: shape mismatch");
    out_shape = Shape({a.shape().dim(0), b.shape().dim(1)});
  }
  // Case 5: batched matrix multiplication
  else {
    INS_CHECK(ndim_a >= 2 && ndim_b >= 2, "matmul: batched requires >= 2D");

    int64_t m = a.shape().dim(ndim_a - 2);
    int64_t k_a = a.shape().dim(ndim_a - 1);
    int64_t k_b = b.shape().dim(ndim_b - 2);
    int64_t n = b.shape().dim(ndim_b - 1);

    INS_CHECK(k_a == k_b, "matmul: shape mismatch for batch matrices");

    // Broadcast batch dimensions (all except last 2)
    int max_batch_ndim = std::max(ndim_a - 2, ndim_b - 2);
    std::vector<int64_t> batch_dims(max_batch_ndim);
    for (int i = 0; i < max_batch_ndim; ++i) {
      int64_t a_dim = (i < ndim_a - 2) ? a.shape().dim(ndim_a - 3 - i) : 1;
      int64_t b_dim = (i < ndim_b - 2) ? b.shape().dim(ndim_b - 3 - i) : 1;
      if (a_dim != 1 && b_dim != 1 && a_dim != b_dim) {
        INS_THROW("matmul: batch dimensions not broadcastable");
      }
      batch_dims[max_batch_ndim - 1 - i] = std::max(a_dim, b_dim);
    }

    // Build output shape: batch_dims + [m, n]
    std::vector<int64_t> out_dims(batch_dims);
    out_dims.push_back(m);
    out_dims.push_back(n);
    out_shape = Shape(out_dims);

    // Broadcast inputs to match batch dimensions
    std::vector<int64_t> a_batch, b_batch;
    for (int i = 0; i < ndim_a - 2; ++i)
      a_batch.push_back(a.shape().dim(i));
    for (int i = 0; i < ndim_b - 2; ++i)
      b_batch.push_back(b.shape().dim(i));
    while ((int)a_batch.size() < max_batch_ndim)
      a_batch.insert(a_batch.begin(), 1);
    while ((int)b_batch.size() < max_batch_ndim)
      b_batch.insert(b_batch.begin(), 1);

    bool need_bc = false;
    for (int i = 0; i < max_batch_ndim; ++i) {
      if (a_batch[i] != batch_dims[i] || b_batch[i] != batch_dims[i]) {
        need_bc = true;
        break;
      }
    }

    if (need_bc) {
      std::vector<int64_t> a_full(batch_dims);
      a_full.push_back(m);
      a_full.push_back(k_a);
      std::vector<int64_t> b_full(batch_dims);
      b_full.push_back(k_b);
      b_full.push_back(n);
      a_work = broadcast_to(a_work, Shape(a_full));
      b_work = broadcast_to(b_work, Shape(b_full));
    }
  }

  Array result(out_shape, working_dtype, a_work.place());
  launch_linalg("matmul", a_work.place(), working_dtype, {a_work, b_work},
                result);

  return to_original_dtype(result, a.dtype());
}

// ============================================================================
// dot - Dot product of two 1D vectors
// ============================================================================

Array dot(const Array &a, const Array &b) {
  Array a_1d = ensure_1d(a, "dot");
  Array b_1d = ensure_1d(b, "dot");
  INS_CHECK(a_1d.numel() == b_1d.numel(), "dot: size mismatch");

  DType working_dtype = get_working_dtype(a.dtype());
  Array a_work = to_working_dtype(a_1d, working_dtype);
  Array b_work = to_working_dtype(b_1d, working_dtype);

  if (a_work.place() != b_work.place()) {
    b_work = b_work.to(a_work.place());
  }

  Array result(Shape({}), working_dtype, a_work.place());
  launch_linalg("dot", a_work.place(), working_dtype, {a_work, b_work}, result);

  return to_original_dtype(result, a.dtype());
}

// ============================================================================
// outer - Outer product of two 1D vectors
// ============================================================================

Array outer(const Array &a, const Array &b) {
  Array a_1d = ensure_1d(a, "outer");
  Array b_1d = ensure_1d(b, "outer");

  DType working_dtype = get_working_dtype(a.dtype());
  Array a_work = to_working_dtype(a_1d, working_dtype);
  Array b_work = to_working_dtype(b_1d, working_dtype);

  if (a_work.place() != b_work.place()) {
    b_work = b_work.to(a_work.place());
  }

  Shape out_shape({a_work.numel(), b_work.numel()});
  Array result(out_shape, working_dtype, a_work.place());
  launch_linalg("outer", a_work.place(), working_dtype, {a_work, b_work},
                result);

  return to_original_dtype(result, a.dtype());
}

// ============================================================================
// trace - Sum of diagonal elements
// ============================================================================

Array trace(const Array &x) {
  Array x_2d = ensure_2d(x, "trace");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  Array result(Shape({}), working_dtype, work.place());
  launch_linalg("trace", work.place(), working_dtype, {work}, result);

  return to_original_dtype(result, x.dtype());
}

// ============================================================================
// norm - Matrix or vector norm
// ============================================================================

Array norm(const Array &x, double ord) {
  INS_CHECK(x.defined(), "norm: input is undefined");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x, working_dtype);

  Array result(Shape({}), working_dtype, work.place());
  launch_linalg("norm", work.place(), working_dtype, {work}, result,
                {OpRegistry::scalar_arg(&ord)});

  return to_original_dtype(result, x.dtype());
}

// ============================================================================
// det - Determinant of a square matrix
// ============================================================================

Array det(const Array &x) {
  Array x_2d = ensure_2d_square(x, "det");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  Array result(Shape({}), working_dtype, work.place());
  launch_linalg("det", work.place(), working_dtype, {work}, result);

  return to_original_dtype(result, x.dtype());
}

// ============================================================================
// slogdet - Sign and log determinant
// ============================================================================

std::pair<Array, Array> slogdet(const Array &x) {
  Array x_2d = ensure_2d_square(x, "slogdet");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  Array sign(Shape({}), working_dtype, work.place());
  Array logdet(Shape({}), DType::F64, work.place()); // logdet always double
  launch_linalg("slogdet", work.place(), working_dtype, {work}, {sign, logdet});

  return {to_original_dtype(sign, x.dtype()), logdet};
}

// ============================================================================
// cond - Condition number
// ============================================================================

Array cond(const Array &x, double p) {
  Array x_2d = ensure_2d(x, "cond");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  Array result(Shape({}), working_dtype, work.place());
  launch_linalg("cond", work.place(), working_dtype, {work}, result,
                {OpRegistry::scalar_arg(&p)});

  return to_original_dtype(result, x.dtype());
}

// ============================================================================
// matrix_rank - Rank of a matrix
// ============================================================================

Array matrix_rank(const Array &x, double tol) {
  Array x_2d = ensure_2d(x, "matrix_rank");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  Array result(Shape({}), DType::I64, work.place());
  launch_linalg("matrix_rank", work.place(), working_dtype, {work}, result,
                {OpRegistry::scalar_arg(&tol)});

  return result; // rank is int64, no dtype conversion needed
}

// ============================================================================
// inv - Matrix inverse
// ============================================================================

Array inv(const Array &x) {
  Array x_2d = ensure_2d_square(x, "inv");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  Array result(work.shape(), working_dtype, work.place());
  launch_linalg("inv", work.place(), working_dtype, {work}, result);

  return to_original_dtype(result, x.dtype());
}

// ============================================================================
// pinv - Moore-Penrose pseudo-inverse
// ============================================================================

Array pinv(const Array &x, double rcond) {
  Array x_2d = ensure_2d(x, "pinv");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  // Output shape is (n, m) for input (m, n)
  int64_t m = work.shape().dim(0);
  int64_t n = work.shape().dim(1);
  Array result(Shape({n, m}), working_dtype, work.place());
  launch_linalg("pinv", work.place(), working_dtype, {work}, result,
                {OpRegistry::scalar_arg(&rcond)});

  return to_original_dtype(result, x.dtype());
}

// ============================================================================
// matrix_power - Integer power of a square matrix
// ============================================================================

Array matrix_power(const Array &x, int n) {
  Array x_2d = ensure_2d_square(x, "matrix_power");
  INS_CHECK(n >= 0, "matrix_power: exponent must be non-negative");

  if (n == 0) {
    return eye(x.shape().dim(0), x.shape().dim(1), 0, DType::F64, x.place())
        .to(x.dtype());
  }

  if (n == 1) {
    return x;
  }

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  Array result(work.shape(), working_dtype, work.place());
  launch_linalg("matrix_power", work.place(), working_dtype, {work}, result,
                {OpRegistry::scalar_arg(&n)});

  return to_original_dtype(result, x.dtype());
}

// ============================================================================
// lu - LU decomposition
// ============================================================================

std::tuple<Array, Array> lu(const Array &x, bool pivot) {
  Array x_2d = ensure_2d_square(x, "lu");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  Array LU(work.shape(), working_dtype, work.place());
  Array pivots(Shape({work.shape().dim(0)}), DType::I32, work.place());

  int pivot_flag = pivot ? 1 : 0;
  launch_linalg("lu", work.place(), working_dtype, {work}, {LU, pivots},
                {OpRegistry::scalar_arg(&pivot_flag)});

  return {to_original_dtype(LU, x.dtype()), pivots};
}

// ============================================================================
// lu_unpack - Unpack LU decomposition into P, L, U
// ============================================================================

std::tuple<Array, Array, Array> lu_unpack(const Array &LU,
                                          const Array &pivots) {
  Array lu_2d = ensure_2d(LU, "lu_unpack");

  DType working_dtype = get_working_dtype(lu_2d.dtype());
  Array work = to_working_dtype(lu_2d, working_dtype);

  int n = static_cast<int>(work.shape().dim(0));

  Array P(Shape({n, n}), working_dtype, work.place());
  Array L(Shape({n, n}), working_dtype, work.place());
  Array U(Shape({n, n}), working_dtype, work.place());

  launch_linalg("lu_unpack", work.place(), working_dtype, {work, pivots},
                {P, L, U});

  return {to_original_dtype(P, LU.dtype()), to_original_dtype(L, LU.dtype()),
          to_original_dtype(U, LU.dtype())};
}

// ============================================================================
// qr - QR decomposition
// ============================================================================

std::tuple<Array, Array> qr(const Array &x, const std::string &mode) {
  Array x_2d = ensure_2d(x, "qr");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  int m = static_cast<int>(work.shape().dim(0));
  int n = static_cast<int>(work.shape().dim(1));
  int k = std::min(m, n);

  int mode_code = (mode == "complete") ? 0 : 1; // 0: complete, 1: reduced

  Array Q, R;

  if (mode == "complete") {
    Q = Array(Shape({m, m}), working_dtype, work.place());
    R = Array(Shape({m, n}), working_dtype, work.place());
  } else {
    Q = Array(Shape({m, k}), working_dtype, work.place());
    R = Array(Shape({k, n}), working_dtype, work.place());
  }

  launch_linalg("qr", work.place(), working_dtype, {work}, {Q, R},
                {OpRegistry::scalar_arg(&mode_code)});

  return {to_original_dtype(Q, x.dtype()), to_original_dtype(R, x.dtype())};
}

// ============================================================================
// lq - LQ decomposition
// ============================================================================

std::tuple<Array, Array> lq(const Array &x, const std::string &mode) {
  Array x_2d = ensure_2d(x, "lq");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  int m = static_cast<int>(work.shape().dim(0));
  int n = static_cast<int>(work.shape().dim(1));
  int k = std::min(m, n);

  int mode_code = (mode == "complete") ? 0 : 1;

  Array L, Q;

  if (mode == "complete") {
    L = Array(Shape({m, n}), working_dtype, work.place());
    Q = Array(Shape({n, n}), working_dtype, work.place());
  } else {
    L = Array(Shape({m, k}), working_dtype, work.place());
    Q = Array(Shape({n, k}), working_dtype, work.place());
  }

  launch_linalg("lq", work.place(), working_dtype, {work}, {L, Q},
                {OpRegistry::scalar_arg(&mode_code)});

  return {to_original_dtype(L, x.dtype()), to_original_dtype(Q, x.dtype())};
}

// ============================================================================
// cholesky - Cholesky decomposition
// ============================================================================

Array cholesky(const Array &x, bool lower) {
  Array x_2d = ensure_2d_square(x, "cholesky");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  Array result(work.shape(), working_dtype, work.place());

  int lower_flag = lower ? 1 : 0;
  launch_linalg("cholesky", work.place(), working_dtype, {work}, result,
                {OpRegistry::scalar_arg(&lower_flag)});

  return to_original_dtype(result, x.dtype());
}

// ============================================================================
// cholesky_solve - Solve linear system using Cholesky decomposition
// ============================================================================

Array cholesky_solve(const Array &A, const Array &B, bool lower) {
  Array A_2d = ensure_2d_square(A, "cholesky_solve");

  DType working_dtype = get_working_dtype(A.dtype());
  Array A_work = to_working_dtype(A_2d, working_dtype);
  Array B_work = to_working_dtype(B, working_dtype);

  if (A_work.place() != B_work.place()) {
    B_work = B_work.to(A_work.place());
  }

  Array result(B_work.shape(), working_dtype, A_work.place());

  int lower_flag = lower ? 1 : 0;
  launch_linalg("cholesky_solve", A_work.place(), working_dtype,
                {A_work, B_work}, result,
                {OpRegistry::scalar_arg(&lower_flag)});

  return to_original_dtype(result, B.dtype());
}

// ============================================================================
// svd - Singular value decomposition
// ============================================================================

std::tuple<Array, Array, Array> svd(const Array &x, bool full_matrices) {
  Array x_2d = ensure_2d(x, "svd");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  int m = static_cast<int>(work.shape().dim(0));
  int n = static_cast<int>(work.shape().dim(1));
  int min_mn = std::min(m, n);

  int full_flag = full_matrices ? 1 : 0;

  Array U, S, VT;

  if (full_matrices) {
    U = Array(Shape({m, m}), working_dtype, work.place());
    S = Array(Shape({min_mn}), working_dtype, work.place());
    VT = Array(Shape({n, n}), working_dtype, work.place());
  } else {
    U = Array(Shape({m, min_mn}), working_dtype, work.place());
    S = Array(Shape({min_mn}), working_dtype, work.place());
    VT = Array(Shape({min_mn, n}), working_dtype, work.place());
  }

  launch_linalg("svd", work.place(), working_dtype, {work}, {U, S, VT},
                {OpRegistry::scalar_arg(&full_flag)});

  return {to_original_dtype(U, x.dtype()), to_original_dtype(S, x.dtype()),
          to_original_dtype(VT, x.dtype())};
}

// ============================================================================
// svdvals - Singular values only
// ============================================================================

Array svdvals(const Array &x) {
  Array x_2d = ensure_2d(x, "svdvals");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  int m = static_cast<int>(work.shape().dim(0));
  int n = static_cast<int>(work.shape().dim(1));
  int min_mn = std::min(m, n);

  Array result(Shape({min_mn}), working_dtype, work.place());
  launch_linalg("svdvals", work.place(), working_dtype, {work}, result);

  return to_original_dtype(result, x.dtype());
}

// ============================================================================
// eig - Eigenvalues and eigenvectors (general matrix, may be complex)
// ============================================================================

std::tuple<Array, Array> eig(const Array &x) {
  Array x_2d = ensure_2d(x, "eig");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  int n = static_cast<int>(work.shape().dim(0));

  // Eigenvalues are always complex
  DType complex_dtype = (working_dtype == DType::F32) ? DType::C32 : DType::C64;
  Array eigenvalues(Shape({n}), complex_dtype, work.place());
  Array eigenvectors(Shape({n, n}), complex_dtype, work.place());
  launch_linalg("eig", work.place(), working_dtype, {work},
                {eigenvalues, eigenvectors});

  return {eigenvalues, eigenvectors};
}

// ============================================================================
// eigvals - Eigenvalues only (general matrix)
// ============================================================================

Array eigvals(const Array &x) {
  Array x_2d = ensure_2d(x, "eigvals");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  int n = static_cast<int>(work.shape().dim(0));

  DType complex_dtype = (working_dtype == DType::F32) ? DType::C32 : DType::C64;
  Array result(Shape({n}), complex_dtype, work.place());
  launch_linalg("eigvals", work.place(), working_dtype, {work}, result);

  return result;
}

// ============================================================================
// eigh - Eigenvalues and eigenvectors (symmetric/Hermitian)
// ============================================================================

std::tuple<Array, Array> eigh(const Array &x, const std::string &uplo) {
  Array x_2d = ensure_2d_square(x, "eigh");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  int n = static_cast<int>(work.shape().dim(0));

  Array eigenvalues(Shape({n}), working_dtype, work.place());
  Array eigenvectors(work.shape(), working_dtype, work.place());

  int uplo_code = (uplo == "L") ? 0 : 1;
  launch_linalg("eigh", work.place(), working_dtype, {work},
                {eigenvalues, eigenvectors},
                {OpRegistry::scalar_arg(&uplo_code)});

  return {to_original_dtype(eigenvalues, x.dtype()),
          to_original_dtype(eigenvectors, x.dtype())};
}

// ============================================================================
// eigvalsh - Eigenvalues only (symmetric/Hermitian)
// ============================================================================

Array eigvalsh(const Array &x, const std::string &uplo) {
  Array x_2d = ensure_2d_square(x, "eigvalsh");

  DType working_dtype = get_working_dtype(x.dtype());
  Array work = to_working_dtype(x_2d, working_dtype);

  int n = static_cast<int>(work.shape().dim(0));

  Array result(Shape({n}), working_dtype, work.place());

  int uplo_code = (uplo == "L") ? 0 : 1;
  launch_linalg("eigvalsh", work.place(), working_dtype, {work}, result,
                {OpRegistry::scalar_arg(&uplo_code)});

  return to_original_dtype(result, x.dtype());
}

// ============================================================================
// solve - Solve linear system AX = B
// ============================================================================

Array solve(const Array &A, const Array &B) {
  Array A_2d = ensure_2d_square(A, "solve");

  DType working_dtype = get_working_dtype(A.dtype());
  Array A_work = to_working_dtype(A_2d, working_dtype);
  Array B_work = to_working_dtype(B, working_dtype);

  if (A_work.place() != B_work.place()) {
    B_work = B_work.to(A_work.place());
  }

  Array result(B_work.shape(), working_dtype, A_work.place());
  launch_linalg("solve", A_work.place(), working_dtype, {A_work, B_work},
                result);

  return to_original_dtype(result, B.dtype());
}

// ============================================================================
// lstsq - Least squares solution
// ============================================================================

Array lstsq(const Array &A, const Array &B, double rcond) {
  Array A_2d = ensure_2d(A, "lstsq");

  DType working_dtype = get_working_dtype(A.dtype());
  Array A_work = to_working_dtype(A_2d, working_dtype);
  Array B_work = to_working_dtype(B, working_dtype);

  if (A_work.place() != B_work.place()) {
    B_work = B_work.to(A_work.place());
  }

  // Output shape: (n, nrhs) where n = columns of A
  int n = static_cast<int>(A_work.shape().dim(1));
  int nrhs = (B_work.shape().ndim() == 1)
                 ? 1
                 : static_cast<int>(B_work.shape().dim(1));

  Shape out_shape =
      (B_work.shape().ndim() == 1) ? Shape({n}) : Shape({n, nrhs});
  Array result(out_shape, working_dtype, A_work.place());
  launch_linalg("lstsq", A_work.place(), working_dtype, {A_work, B_work},
                result, {OpRegistry::scalar_arg(&rcond)});

  return to_original_dtype(result, B.dtype());
}

// ============================================================================
// solve_triangular - Solve triangular system
// ============================================================================

Array solve_triangular(const Array &A, const Array &B, bool lower,
                       bool unit_diag) {
  Array A_2d = ensure_2d_square(A, "solve_triangular");

  DType working_dtype = get_working_dtype(A.dtype());
  Array A_work = to_working_dtype(A_2d, working_dtype);
  Array B_work = to_working_dtype(B, working_dtype);

  if (A_work.place() != B_work.place()) {
    B_work = B_work.to(A_work.place());
  }

  Array result(B_work.shape(), working_dtype, A_work.place());

  int lower_flag = lower ? 1 : 0;
  int unit_flag = unit_diag ? 1 : 0;
  launch_linalg("solve_triangular", A_work.place(), working_dtype,
                {A_work, B_work}, result,
                {OpRegistry::scalar_arg(&lower_flag),
                 OpRegistry::scalar_arg(&unit_flag)});

  return to_original_dtype(result, B.dtype());
}

Array cov(const Array &x, bool rowvar, int ddof) {
  INS_CHECK(x.defined(), "cov: input is undefined");
  INS_CHECK(x.shape().ndim() == 2, "cov: input must be 2D, got ",
            x.shape().ndim(), "D");
  INS_CHECK(ddof >= 0 && ddof <= x.shape().dim(1),
            "cov: ddof must be between 0 and number of observations");

  // If rowvar is false, transpose so that rows are variables
  Array data = x;
  if (!rowvar) {
    data = x.transpose();
  }

  // data shape: (m, n) where m = number of variables, n = number of
  // observations
  int64_t m = data.shape().dim(0);
  int64_t n = data.shape().dim(1);

  // Working dtype (float64 for better precision)
  DType working_dtype = (data.dtype() == DType::F32) ? DType::F32 : DType::F64;
  Array data_work =
      (data.dtype() == working_dtype) ? data : data.to(working_dtype);

  // Subtract mean along observations axis (axis=1)
  Array mean_vals = mean(data_work, 1, true);
  Array centered = sub(data_work, mean_vals);

  // Covariance matrix = (centered @ centered.T) / (n - ddof)
  Array centered_T = centered.transpose();
  Array cov_matrix = matmul(centered, centered_T);
  Array divisor = full(Shape({1}), static_cast<double>(n - ddof), working_dtype,
                       cov_matrix.place());
  Array result = div(cov_matrix, divisor);

  // Convert back to original dtype if needed
  if (result.dtype() != x.dtype()) {
    result = result.to(x.dtype());
  }

  return result;
}

} // namespace ins