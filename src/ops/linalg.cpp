// src/ops/linalg.cpp
/**
 * @file linalg.cpp
 * @brief Linear algebra operations.
 *
 * Provides matrix multiplication, decompositions, eigenvalue computations,
 * equation solving, and other linear algebra routines.
 */

#include "insight/ops/linalg.h"
#include "insight/core/axis.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include "insight/ops/broadcast.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/reduction.h"
#include "insight/utils/promotion.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace ins {

// ============================================================================
// Helper functions
// ============================================================================

static DType get_working_dtype(DType dtype) {
  // Complex inputs would silently lose their imaginary part when cast to the
  // real working dtype; no linalg kernel is registered for complex yet, so
  // reject them explicitly instead of returning wrong results.
  INS_CHECK(!is_complex(dtype), "linalg: complex inputs are not supported yet");
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
                          const std::vector<Array> &inputs, const Array &output,
                          std::vector<OpRegistry::Arg> scalars = {}) {
  launch_linalg(name, place, dtype, inputs, std::vector<Array>{output},
                scalars);
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
      // broadcast_to returns stride-0 views; the matmul kernel requires
      // dense inputs (cpu_ensure_contiguous only validates, never fixes).
      a_work = a_work.contiguous();
      b_work = b_work.contiguous();
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

// ============================================================================
// tensordot - Tensor dot product
// ============================================================================

Array tensordot(const Array &a, const Array &b, const std::vector<int> &axes_a,
                const std::vector<int> &axes_b) {
  INS_CHECK(a.defined() && b.defined(), "tensordot: input is undefined");
  INS_CHECK(axes_a.size() == axes_b.size(),
            "tensordot: axes lists must have the same length");
  INS_CHECK(a.shape().ndim() >= static_cast<int>(axes_a.size()) &&
                b.shape().ndim() >= static_cast<int>(axes_b.size()),
            "tensordot: too many axes for input ndim");

  std::vector<int> a_axes, b_axes;
  for (int ax : axes_a)
    a_axes.push_back(normalize_axis(ax, a.shape().ndim(), "tensordot"));
  for (int ax : axes_b)
    b_axes.push_back(normalize_axis(ax, b.shape().ndim(), "tensordot"));

  // Contracted dimensions must match.
  for (size_t i = 0; i < a_axes.size(); ++i) {
    INS_CHECK(a.shape().dim(a_axes[i]) == b.shape().dim(b_axes[i]),
              "tensordot: contracted dimensions must match");
  }

  // Keep axes: everything not contracted, in original order.
  std::vector<int> a_keep, b_keep;
  for (int i = 0; i < a.shape().ndim(); ++i) {
    if (std::find(a_axes.begin(), a_axes.end(), i) == a_axes.end())
      a_keep.push_back(i);
  }
  for (int i = 0; i < b.shape().ndim(); ++i) {
    if (std::find(b_axes.begin(), b_axes.end(), i) == b_axes.end())
      b_keep.push_back(i);
  }

  // Promote to a common dtype before computing.
  DType work_dtype = promote_types(a.dtype(), b.dtype());
  Array a_work = (a.dtype() == work_dtype) ? a : a.to(work_dtype);
  Array b_work = (b.dtype() == work_dtype) ? b : b.to(work_dtype);
  if (a_work.place() != b_work.place()) {
    b_work = b_work.to(a_work.place());
  }

  // a: [keep_a..., contract...] -> 2D (prod_keep_a, prod_contract)
  std::vector<int> a_perm(a_keep);
  a_perm.insert(a_perm.end(), a_axes.begin(), a_axes.end());
  Array a_p = permute(a_work, a_perm);
  std::vector<int64_t> a_dims = a_p.shape().dims();
  int64_t prod_keep_a = 1, prod_contract = 1;
  for (size_t i = 0; i < a_dims.size(); ++i) {
    if (i < a_keep.size())
      prod_keep_a *= a_dims[i];
    else
      prod_contract *= a_dims[i];
  }
  Array a2 = reshape(a_p, Shape({prod_keep_a, prod_contract}));

  // b: [contract..., keep_b...] -> 2D (prod_contract, prod_keep_b)
  std::vector<int> b_perm(b_axes);
  b_perm.insert(b_perm.end(), b_keep.begin(), b_keep.end());
  Array b_p = permute(b_work, b_perm);
  std::vector<int64_t> b_dims = b_p.shape().dims();
  int64_t prod_keep_b = 1;
  int64_t prod_contract_b = 1;
  for (size_t i = 0; i < b_dims.size(); ++i) {
    if (i < b_axes.size())
      prod_contract_b *= b_dims[i];
    else
      prod_keep_b *= b_dims[i];
  }
  INS_CHECK(prod_contract_b == prod_contract,
            "tensordot: contracted dimensions must match");
  Array b2 = reshape(b_p, Shape({prod_contract, prod_keep_b}));

  Array prod = matmul(a2, b2); // (prod_keep_a, prod_keep_b)

  // Output shape: a keep dims followed by b keep dims.
  std::vector<int64_t> out_dims;
  for (int i = 0; i < a.shape().ndim(); ++i) {
    if (std::find(a_axes.begin(), a_axes.end(), i) == a_axes.end())
      out_dims.push_back(a.shape().dim(i));
  }
  for (int i = 0; i < b.shape().ndim(); ++i) {
    if (std::find(b_axes.begin(), b_axes.end(), i) == b_axes.end())
      out_dims.push_back(b.shape().dim(i));
  }

  // reshape handles the scalar case (out_dims empty) since prod is 1x1.
  Array result = reshape(prod, Shape(out_dims));
  if (result.dtype() != work_dtype) {
    result = result.to(work_dtype);
  }
  return result;
}

Array tensordot(const Array &a, const Array &b, int axes) {
  INS_CHECK(a.defined() && b.defined(), "tensordot: input is undefined");
  INS_CHECK(axes >= 0, "tensordot: axes must be non-negative");

  int ndim_a = a.shape().ndim();
  int ndim_b = b.shape().ndim();
  INS_CHECK(axes <= ndim_a && axes <= ndim_b,
            "tensordot: axes exceeds input ndim");

  std::vector<int> axes_a(axes), axes_b(axes);
  for (int i = 0; i < axes; ++i) {
    axes_a[i] = ndim_a - axes + i;
    axes_b[i] = i;
  }
  return tensordot(a, b, axes_a, axes_b);
}

// ============================================================================
// kron - Kronecker product
// ============================================================================

Array kron(const Array &a, const Array &b) {
  INS_CHECK(a.defined() && b.defined(), "kron: input is undefined");

  int ndim = std::max(a.shape().ndim(), b.shape().ndim());

  // Left-pad both shapes to the same ndim.
  std::vector<int64_t> a_pad(ndim, 1), b_pad(ndim, 1);
  for (int i = 0; i < a.shape().ndim(); ++i)
    a_pad[ndim - a.shape().ndim() + i] = a.shape().dim(i);
  for (int i = 0; i < b.shape().ndim(); ++i)
    b_pad[ndim - b.shape().ndim() + i] = b.shape().dim(i);

  // Interleave: a -> [a0,1,a1,1,...], b -> [1,b0,1,b1,...]
  std::vector<int64_t> a_inter, b_inter;
  for (int i = 0; i < ndim; ++i) {
    a_inter.push_back(a_pad[i]);
    a_inter.push_back(1);
    b_inter.push_back(1);
    b_inter.push_back(b_pad[i]);
  }
  Array a2 = reshape(a, Shape(a_inter));
  Array b2 = reshape(b, Shape(b_inter));

  Array prod = mul(a2, b2); // broadcasting gives [a0,b0,a1,b1,...]

  std::vector<int64_t> out_dims;
  for (int i = 0; i < ndim; ++i)
    out_dims.push_back(a_pad[i] * b_pad[i]);
  return reshape(prod, Shape(out_dims));
}

// ============================================================================
// cross - Cross product of 3-vectors along an axis
// ============================================================================

Array cross(const Array &a, const Array &b, int axis) {
  INS_CHECK(a.defined() && b.defined(), "cross: input is undefined");
  INS_CHECK(a.shape() == b.shape(), "cross: inputs must have the same shape");

  int ax = normalize_axis(axis, a.shape().ndim(), "cross");
  INS_CHECK(a.shape().dim(ax) == 3, "cross: axis must have size 3");

  Array a0 = a.slice(ax, 0, 1);
  Array a1 = a.slice(ax, 1, 2);
  Array a2 = a.slice(ax, 2, 3);
  Array b0 = b.slice(ax, 0, 1);
  Array b1 = b.slice(ax, 1, 2);
  Array b2 = b.slice(ax, 2, 3);

  // c = (a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0)
  Array c0 = sub(mul(a1, b2), mul(a2, b1));
  Array c1 = sub(mul(a2, b0), mul(a0, b2));
  Array c2 = sub(mul(a0, b1), mul(a1, b0));

  return concat({c0, c1, c2}, ax);
}

// ============================================================================
// multi_dot - Optimal-order chain matrix multiplication
// ============================================================================

Array multi_dot(const std::vector<Array> &arrays) {
  INS_CHECK(arrays.size() >= 2, "multi_dot: need at least two matrices");

  int n = static_cast<int>(arrays.size());
  std::vector<Array> works(n);
  DType work_dtype = arrays[0].dtype();
  for (size_t i = 1; i < arrays.size(); ++i)
    work_dtype = promote_types(work_dtype, arrays[i].dtype());

  Place place = arrays[0].place();
  for (size_t i = 0; i < arrays.size(); ++i) {
    INS_CHECK(arrays[i].shape().ndim() == 2,
              "multi_dot: all inputs must be 2D");
    works[i] = (arrays[i].dtype() == work_dtype) ? arrays[i]
                                                 : arrays[i].to(work_dtype);
    if (works[i].place() != place) {
      works[i] = works[i].to(place);
    }
  }
  for (size_t i = 1; i < works.size(); ++i) {
    INS_CHECK(works[i - 1].shape().dim(1) == works[i].shape().dim(0),
              "multi_dot: incompatible matrix dimensions");
  }

  // Dimensions chain: p[0] x p[1] x ... x p[n].
  std::vector<int64_t> p(n + 1);
  p[0] = works[0].shape().dim(0);
  for (int i = 0; i < n; ++i)
    p[i + 1] = works[i].shape().dim(1);

  // DP for optimal parenthesization.
  std::vector<std::vector<int64_t>> cost(n, std::vector<int64_t>(n, 0));
  std::vector<std::vector<int>> split(n, std::vector<int>(n, -1));
  for (int len = 2; len <= n; ++len) {
    for (int i = 0; i <= n - len; ++i) {
      int j = i + len - 1;
      cost[i][j] = std::numeric_limits<int64_t>::max();
      for (int k = i; k < j; ++k) {
        int64_t c = cost[i][k] + cost[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
        if (c < cost[i][j]) {
          cost[i][j] = c;
          split[i][j] = k;
        }
      }
    }
  }

  // Recursively multiply using the optimal split.
  std::function<Array(int, int)> solve = [&](int i, int j) -> Array {
    if (i == j)
      return works[i];
    int k = split[i][j];
    return matmul(solve(i, k), solve(k + 1, j));
  };

  Array result = solve(0, n - 1);
  if (result.dtype() != work_dtype) {
    result = result.to(work_dtype);
  }
  return result;
}

// ============================================================================
// einsum - Einstein summation
// ============================================================================

namespace {

// Parsed subscript: per-operand label lists plus output labels.
struct EinsumSpec {
  std::vector<std::vector<char>> terms;
  std::vector<char> output;
  bool explicit_output = false;
};

// Parse "ij,jk->ik" into terms {"ij","jk"} and output "ik".
EinsumSpec parse_einsum(const std::string &subscripts, size_t operand_count) {
  EinsumSpec spec;
  size_t arrow = subscripts.find("->");
  std::string lhs =
      (arrow == std::string::npos) ? subscripts : subscripts.substr(0, arrow);
  std::string rhs =
      (arrow == std::string::npos) ? "" : subscripts.substr(arrow + 2);
  spec.explicit_output = (arrow != std::string::npos);

  size_t start = 0;
  while (start <= lhs.size()) {
    size_t comma = lhs.find(',', start);
    if (comma == std::string::npos)
      comma = lhs.size();
    std::string term = lhs.substr(start, comma - start);
    if (!term.empty()) {
      std::vector<char> labels;
      for (char c : term) {
        if (c == '.')
          INS_THROW("einsum: ellipsis '...' is not supported yet");
        labels.push_back(c);
      }
      spec.terms.push_back(labels);
    }
    if (comma == lhs.size())
      break;
    start = comma + 1;
  }

  for (char c : rhs) {
    if (c == '.')
      INS_THROW("einsum: ellipsis '...' is not supported yet");
    spec.output.push_back(c);
  }

  INS_CHECK(spec.terms.size() == operand_count,
            "einsum: operand count does not match subscripts");
  for (size_t i = 0; i < spec.terms.size(); ++i) {
    // Repeated labels within one operand (diagonal) are not supported.
    std::vector<char> seen = spec.terms[i];
    std::sort(seen.begin(), seen.end());
    for (size_t k = 1; k < seen.size(); ++k) {
      if (seen[k] == seen[k - 1])
        INS_THROW("einsum: repeated labels within one operand are not "
                  "supported (diagonal)");
    }
  }

  if (!spec.explicit_output) {
    // Implicit output: labels appearing exactly once, in alphabetical order.
    std::map<char, int> count;
    for (const auto &term : spec.terms) {
      for (char c : term)
        count[c]++;
    }
    for (const auto &kv : count) {
      if (kv.second == 1)
        spec.output.push_back(kv.first);
    }
  }
  return spec;
}

// Common matmul fast path for exactly two operands with a shared
// contraction set and no broadcast overlap in the output labels.
bool einsum_try_matmul(const EinsumSpec &spec, const std::vector<Array> &ops,
                       Array &out) {
  if (spec.terms.size() != 2)
    return false;

  const std::vector<char> &la = spec.terms[0];
  const std::vector<char> &lb = spec.terms[1];

  // Common labels (contracted unless they also appear in the output).
  std::vector<char> common;
  for (char c : la) {
    if (std::find(lb.begin(), lb.end(), c) != lb.end() &&
        std::find(common.begin(), common.end(), c) == common.end())
      common.push_back(c);
  }
  // If any common label appears in the output, broadcasting is needed ->
  // fall back to the generic path.
  for (char c : common) {
    if (std::find(spec.output.begin(), spec.output.end(), c) !=
        spec.output.end())
      return false;
  }

  // Keep labels (in operand order).
  std::vector<char> keep_a, keep_b;
  for (char c : la) {
    if (std::find(common.begin(), common.end(), c) == common.end())
      keep_a.push_back(c);
  }
  for (char c : lb) {
    if (std::find(common.begin(), common.end(), c) == common.end())
      keep_b.push_back(c);
  }

  // Common axes in each operand.
  std::vector<int> common_axes_a, common_axes_b;
  for (char c : common) {
    common_axes_a.push_back(
        static_cast<int>(std::find(la.begin(), la.end(), c) - la.begin()));
    common_axes_b.push_back(
        static_cast<int>(std::find(lb.begin(), lb.end(), c) - lb.begin()));
  }
  // Contracted dimensions must match.
  for (size_t i = 0; i < common.size(); ++i) {
    if (ops[0].shape().dim(common_axes_a[i]) !=
        ops[1].shape().dim(common_axes_b[i]))
      return false;
  }

  DType work_dtype = promote_types(ops[0].dtype(), ops[1].dtype());
  Array a_work =
      (ops[0].dtype() == work_dtype) ? ops[0] : ops[0].to(work_dtype);
  Array b_work =
      (ops[1].dtype() == work_dtype) ? ops[1] : ops[1].to(work_dtype);
  if (a_work.place() != b_work.place()) {
    b_work = b_work.to(a_work.place());
  }

  // a: [keep_a..., common...]
  std::vector<int> perm_a;
  for (char c : keep_a)
    perm_a.push_back(
        static_cast<int>(std::find(la.begin(), la.end(), c) - la.begin()));
  for (char c : common)
    perm_a.push_back(
        static_cast<int>(std::find(la.begin(), la.end(), c) - la.begin()));
  Array a_p = permute(a_work, perm_a);

  // b: [common..., keep_b...]
  std::vector<int> perm_b;
  for (char c : common)
    perm_b.push_back(
        static_cast<int>(std::find(lb.begin(), lb.end(), c) - lb.begin()));
  for (char c : keep_b)
    perm_b.push_back(
        static_cast<int>(std::find(lb.begin(), lb.end(), c) - lb.begin()));
  Array b_p = permute(b_work, perm_b);

  int64_t prod_keep_a = 1, prod_contract = 1, prod_keep_b = 1;
  std::vector<int64_t> keep_a_dims, keep_b_dims;
  for (size_t i = 0; i < keep_a.size(); ++i)
    keep_a_dims.push_back(a_p.shape().dim(i));
  for (size_t i = 0; i < common.size(); ++i)
    prod_contract *= a_p.shape().dim(keep_a.size() + i);
  for (size_t i = 0; i < keep_b.size(); ++i)
    keep_b_dims.push_back(b_p.shape().dim(common.size() + i));
  for (int64_t d : keep_a_dims)
    prod_keep_a *= d;
  for (int64_t d : keep_b_dims)
    prod_keep_b *= d;

  Array a2 = reshape(a_p, Shape({prod_keep_a, prod_contract}));
  Array b2 = reshape(b_p, Shape({prod_contract, prod_keep_b}));
  Array prod = matmul(a2, b2); // (prod_keep_a, prod_keep_b)

  std::vector<int64_t> mid_dims(keep_a_dims);
  mid_dims.insert(mid_dims.end(), keep_b_dims.begin(), keep_b_dims.end());
  Array mid = reshape(prod, Shape(mid_dims));

  // Sum over keep labels that are not in the output.
  std::vector<char> keep_all(keep_a);
  keep_all.insert(keep_all.end(), keep_b.begin(), keep_b.end());

  // Surviving labels (those in the output) keep their keep_all order.
  std::vector<char> surviving;
  std::vector<int64_t> surviving_dims;
  for (size_t i = 0; i < keep_all.size(); ++i) {
    if (std::find(spec.output.begin(), spec.output.end(), keep_all[i]) !=
        spec.output.end()) {
      surviving.push_back(keep_all[i]);
      surviving_dims.push_back(mid.shape().dim(i));
    }
  }

  Array res = mid;
  // Reduce from the back so axis indices stay valid.
  for (int i = static_cast<int>(keep_all.size()) - 1; i >= 0; --i) {
    if (std::find(spec.output.begin(), spec.output.end(), keep_all[i]) ==
        spec.output.end()) {
      res = sum(res, i, false);
    }
  }

  // Permute surviving axes (in keep_all order) to output label order.
  std::vector<int> perm_out;
  std::vector<int64_t> out_dims;
  for (char c : spec.output) {
    auto it = std::find(surviving.begin(), surviving.end(), c);
    if (it == surviving.end())
      return false; // output label not produced by the matmul path
    int pos = static_cast<int>(it - surviving.begin());
    perm_out.push_back(pos);
    out_dims.push_back(surviving_dims[pos]);
  }
  Array permuted = (!perm_out.empty() && perm_out.size() == res.shape().ndim())
                       ? permute(res, perm_out)
                       : res;
  out = reshape(permuted, Shape(out_dims));
  if (out.dtype() != work_dtype) {
    out = out.to(work_dtype);
  }
  return true;
}

} // namespace

Array einsum(const std::string &subscripts,
             const std::vector<Array> &operands) {
  INS_CHECK(!operands.empty(), "einsum: no operands provided");
  for (const Array &op : operands) {
    INS_CHECK(op.defined(), "einsum: input is undefined");
  }

  EinsumSpec spec = parse_einsum(subscripts, operands.size());
  for (size_t i = 0; i < spec.terms.size(); ++i) {
    INS_CHECK(spec.terms[i].size() ==
                  static_cast<size_t>(operands[i].shape().ndim()),
              "einsum: subscript labels must match operand ndim");
  }

  Array result;
  if (einsum_try_matmul(spec, operands, result)) {
    return result;
  }

  // ---- Generic path: broadcast all operands to the full label space,
  // multiply, then sum over non-output labels and permute to output order.

  // Global label order: output labels first (in order), then remaining
  // labels in order of first appearance.
  std::vector<char> global;
  for (char c : spec.output) {
    if (std::find(global.begin(), global.end(), c) == global.end())
      global.push_back(c);
  }
  for (const auto &term : spec.terms) {
    for (char c : term) {
      if (std::find(global.begin(), global.end(), c) == global.end())
        global.push_back(c);
    }
  }

  // Full shape over all global labels (1 where an operand lacks the label).
  std::vector<int64_t> full_shape(global.size(), 1);
  for (size_t t = 0; t < spec.terms.size(); ++t) {
    for (size_t i = 0; i < spec.terms[t].size(); ++i) {
      char c = spec.terms[t][i];
      int gi = static_cast<int>(std::find(global.begin(), global.end(), c) -
                                global.begin());
      int64_t d = operands[t].shape().dim(i);
      if (full_shape[gi] != 1 && full_shape[gi] != d)
        INS_THROW("einsum: label '", std::string(1, c),
                  "' has incompatible dimensions");
      full_shape[gi] = d;
    }
  }

  DType work_dtype = operands[0].dtype();
  for (size_t i = 1; i < operands.size(); ++i)
    work_dtype = promote_types(work_dtype, operands[i].dtype());

  Array acc;
  for (size_t t = 0; t < spec.terms.size(); ++t) {
    // Permute operand axes to the global label order.
    std::vector<int> perm;
    for (size_t gi = 0; gi < global.size(); ++gi) {
      char c = global[gi];
      auto it = std::find(spec.terms[t].begin(), spec.terms[t].end(), c);
      if (it != spec.terms[t].end()) {
        perm.push_back(static_cast<int>(it - spec.terms[t].begin()));
      }
    }
    // Missing labels are size-1; reshape inserts them. term_full starts
    // from all-1s (NOT from full_shape, which mixes in other operands'
    // dims and would inflate the element count).
    std::vector<int64_t> term_full(global.size(), 1);
    for (size_t i = 0; i < spec.terms[t].size(); ++i) {
      char c = spec.terms[t][i];
      int gi = static_cast<int>(std::find(global.begin(), global.end(), c) -
                                global.begin());
      term_full[gi] = operands[t].shape().dim(i);
    }
    Array t_work = (operands[t].dtype() == work_dtype)
                       ? operands[t]
                       : operands[t].to(work_dtype);
    if (!acc.defined()) {
      acc = reshape(permute(t_work, perm), Shape(term_full));
    } else {
      if (acc.place() != t_work.place()) {
        t_work = t_work.to(acc.place());
      }
      acc = mul(acc, reshape(permute(t_work, perm), Shape(term_full)));
    }
  }

  // Broadcast to the full shape (mul already broadcast-combined labels, but
  // ensure any label present in only some operands is expanded).
  acc = broadcast_to(acc, Shape(full_shape));

  // Sum over labels not in the output.
  for (int gi = static_cast<int>(global.size()) - 1; gi >= 0; --gi) {
    if (std::find(spec.output.begin(), spec.output.end(), global[gi]) ==
        spec.output.end()) {
      acc = sum(acc, gi, false);
    }
  }

  // Permute to output label order.
  std::vector<int> perm_out;
  std::vector<int64_t> out_dims;
  for (char c : spec.output) {
    int gi = static_cast<int>(std::find(global.begin(), global.end(), c) -
                              global.begin());
    perm_out.push_back(gi);
    out_dims.push_back(full_shape[gi]);
  }
  Array permuted = (!perm_out.empty() && perm_out.size() == acc.shape().ndim())
                       ? permute(acc, perm_out)
                       : acc;
  Array result2 = reshape(permuted, Shape(out_dims));
  if (result2.dtype() != work_dtype) {
    result2 = result2.to(work_dtype);
  }
  return result2;
}

} // namespace ins