// src/ops/signal/estimation.cpp
#include "insight/ops/signal/estimation.h"
#include "insight/core/exception.h"
#include "insight/core/op_registry.h"
#include "insight/core/op_schema.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/linalg.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/operator.h"
#include "insight/ops/reduction.h"
#include <cmath>
#include <string>
#include <vector>

namespace ins {
namespace signal {

namespace {

static OpSchema signal_schema(const char *name, size_t input_count,
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

  return OpSchema(name, OpKind::Signal, inputs, outputs)
      .promotion(OpPromotionRule::Identity)
      .fallback(OpFallbackRule::StructuredCpu);
}

static void launch_signal_kernel(const char *name, const Place &place,
                                 DType dtype,
                                 const std::vector<Array> &inputs,
                                 const std::vector<Array> &outputs) {
  OpSchema schema = signal_schema(name, inputs.size(), outputs.size());
  ArrayIterator iter = schema.make_transform_iterator(inputs, outputs);
  auto arrays = iter.arrays();

  std::vector<OpRegistry::Arg> launch_inputs;
  launch_inputs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    launch_inputs.push_back(OpRegistry::array_arg(arrays[i].layout_ptr()));
  }

  std::vector<OpRegistry::Arg> launch_outputs;
  launch_outputs.reserve(outputs.size());
  for (size_t i = inputs.size(); i < arrays.size(); ++i) {
    launch_outputs.push_back(OpRegistry::array_arg(arrays[i].layout_ptr()));
  }

  OpRegistry::launch_schema(name, place, dtype, launch_inputs, launch_outputs);
}

static void launch_signal_kernel(const char *name, const Place &place,
                                 DType dtype,
                                 const std::vector<Array> &inputs,
                                 const Array &output) {
  launch_signal_kernel(name, place, dtype, inputs, std::vector<Array>{output});
}

// Batched transpose: transpose last two dimensions of a 3D array
Array batched_transpose(const Array &x, int points, int rows, int cols) {
  return permute(x, {0, 2, 1});
}

// Build batched identity matrix: [points, n, n] with 1s on diagonal
Array batched_eye(int points, int n, DType dtype, const Place &place) {
  Array I_single = eye(n, n, 0, dtype, place);
  std::vector<Array> copies(points, I_single);
  return stack(copies, 0);
}

// Simple Gauss-Jordan inverse for small matrices (backend kernel)
Array simple_inv(const Array &mat) {
  Place cpu = CPUPlace();
  Array m = (mat.place().kind() == DeviceKind::CPU) ? mat : mat.to(cpu);
  DType dtype = m.dtype();
  int n = m.shape().dim(0);

  // Ensure F64 for the kernel (kernel expects double)
  if (dtype != DType::F64)
    m = m.to(DType::F64);

  Array out = zeros({n, n}, DType::F64, cpu);
  launch_signal_kernel("simple_inv", cpu, DType::F64, {m}, out);

  if (dtype != DType::F64)
    out = out.to(dtype);
  return out;
}

} // anonymous namespace

KalmanFilter::KalmanFilter(int dim_x, int dim_z, int dim_u, int points,
                           DType dtype)
    : dim_x(dim_x), dim_z(dim_z), dim_u(dim_u), points(points), dtype_(dtype),
      place_(CPUPlace()) {
  INS_CHECK(dim_x > 0, "KalmanFilter: dim_x must be > 0");
  INS_CHECK(dim_z > 0, "KalmanFilter: dim_z must be > 0");
  INS_CHECK(points > 0, "KalmanFilter: points must be > 0");

  // Initialize state arrays
  x = zeros(Shape({points, dim_x, 1}), dtype_, place_);
  P = zeros(Shape({points, dim_x, dim_x}), dtype_, place_);
  z = zeros(Shape({points, dim_z, 1}), dtype_, place_);
  R = zeros(Shape({points, dim_z, dim_z}), dtype_, place_);
  Q = zeros(Shape({points, dim_x, dim_x}), dtype_, place_);
  F = batched_eye(points, dim_x, dtype_, place_);
  H = zeros(Shape({points, dim_z, dim_x}), dtype_, place_);

  // Set H to identity-like: H[i,i] = 1 for i < min(dim_z, dim_x)
  Array H_eye = eye(dim_z, dim_x, 0, dtype_, place_);
  // Copy to each point
  for (int p = 0; p < points; ++p) {
    // H[p] = H_eye (via slice assignment)
    Array H_slice = slice(H, {0}, {p}, {p + 1});
    H_slice = reshape(H_slice, {dim_z, dim_x});
    // Copy H_eye into H_slice
    Array H_eye_c = H_eye;
    if (H_eye_c.place() != H_slice.place())
      H_eye_c = H_eye_c.to(H_slice.place());
    if (H_eye_c.dtype() != H_slice.dtype())
      H_eye_c = H_eye_c.to(H_slice.dtype());
    // Use put or direct copy
    // For simplicity, rebuild H with eye values
  }

  // Rebuild H properly using eye + stack
  {
    std::vector<Array> H_parts;
    for (int p = 0; p < points; ++p) {
      H_parts.push_back(eye(dim_z, dim_x, 0, dtype_, place_));
    }
    H = stack(H_parts, 0);
  }
}

// Ensure a matrix is 3D [points, rows, cols] by broadcasting 2D -> 3D
static Array ensure_3d(const Array &mat, int points) {
  if (mat.shape().ndim() == 2) {
    int rows = mat.shape().dim(0);
    int cols = mat.shape().dim(1);
    return reshape(mat, {1, rows, cols});
  }
  return mat;
}

void KalmanFilter::predict() {
  // Ensure F is 3D (broadcast 2D -> [1, dim_x, dim_x])
  Array F3 = ensure_3d(F, points);

  // x = F * x
  Array new_x = matmul(F3, x);

  // P = F * P * F^T + Q
  Array FP = matmul(F3, P);
  Array FT = batched_transpose(F3, points, dim_x, dim_x);
  Array P_new = add(matmul(FP, FT), Q);

  // Update state
  x = new_x;
  P = P_new;
}

void KalmanFilter::update(const Array &z_in) {
  INS_CHECK(z_in.defined(), "update: measurement z is undefined");

  Place cpu = CPUPlace();
  Array z_cpu = (z_in.place().kind() == DeviceKind::CPU) ? z_in : z_in.to(cpu);

  // Ensure z is 3D [points, dim_z, 1]
  z = ensure_3d(z_cpu, points);

  // Ensure H is 3D (broadcast 2D -> [1, dim_z, dim_x])
  Array H3 = ensure_3d(H, points);

  // Innovation: y = z - H * x
  Array Hx = matmul(H3, x);
  Array y = sub(z, Hx);

  // Innovation covariance: S = H * P * H^T + R
  Array HP = matmul(H3, P);
  Array HT = batched_transpose(H3, points, dim_x, dim_z);
  Array S = add(matmul(HP, HT), R);

  // Regularize S to avoid singular matrix when P and R are zero-initialized
  Array eps_I = batched_eye(points, dim_z, S.dtype(), S.place());
  // Scale identity by epsilon using element-wise multiplication with scalar
  Array eps_scalar = full(S.shape(), 1e-6, S.dtype(), S.place());
  S = add(S, mul(eps_I, eps_scalar));

  // Kalman gain: K = P * H^T * inv(S)
  Array P_Ht = matmul(P, HT); // [points, dim_x, dim_z]

  // Compute inv(S) for each point using simple Gauss-Jordan (no LAPACK)
  std::vector<Array> S_inv_parts;
  for (int p = 0; p < points; ++p) {
    Array S_p = slice(S, {0}, {p}, {p + 1});
    S_p = reshape(S_p, {dim_z, dim_z});
    Array inv_p = simple_inv(S_p);
    S_inv_parts.push_back(inv_p);
  }
  Array S_inv = stack(S_inv_parts, 0);

  // K = P * H^T * S_inv  [points, dim_x, dim_z]
  Array K = matmul(P_Ht, S_inv);

  // x = x + K * y
  Array Ky = matmul(K, y);
  x = add(x, Ky);

  // P = (I - K*H) * P
  Array KH = matmul(K, H3);
  Array I_x = batched_eye(points, dim_x, dtype_, cpu);
  if (KH.place() != I_x.place())
    I_x = I_x.to(KH.place());
  Array I_minus_KH = sub(I_x, KH);
  P = matmul(I_minus_KH, P);
}

} // namespace signal
} // namespace ins
