# Insight Architecture v2

This document is the working contract for the next Insight7 architecture pass. It is deliberately written before large code movement so future agents can make local changes without rediscovering the project direction.

## Goals

Insight is the NumPy-like array layer for the Paddle-Lua stack. It should remain lightweight, language-bindable, and backend-aware, while making its core protocols explicit enough to support CPU/GPU execution, zero-copy views, and interop.

The target architecture borrows from:

- PaddlePaddle for public API behavior where Paddle intentionally differs from PyTorch.
- NumPy for ndarray layout, dtype/promotion, ufunc dispatch, and interoperability protocols.
- CuPy for GPU memory, stream-aware DLPack, explicit host/device conversion, and kernel layering.

## Non-Goals

- Do not add autograd.
- Do not make Insight a PyTorch-compatible API surface.
- Do not replace the CPU/GPU runtime model only to support theoretical simultaneous CUDA+ROCm+other GPU execution. The selected GPU backend is effectively singular in practical training and cloud compute deployments.
- Do not expand operator count before the storage, dtype, dispatch, and binding protocols are coherent.

## Runtime Device Model

Keep the public runtime shape as two device kinds:

```cpp
enum class DeviceKind {
  CPU,
  GPU,
};
```

Multiple GPU implementations may exist as build artifacts or loadable plugins, but process runtime should register one active GPU backend. CUDA/ROCm/Iluvatar should not compete inside a generalized multi-vendor device table until a real use case appears.

Required cleanup inside this model:

- Make backend selection explicit during initialization.
- Record the active GPU backend name for diagnostics.
- Avoid hardcoding `GPUPlace(0)` in fallback or copy paths; use the actual source/output place.
- Keep `cpu:0` and `gpu:N` as stable user-facing places.

## Array Protocol

The core array must be modeled as a small value object pointing at explicit storage metadata:

```text
Array
  ├─ ArrayLayout: shape, strides, offset, dtype, place, flags
  └─ Storage: data pointer, byte size, owning place, deleter, reference count
```

The current `InsightArray` combines layout, data pointer, device fields, view state, and manual refcount. v2 should separate these ideas even if the C ABI remains source-compatible for a transition period.

### Required Semantics

- A root array owns a storage allocation.
- A view shares the same storage and changes only shape, strides, and offset.
- `data()` returns the address of the first logical element, including offset.
- `storage_data()` or equivalent returns the base allocation pointer.
- `nbytes()` for logical array size is distinct from storage allocation size.
- Refcount must be safe for binding/language lifetimes and future threaded use.
- C ABI comments and implementation must agree on who deallocates data.

### Current Risks To Remove

- `InsightArray` comments say data is externally owned, while `Array::~Array()` deallocates it.
- `ref_count` is a raw `int32_t*` and increments are not atomic.
- View byte calculations use logical `numel * dtype_size`, which is not the same as storage allocation size.
- Some backend kernels mutate output layout/refcount directly for dynamic-shape outputs.

## DType Protocol

The current enum is useful for built-in fast paths, but v2 should treat dtype as a descriptor concept:

```text
DTypeId       compact enum for built-ins and ABI dispatch
DTypeDescr    name, kind, itemsize, alignment, flags, scalar class
CastRule      source dtype, destination dtype, casting safety, implementation
PromotionRule input dtype set + scalar categories -> result dtype
```

### Required Semantics

- Built-in dtype names and aliases should match Paddle where Paddle exposes them.
- Promotion must be centralized, not reimplemented per op.
- Cast safety must be explicit.
- Complex, bool, integer, float16/bfloat16, and float8 support should remain first-class.
- Future custom/parametric dtype support should not require rewriting every op.

## Operator And Kernel Protocol

The current kernel ABI passes null-terminated `void**` inputs/outputs, with scalar parameters mixed into the same channel. This is too implicit.

v2 should introduce an op schema layer:

```text
OpSchema
  name
  inputs: Array parameters
  outputs: Array parameters
  attrs: typed scalar/list/string parameters
  shape inference
  dtype inference / promotion
  dispatch keys
```

```text
KernelSignature
  op name
  backend kind: CPU or GPU
  dtype loop key
  typed array args
  typed attrs
```

### Required Semantics

- Shape inference happens before allocation.
- Type promotion happens before kernel dispatch.
- Scalar attributes are not guessed from `void*`.
- Kernel arity and argument types are inspectable.
- CPU fallback receives a structured invocation, not a guessed list of raw pointers.
- Fallback must preserve strides, offset, logical shape, and target device.

### Current Risks To Remove

- CPU fallback scans raw pointers and guesses which are `InsightArray*`.
- Fallback hardcodes `GPUPlace(0)`.
- Fallback copies `numel * dtype_size` from `data`, which is wrong for some non-contiguous views and offsets.
- Kernel registration is keyed only by `op|device|dtype`, with no schema or attr contract.

## Binding Semantics

C++ core should use internal zero-based axes and row-major logical layout. Language bindings own user-facing convention conversion.

### Python

- Python indexing and axes are zero-based.
- Python API should follow Paddle where Paddle intentionally differs from PyTorch.
- NumPy conversion must keep backing storage alive through a base object or capsule.
- Device arrays should not silently become host NumPy arrays unless the API name clearly implies a copy, e.g. `.numpy()` / `.get()`.

### Lua

- Lua user-facing axes and positive integer indexing are one-based.
- Negative axes and negative indices pass through Python-style from-end semantics.
- Conversion to C++ core should happen in Lua wrappers or a shared Lua binding helper before calling native zero-based APIs.
- Named/table calls should use the argrule direction, not ad hoc wrappers long-term.

### Julia

- Julia axes are one-based.
- Julia column-major convention must be handled at the binding boundary.
- Shape reversal must be centralized and applied consistently to constructors, shape access, indexing, reductions, and data import/export.

## Interop Protocols

Interop is part of the core design, not a wrapper afterthought.

### Required Protocols

- DLPack export/import for CPU and GPU arrays.
- Python `__dlpack__` and `__dlpack_device__`.
- Python `__array_interface__` for CPU arrays where lifetime can be guaranteed.
- Python `__cuda_array_interface__` for GPU arrays if stream semantics are explicit.
- Explicit host copy APIs for GPU arrays, following CuPy's stance against accidental implicit device-to-host conversion.

### Lifetime Rule

Any exported view must retain the owning storage until the consumer releases it. Returning a NumPy array that points to a local temporary is forbidden.

## API Direction

Insight is part of the Paddle-Lua ecosystem. When naming/defaults differ across NumPy, CuPy, Paddle, and PyTorch:

1. Use NumPy/CuPy to guide array protocol and implementation structure.
2. Use Paddle to guide public API behavior where Paddle intentionally differs from PyTorch.
3. Avoid PyTorch-only names or semantics unless there is no Paddle/NumPy equivalent and the feature is clearly useful.

## Migration Plan

### Phase 0: Safety Fixes

- Fix Python `to_numpy()` lifetime.
- Make C ABI ownership comments match implementation.
- Add helper functions for Lua axis conversion and apply them to high-risk wrappers.
- Document CPU/GPU two-slot backend decision.

### Phase 1: Storage Refactor

- Introduce explicit storage metadata in C++.
- Replace raw `int32_t* ref_count` with a safer ownership object.
- Preserve C ABI compatibility or add a v2 ABI alongside v1.
- Add tests for root/view destruction, assignment, slicing, and cross-binding lifetime.

### Phase 2: Axis/Indexing Contract

- Add shared core helpers for axis normalization.
- Move language convention conversion into binding helpers.
- Add Python/Lua/Julia axis alignment tests for reductions, manipulation, indexing, and signal ops.

### Phase 3: Kernel Schema

- Add `OpSchema` and typed attr plumbing.
- Migrate fallback-sensitive ops first: reductions, indexing, gather/scatter, unique/nonzero.
- Replace heuristic CPU fallback with structured fallback.

### Phase 4: DType Descriptor And Promotion

- Centralize promotion and casting policy.
- Keep enum fast paths for built-ins.
- Add conformance tests against NumPy/CuPy/Paddle where appropriate.

### Phase 5: Interop

- Implement DLPack.
- Fix Python NumPy/CUDA interop around lifetime and stream semantics.
- Add round-trip tests with NumPy and CuPy.

## Acceptance Criteria

- No dangling memory in Python NumPy export.
- Views preserve storage lifetime and write-through semantics.
- Lua positive axes are one-based consistently at binding boundary.
- Python axes remain zero-based.
- Julia shape/axis conversion is centralized.
- CPU fallback does not guess raw pointer types.
- DType promotion behavior is centralized and testable.
- Public docs explicitly state Paddle alignment and non-PyTorch goals.
