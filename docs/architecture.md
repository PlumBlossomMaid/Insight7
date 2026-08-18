# Insight Architecture

This document is the repository-tracked architecture contract for Insight7. It records the current design decisions after the Array/core/backend refactor so future work can continue from shared source-controlled context instead of local notes.

## Goals

Insight is the NumPy-like array layer for the Paddle-Lua stack. It should remain lightweight, language-bindable, and backend-aware, while making its core protocols explicit enough to support CPU/GPU execution, zero-copy views, interop, and vendor GPU backends such as CUDA, ROCm, IXUCA, Ascend, and Tecorigin SDAA.

The target architecture borrows from:

- PaddlePaddle for public API behavior where Paddle intentionally differs from PyTorch.
- NumPy for ndarray layout, dtype/promotion, ufunc dispatch, strided iteration, and interoperability protocols.
- CuPy for GPU memory, stream-aware DLPack, explicit host/device conversion, selected GPU backend behavior, kernel caching, and accelerator layering.

The core direction is:

```text
NumPy-like semantics
  + CuPy-like GPU execution model
  + Paddle-style backend/HAL discipline
  + Lua-generated repetitive operator glue
  + language-neutral C++/C ABI core
```

## Non-Goals

- Do not add autograd.
- Do not make Insight a PyTorch-compatible API surface.
- Do not replace the CPU/GPU runtime model only to support theoretical simultaneous CUDA+ROCm+SDAA+other GPU execution. The selected GPU backend is effectively singular in practical training, inference, and cloud compute deployments.
- Do not expose every vendor backend as a public device kind until a real multi-backend-in-one-process use case appears.
- Do not expand operator count before the storage, dtype, dispatch, and binding protocols are coherent.
- Do not hand-write a full new SDAA operator tree before the op schema, ArrayIterator, and backend selection protocols exist.

## Runtime Device Model

Keep the public runtime shape as two device kinds:

```cpp
enum class DeviceKind {
  CPU,
  GPU,
};
```

Multiple GPU implementations may exist as build artifacts or loadable plugins, but process runtime should register one active GPU backend. CUDA/ROCm/IXUCA/Ascend/SDAA should not compete inside a generalized multi-vendor public device table until a real use case appears.

The public user-facing device namespace remains stable:

```text
cpu:0
gpu:0
gpu:1
```

The backend implementation selected behind `gpu:*` is an internal runtime property:

```text
active_gpu_backend = cuda | rocm | ixuca | ascend | sdaa | ...
```

### Backend Selection

Backend selection must be explicit and diagnosable:

- `ins::init()` may auto-discover a GPU backend when no preference is supplied.
- `ins::init(InitOptions{.gpu_backend = "sdaa"})` or equivalent should force a backend.
- `INSIGHT_GPU_BACKEND=sdaa` should be supported for CLI/demo/binding convenience.
- If multiple GPU backend shared libraries are present, initialization should choose exactly one active GPU backend and report the decision.
- Diagnostics should expose `active_gpu_backend_name()`, backend version, device count, and useful capability flags.
- Errors must say `gpu backend=sdaa` or `gpu backend=cuda`, not only `GPU`.

### Required Cleanup Inside This Model

- Make backend selection explicit during initialization.
- Record the active GPU backend name for diagnostics.
- Avoid hardcoding `GPUPlace(0)` in fallback or copy paths; use the actual source/output place.
- Keep `cpu:0` and `gpu:N` as stable user-facing places.
- Keep the C ABI `INSIGHT_DEVICE_CPU` / `INSIGHT_DEVICE_GPU` stable while evolving the framework contract.
- Store backend-specific state behind the GPU device interface, not in public `Place`.

## GPU Backend Contract

A GPU backend is the implementation selected behind `DeviceKind::GPU`. It must provide a complete enough runtime interface for arrays, copies, streams, events, kernels, profiling, and diagnostics.

```text
GpuBackendDescriptor
  name: cuda | rocm | ixuca | ascend | sdaa | ...
  version/runtime/driver strings
  device count and device names
  capabilities: fp16, bf16, complex, blas, fft, random, matrix engines/custom accelerators
  device interface: allocation, copy, memset, stream, event, profiler
  kernel registry: op schema + dtype loop key -> kernel implementation
```

Required principles:

- The public `GPUPlace(i)` means device `i` of the active GPU backend.
- Only one active GPU backend is registered in a process.
- Backends may share generated code where practical, but vendor-specific code must remain isolated.
- CUDA-to-HIP transformation is acceptable for ROCm only where the generated source remains inspectable and testable.
- SDAA/Ascend-style backends should use their native runtime and library handles instead of pretending to be CUDA.

## SDAA Backend Direction

Tecorigin SDAA is implemented as one active GPU backend implementation, not as a third public device kind. The current runtime backend covers device discovery, memory/copy, stream/event hooks, diagnostics, profiler plumbing, kernel registration, and structured CPU fallback.

Current layout:

```text
backends/sdaa/
  CMakeLists.txt
  init.cpp
  device/sdaa_device.cpp
  registry/sdaa_registry.{h,cpp}
  kernels/
  generated/
```

SDAA runtime dependencies should be discovered from environment variables and CMake cache entries, following the Paddle CustomDevice SDAA pattern:

- `SDAA_ROOT` for `sdaa_runtime.h` and runtime library.
- `TECODNN_ROOT` for `tecodnn`.
- `TBLAS_ROOT` for `tecoblas`.
- `TECOCUSTOM_ROOT` for `tecocustom`.
- `TECOCUSTOM_EXT_ROOT` for custom DNN extension APIs if needed.
- `SDPTI_ROOT` for profiling if profiling support is enabled.

Minimum SDAA runtime surface:

- device count, set/get device, synchronize device
- device malloc/free/memset
- H2D/D2H/D2D copy and async copy when available
- stream create/destroy/synchronize/query
- event create/destroy/record/query/synchronize/elapsed time
- device memory stats and device name
- backend last-error reporting
- optional profiler integration through SDPTI

SDAA stream state should bundle vendor library handles where the SDK expects stream-bound handles:

```text
SDAAStreamState
  sdaaStream_t
  tecodnnHandle_t
  tblasHandle_t
  tecocustomHandle_t
```

Initial SDAA operator priority:

1. storage/copy/memset/device info
2. creation: empty, zeros, ones, full, arange where practical
3. cast/copy/contiguous
4. generated elementwise unary/binary/comparison kernels or tecocustom equivalents
5. reductions: sum, mean, min, max, prod
6. matmul/dot via Tecoblas
7. FFT/signal functionality only after primitive correctness is established

Avoid porting the entire current CUDA kernel tree to SDAA manually. The first SDAA milestone should prove the backend contract, ArrayIterator, and generated operator path.

## Array Protocol

The core array must be modeled as a small value object pointing at explicit storage metadata:

```text
Array
  ├─ ArrayLayout: shape, strides, offset, dtype, place, flags
  └─ Storage: data pointer, byte size, owning place, deleter, reference count
```

The current `InsightArray` remains C ABI-compatible while the C++ layer exposes explicit storage metadata and lifetime tests. A complete `Storage` object split is intentionally deferred until dynamic-output kernels share one centralized ownership creation path.

### Required Semantics

- A root array owns a storage allocation.
- A view shares the same storage and changes only shape, strides, and offset.
- `data()` returns the address of the first logical element, including offset.
- `storage_data()` or equivalent returns the base allocation pointer.
- `nbytes()` for logical array size is distinct from storage allocation size.
- Refcount must be safe for binding/language lifetimes and future threaded use.
- C ABI comments and implementation must agree on who deallocates data.
- Non-contiguous views must remain first-class operands for elementwise, copy, cast, and fallback paths.

### Current Risks To Remove

- `InsightArray` comments say data is externally owned, while `Array::~Array()` deallocates it.
- `ref_count` is a raw `int32_t*` and increments are not represented as a clear ownership object.
- View byte calculations use logical `numel * dtype_size`, which is not the same as storage allocation size.
- Some backend kernels mutate output layout/refcount directly for dynamic-shape outputs.
- CPU fallback cannot safely handle some non-contiguous view storage cases.

## DType Protocol

The enum remains useful for built-in fast paths, and the refactor treats dtype behavior as descriptor-style metadata:

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
- NumPy/CuPy should guide edge semantics; Paddle should guide public naming/default differences where intentional.

## ArrayIterator Protocol

Insight needs a NumPy-inspired ArrayIterator to stop duplicating broadcast, stride, offset, contiguous fast path, and scalar handling in every op.

```text
ArrayIterator
  operands: input/output arrays and scalars
  logical shape after broadcasting
  per-operand dtype, strides, offsets, read/write flags
  casting/writeback policy
  contiguous fast-path descriptor
  reduction descriptor when applicable
```

Required semantics:

- Broadcasting is resolved once before kernel launch.
- Contiguous arrays use fast contiguous loops/kernels.
- Non-contiguous arrays use stride-aware loops/kernels.
- Offset and storage base are represented explicitly.
- Scalar host attrs are not confused with array pointers.
- Reduction iteration is represented as a structured operation, not ad hoc transpose/copy in every frontend.
- CPU fallback can use the same iterator metadata to preserve logical shape, strides, offsets, and writeback behavior.

Initial iterator users:

- copy/contiguous
- cast
- unary elementwise
- binary elementwise
- comparison/logical ops
- simple reductions

## Operator And Kernel Protocol

The current kernel ABI passes null-terminated `void**` inputs/outputs, with scalar parameters mixed into the same channel. This is too implicit.

The refactor introduced an op schema layer for schema-aware launch and fallback:

```text
OpSchema
  name
  inputs: Array parameters
  outputs: Array parameters
  attrs: typed scalar/list/string parameters
  shape inference
  dtype inference / promotion
  broadcast and iterator policy
  dispatch keys
  fallback policy
```

```text
KernelSignature
  op name
  backend kind: CPU or GPU
  active GPU backend name when backend kind is GPU
  dtype loop key
  typed array args
  typed attrs
  iterator descriptor when needed
```

### Required Semantics

- Shape inference happens before allocation.
- Type promotion happens before kernel dispatch.
- Scalar attributes are not guessed from `void*`.
- Kernel arity and argument types are inspectable.
- CPU fallback receives a structured invocation, not a guessed list of raw pointers.
- Fallback must preserve strides, offset, logical shape, and target device.
- Backend capability can choose among native kernel, vendor library, generated iterator kernel, or structured CPU fallback.

### Current Risks To Remove

- CPU fallback scans raw pointers and guesses which are `InsightArray*`.
- Fallback hardcodes `GPUPlace(0)`.
- Fallback copies `numel * dtype_size` from `data`, which is wrong for some non-contiguous views and offsets.
- Kernel registration is keyed only by `op|device|dtype`, with no schema or attr contract.

## Lua Code Generation

Insight should use Lua for build-time code generation of repetitive operator glue. Lua fits the project direction because Insight already cares about Lua bindings and the Paddle-Lua stack, while keeping the generator lightweight and easy to embed in development workflows.

Generation should happen at configure/build/developer time, not at user runtime.

Current layout:

```text
tools/codegen/
  gen.lua
  schema/
    cast.lua
    creation.lua
    elementwise.lua
    reduction.lua
    unary.lua
  templates/
    kernel_plan.lua
    manifest.lua
    op_header.lua
    source_manifest.lua

include/insight/generated/
  kernel_plan.h
  source_manifest.h
  cast_ops.h
  creation_ops.h
  elementwise_ops.h
  reduction_ops.h
  unary_ops.h

docs/
  op_manifest.md
  kernel_plan.md
  source_manifest.md
```

Example schema style:

```lua
op {
  name = "add",
  kind = "binary_elementwise",
  inputs = { "a:array", "b:array" },
  outputs = { "out:array" },
  promote = "numeric",
  broadcast = true,
  dtypes = { "bool", "i32", "i64", "f32", "f64", "c32", "c64" },
  host = { expr = "a + b" },
  device = { expr = "a + b" },
}
```

Generator responsibilities:

- Emit deterministic operator manifests and kernel/source plans.
- Emit frontend declarations and repetitive dispatch boilerplate as the generated surface grows.
- Emit op schema metadata for runtime inspection.
- Emit host iterator kernels for simple elementwise/cast/reduction patterns.
- Emit backend-neutral device iterator metadata that CUDA/HIP/SDAA/IXUCA adapters can consume.
- Emit binding glue where all languages should expose the same operator set.
- Keep generated files deterministic and reviewable.

Generator non-goals:

- Do not hide complex algorithms in template strings.
- Do not generate handwritten-quality linalg/FFT/signal algorithms before their primitive protocols are stable.
- Do not make Lua a runtime dependency for end users unless they build from generated sources.

First generated operator set:

```text
copy, contiguous, cast,
full, zeros, ones,
add, sub, mul, div,
equal, not_equal, greater, less,
neg, abs, exp, log, sqrt,
sum, mean, min, max
```

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
- Generated Lua bindings should share one axis/index conversion helper instead of duplicating conversion logic per op.

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
- Python `__cuda_array_interface__` for CUDA-compatible GPU arrays if stream semantics are explicit.
- A backend-neutral GPU interop story for non-CUDA GPU backends where CUDA array interface is not semantically correct.
- Explicit host copy APIs for GPU arrays, following CuPy's stance against accidental implicit device-to-host conversion.

### Lifetime Rule

Any exported view must retain the owning storage until the consumer releases it. Returning a NumPy array that points to a local temporary is forbidden.

## API Direction

Insight is part of the Paddle-Lua ecosystem. When naming/defaults differ across NumPy, CuPy, Paddle, and PyTorch:

1. Use NumPy/CuPy to guide array protocol and implementation structure.
2. Use Paddle to guide public API behavior where Paddle intentionally differs from PyTorch.
3. Avoid PyTorch-only names or semantics unless there is no Paddle/NumPy equivalent and the feature is clearly useful.

Insight should feel like a language-neutral NumPy/CuPy core rather than a Python-first library with bindings added later.

## Migration Plan

### Phase 0: Safety Fixes And Contract Freeze

- Fix Python `to_numpy()` lifetime.
- Make C ABI ownership comments match implementation.
- Add helper functions for Lua axis conversion and apply them to high-risk wrappers.
- Document CPU/GPU two-slot backend decision.
- Add `active_gpu_backend_name()` diagnostics even before broader backend refactoring.
- Write semantic tests for view lifetime, non-contiguous copy, offset handling, and fallback correctness.

### Phase 1: Storage Refactor

- Introduce explicit storage metadata in C++.
- Replace raw `int32_t* ref_count` with a safer ownership object.
- Preserve C ABI compatibility or add a versioned ABI alongside the current ABI.
- Add tests for root/view destruction, assignment, slicing, and cross-binding lifetime.

### Phase 2: Axis/Indexing Contract

- Add shared core helpers for axis normalization.
- Move language convention conversion into binding helpers.
- Add Python/Lua/Julia axis alignment tests for reductions, manipulation, indexing, and signal ops.

### Phase 3: GPU Backend Selection Cleanup

- Add explicit GPU backend selection in initialization.
- Store backend name/version/capabilities behind the GPU device interface.
- Ensure fallback/copy paths use the real source/output place and device id.
- Keep public `GPUPlace` stable while making diagnostics backend-specific.
- Prepare CMake/plugin naming for `insight_cuda_backend`, `insight_rocm_backend`, `insight_ixuca_backend`, `insight_sdaa_backend`, and future backend plugins.

### Phase 4: ArrayIterator MVP

- Add ArrayIterator for copy/cast/elementwise/comparison first.
- Add contiguous and strided CPU fast paths.
- Add GPU iterator metadata format suitable for CUDA/ROCm/SDAA implementations.
- Replace duplicated broadcast/offset loops in migrated ops.

### Phase 5: Kernel Schema And Structured Fallback

- Add `OpSchema` and typed attr plumbing.
- Migrate fallback-sensitive ops first: reductions, indexing, gather/scatter, unique/nonzero.
- Replace heuristic CPU fallback with structured fallback.
- Add tests that fallback preserves view offsets, non-contiguous strides, and target device writeback.

### Phase 6: Lua Codegen Pilot

- Implement `tools/codegen/gen.lua` and a small schema set.
- Generate declarations, frontend boilerplate, and CPU iterator kernels for the first generated operator set.
- Keep generated output committed or otherwise reproducible in CI.
- Compare generated ops against existing handwritten ops before deleting old implementations.

### Phase 7: DType Descriptor And Promotion

- Centralize promotion and casting policy.
- Keep enum fast paths for built-ins.
- Add conformance tests against NumPy/CuPy/Paddle where appropriate.
- Make generated op schemas reference centralized promotion rules instead of per-op ad hoc logic.

### Phase 8: SDAA Backend Skeleton

- Add `backends/sdaa` with runtime, registry, device interface, CMake discovery, and minimal generated kernels.
- Implement SDAA memory, copy, stream, event, device info, and diagnostics.
- Run basic CPU/GPU parity tests for creation, copy, cast, elementwise, and reduction.
- Add optional Tecoblas matmul once storage/copy correctness is proven.
- Add SDPTI profiler integration only after functional correctness is stable.

### Phase 9: Interop

- Implement DLPack.
- Fix Python NumPy/CUDA interop around lifetime and stream semantics.
- Add round-trip tests with NumPy and CuPy.
- Define the correct public interop behavior for non-CUDA GPU backends.

## Implementation Progress And Remaining Work

Completed in the current feature branch:

- Public `cpu:0` / `gpu:0` device model with active GPU backend diagnostics.
- IXUCA naming cleanup and SDAA runtime backend skeleton.
- OpSchema, ArrayIterator, axis helpers, dtype metadata, and schema-aware CPU fallback.
- Lua codegen pilot with deterministic operator/source/kernel manifests and backend-neutral `host` / `device` adapters.
- Python NumPy/DLPack-facing interop entry points and binding lifetime tests.
- Documentation moved into git-tracked `docs/` and binding READMEs so platform work does not depend on local notes.

Still intentionally open:

- Full `Storage` object split after dynamic-output kernel ownership is centralized.
- More generated host/device kernel source emission beyond metadata manifests.
- Native CUDA/ROCm validation on machines with those runtimes.
- Windows CUDA fixes discovered by the next validation pass.
- Julia validation on a machine with a working Julia runtime.

## Acceptance Criteria

- Public device model remains `cpu` / `gpu`, while diagnostics identify the active GPU backend.
- Exactly one GPU backend is active per process unless a real multi-backend use case is introduced later.
- No dangling memory in Python NumPy export.
- Views preserve storage lifetime and write-through semantics.
- Lua positive axes are one-based consistently at binding boundary.
- Python axes remain zero-based.
- Julia shape/axis conversion is centralized.
- CPU fallback does not guess raw pointer types.
- CPU fallback preserves view offsets, strides, logical shape, and GPU writeback.
- DType promotion behavior is centralized and testable.
- Lua code generation is deterministic, reviewable, and limited to repetitive glue/simple kernels.
- SDAA is implemented as the selected `gpu` backend, not as a public third device kind.
- Public docs explicitly state Paddle alignment and non-PyTorch goals.
