# Windows CUDA Adaptation Guide

This guide is the canonical checklist for continuing CUDA work on Windows after the unified `cpu:0` / `gpu:0` architecture refactor. Keep fixes in git so build assumptions, diagnostics, and workarounds are shared with Linux/SDAA/ROCm work instead of living in local notes.

## Target Architecture

Windows CUDA is one implementation of the public GPU device kind:

```text
user-visible place: gpu:0, gpu:1, ...
active GPU backend: cuda
backend selection: INSIGHT_GPU_BACKEND=cuda
```

The user API should not expose `cuda:0`. Backend-specific names are for diagnostics only, for example `active_gpu_backend_name()` returning `cuda`.

## Environment

Recommended baseline:

- Windows 10/11 x64
- Visual Studio 2022 with the Desktop development with C++ workload
- CMake 3.20+ and Ninja
- CUDA Toolkit 11.7+ installed from NVIDIA
- Python 3.8+ if validating Python bindings
- Optional CPU libraries from vcpkg: `fftw3:x64-windows`, `openblas:x64-windows`

Verify the toolchain from an x64 Developer PowerShell or Developer Command Prompt:

```powershell
cl
cmake --version
ninja --version
nvcc --version
where cudart64*.dll
```

If `nvcc` or CUDA DLLs are not found, set `CUDA_PATH` and update `PATH` before configuring:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"
```

## Configure

Use Ninja with MSVC first. Keep optional backends off unless the Windows machine has them installed.

```powershell
cmake -S . -B build-cuda-win -G Ninja `
  -DCMAKE_C_COMPILER=cl.exe `
  -DCMAKE_CXX_COMPILER=cl.exe `
  -DCMAKE_BUILD_TYPE=Release `
  -DINSIGHT_WITH_CUDA=ON `
  -DINSIGHT_WITH_ROCM=OFF `
  -DINSIGHT_WITH_SDAA=OFF `
  -DINSIGHT_BUILD_TESTS=ON `
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" `
  -DCMAKE_PREFIX_PATH="C:/vcpkg/installed/x64-windows"
```

For older GPUs, adjust `CMAKE_CUDA_ARCHITECTURES` to the local card. If configure disables CUDA, read the CMake log first; do not patch around the failure by pretending CUDA is enabled.

## Build

```powershell
cmake --build build-cuda-win -j $env:NUMBER_OF_PROCESSORS
```

Useful focused targets while adapting:

```powershell
cmake --build build-cuda-win --target insight_core
cmake --build build-cuda-win --target insight_cuda_backend
cmake --build build-cuda-win --target insight_tests_cuda
cmake --build build-cuda-win --target insight_python
cmake --build build-cuda-win --target insight_lua
```

## Runtime Setup

Windows DLL discovery must include the core, binding, backend, CUDA, and optional dependency locations. From the repo root:

```powershell
$env:INSIGHT_GPU_BACKEND = "cuda"
$env:PATH = "$PWD\build-cuda-win;$PWD\build-cuda-win\backends\cpu;$PWD\build-cuda-win\backends\cuda;$env:CUDA_PATH\bin;C:\vcpkg\installed\x64-windows\bin;$env:PATH"
```

If testing Python or Lua bindings, also set their module paths:

```powershell
$env:PYTHONPATH = "$PWD\bindings\python"
$env:LUA_PATH = "$PWD\bindings\lua\?\init.lua;$PWD\bindings\lua\?.lua;;"
$env:LUA_CPATH = "$PWD\build-cuda-win\bindings\lua\?.dll;;"
```

## Test Plan

Start with targeted CUDA tests, then broaden.

```powershell
ctest --test-dir build-cuda-win -R "CreationTestGPU|ElementwiseTestGPU|UnaryTestGPU" --output-on-failure -j 8
ctest --test-dir build-cuda-win -R "CUDA|GPU" --output-on-failure -j 8
ctest --test-dir build-cuda-win --output-on-failure -j 8
```

For the CUDA executable directly, run from the test binary directory so copied backend DLLs are adjacent:

```powershell
cd build-cuda-win\tests
$env:INSIGHT_GPU_BACKEND = "cuda"
.\insight_tests_cuda.exe --gtest_filter="ElementwiseTestGPU.*"
```

Python smoke test:

```powershell
python -c "import insight as ins; print(ins.get_device()); print(ins.active_gpu_backend_name())"
```

Expected public output is `gpu:0` for the device and `cuda` only from explicit backend diagnostics.

## CUDA Adaptation Priorities

1. Keep public places backend-neutral: use `CPUPlace()` / `GPUPlace(0)` in C++ and `cpu:0` / `gpu:0` in user-facing docs and diagnostics.
2. Preserve the structured fallback contract: kernels that may return `C_FALLBACK` must launch through schema-aware paths so scalar attrs are not guessed from raw `void**`.
3. Keep CUDA kernels stride-aware: views must respect `offset`, `shape`, and `strides`; never assume contiguous output unless the local metadata proves it.
4. Synchronize asynchronous CUDA work before returning from kernels that expose results to following ops, especially cuFFT and library calls.
5. Prefer fixing shared iterator/schema/codegen metadata over creating Windows-only or CUDA-only special cases.
6. When adding generated CUDA kernels, keep Lua schema backend-neutral (`host` / `device` adapters), then let the CUDA backend consume the device adapter.

## Common Windows CUDA Issues

- `nvcc` not found: check `CUDA_PATH` and ensure the VS toolchain was initialized before CMake configure.
- `cudart64_*.dll` not found at runtime: prepend `$env:CUDA_PATH\bin` to `PATH`.
- `insight_cuda_backend.dll` not found: add `build-cuda-win\backends\cuda` to `PATH` or run tests from `build-cuda-win\tests`.
- MSVC macro conflicts around math constants: `_USE_MATH_DEFINES` is set by the top-level CMake file; avoid redefining it in source files.
- Optional FFT/OpenBLAS missing: either install through vcpkg or configure with the matching `INSIGHT_USE_*` option disabled and verify no optional-dependent tests run unguarded.
- Parallel tests racing on temp paths: prefer unique temp directories in tests and rerun failing tests serially before assuming kernel failure.

## Push Discipline

Documentation, build fixes, and Windows CUDA workarounds should be committed and pushed to the feature branch. Avoid relying on local downloads or manual notes for architecture decisions; if future work depends on it, put it in the repo.
