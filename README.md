[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17/20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-1140%2B%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Insight

**A lightweight, industrial-grade tensor computation framework for signal processing and GPU acceleration.**

Insight is a C++ tensor library inspired by **PaddlePaddle** (operator registration, device HAL), **Torch7** (clean C API, TH/THC spirit), and **NumPy/CuPy** (Python-side API conventions). It delivers CPU/GPU unified computing with zero-copy views, dynamic operator dispatch, and full signal processing support.

## Features

- **Unified API** -- `insight::Array` works seamlessly on `CPUPlace` and `GPUPlace`
- **Zero-Copy Views** -- `reshape`, `transpose`, `slice` via strides & offset
- **Dynamic Operator Registry** -- `ops()["add"][CPU][F32](args)` dispatch (Paddle style)
- **Device HAL** -- ABI-stable plugin system via `Device` base class + `extern "C"` factory
- **Signal Processing** -- 89 functions across 14 submodules (windows, waveforms, B-splines, filter design, convolution, filtering, spectral analysis, wavelets, acoustics, radar, demod, peak finding, estimation, I/O), all with CPU and CUDA backend kernels
- **Half-Precision** -- fp16/bf16 support via `half_utils.h`/`half_utils.cuh`, 116 kernel files with half-precision coverage
- **Language Bindings** -- Python (pybind11), Lua (sol2), Julia (ccall) with per-module wrappers and signal sub-namespaces
- **Modern C++** -- C++17/20, OpenMP parallel, FFTW3, OpenBLAS, cuBLAS, cuFFT
- **No Autograd** -- Keeps the library lightweight and focused

## Architecture

```
insight/
├── include/insight/
│   ├── core/           # Array, Shape, Strides, DType, Place
│   ├── ops/            # Frontend API (elementwise, fft, signal, linalg, etc.)
│   ├── io/             # I/O (csv, print, sndfile)
│   ├── c_api/          # C ABI interfaces (array, kernel, dtype, place)
│   └── plugin/         # Operator registry + device HAL
├── src/
│   ├── core/           # Array implementation, memory management
│   ├── ops/            # Frontend operator logic
│   └── internal/       # Internal utilities
├── backends/
│   ├── cpu/kernels/    # CPU kernels (OpenMP + FFTW + OpenBLAS)
│   └── cuda/kernels/   # CUDA kernels (cuBLAS + cuFFT + Thrust)
├── bindings/
│   ├── python/insight/ # pybind11 bindings (per-module wrappers)
│   ├── lua/insight/    # sol2 bindings (dual calling convention)
│   └── julia/          # ccall bindings (Insight.jl)
├── tests/
│   ├── cpu/            # CPU tests (630+ tests, 27 suites)
│   ├── cuda/           # CUDA tests (510+ tests, 23 suites)
│   └── python_align/   # NumPy precision alignment tests
└── demos/              # Example programs (C++, Python, Lua, Julia)
```

## Quick Start

### Build from Source

**Linux / macOS:**

```bash
git clone https://github.com/PlumBlossomMaid/Insight7.git
cd Insight7
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DINSIGHT_WITH_CUDA=ON \
    -DINSIGHT_USE_FFTW3=ON \
    -DINSIGHT_USE_OPENBLAS=ON
cmake --build . -j$(nproc)
```

**Windows (MSVC):**

```powershell
# Prerequisites: Visual Studio 2022+ (C++ workload), CMake 3.15+, Ninja
# Install dependencies via vcpkg (recommended):
#   vcpkg install fftw3:x64-windows openblas:x64-windows
# Or download OpenBLAS from https://github.com/OpenMathLib/OpenBLAS/releases
#   and extract to e.g. C:\deps\OpenBLAS-0.3.33-x64

# Open VS Developer Command Prompt (x64)
# Adjust path to match your Visual Studio installation:
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

git clone https://github.com/PlumBlossomMaid/Insight7.git
cd Insight7
cmake -S . -B build -G Ninja ^
    -DCMAKE_C_COMPILER=cl.exe ^
    -DCMAKE_CXX_COMPILER=cl.exe ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DINSIGHT_WITH_CUDA=ON ^
    -DCMAKE_PREFIX_PATH="C:/deps/OpenBLAS-0.3.33-x64;E:/vcpkg/installed/x64-windows"
cmake --build build -j %NUMBER_OF_PROCESSORS%
```

> **Note:** For plot functionality, install [gnuplot](http://www.gnuplot.info/) and ensure it is in your `PATH`.

### Install Language Bindings

**Python** (requires CMake build first):
```bash
pip install .
```

**Lua** (via luarocks, requires CMake build first):
```bash
# Lua 5.3
luarocks make bindings/lua/insight-1.0-1.rockspec LUA_DIR=/usr --local

# LuaJIT
luarocks make bindings/lua/insight-1.0-1.rockspec --local
```

**Julia**:
```julia
push!(LOAD_PATH, "/path/to/Insight7/bindings/julia")
using Insight
```

## Examples

### C++

```cpp
#include "insight/insight.h"
using namespace insight;

int main() {
    // Create arrays (auto-selects GPU when available)
    Array a = ones({1000, 1000}, F32);
    Array b = randn({1000, 1000}, F32);

    // Matrix multiplication
    Array c = matmul(a, b);

    // NumPy-style partial indexing
    Array row = c.at({0});     // shape (1000,)
    Array val = c.at({0, 0});  // scalar

    // Signal processing
    Array w = signal::hann(256);
}
```

### Python

```python
import insight as ins

# Auto-selects GPU when available (PaddlePaddle behavior)
print(ins.get_device())  # GPUPlace(0)

a = ins.rand([1000, 1000])
b = ins.randn([1000, 1000])

# Operators: +, -, *, /, //, %, **, @
c = a @ b                # matrix multiplication
d = a ** 2               # elementwise power
e = a // 3.0             # floor division

# NumPy-style indexing
row = a[1]               # partial indexing → shape (1000,)
val = a[1, 2]            # scalar extraction
sub = a[1:, ::2]         # mixed slice indexing

# Signal processing
w = ins.signal.hann(256)
f, Pxx = ins.signal.welch(x, fs=1000)
```

### Lua

```lua
local ins = require("insight")
-- Backend auto-detected, GPU selected when available

print(ins.get_device())       -- "gpu:0" or "cpu:0"
print(ins.gpu_version())      -- GPU runtime version, 0 if no GPU

local a = ins.rand({1000, 1000})
local b = ins.randn({1000, 1000})
local c = ins.matmul(a, b)

-- 1-based indexing (Lua convention)
local row = a[1]              -- partial indexing → shape (1000,)

-- Dual calling convention
local w = ins.signal.hann(256)
local w2 = ins.signal.hann{n=256}
```

### Julia

```julia
using Insight

dt, id = Insight.get_device()  # (1, 0) for GPU

a = Insight.rand(Int64[1000, 1000], Insight.float32)
b = Insight.randn(Int64[1000, 1000], Insight.float32)
c = Insight.matmul(a, b)

# 1-based indexing (Julia convention)
row = a[1]                     # partial indexing → shape (1000,)
val = a[1, 2]                  # scalar extraction
```

## GPU Benchmark (A800-SXM4-80GB)

Tested on Baidu AI Studio with 24-core CPU + NVIDIA A800-SXM4-80GB:

| Test | CPU (24-core) | GPU (A800) | Speedup |
|------|---------------|------------|---------|
| add (20M elements) | 226ms | 601μs | **376x** |
| mul (20M elements) | 229ms | 609μs | **376x** |
| sin (20M elements) | 278ms | 771μs | **361x** |
| sum (20M elements) | 26ms | 8.8μs | **2,962x** |
| max (20M elements) | 42ms | 8.4μs | **4,976x** |
| matmul 256×256 | 19ms | 38μs | **503x** |
| matmul 1024×1024 | 3.6s | 110μs | **32,348x** |
| rfft2 512×512 | 4.5ms | 1.2ms | **3.7x** |
| randn (20M) | 766ms | 82ms | **9.4x** |
| sort (2M) | 206ms | 187ms | **1.1x** |

> GPU excels at large-scale parallel operations. Small FFTs and SVD have kernel launch overhead that favors CPU. The framework automatically selects the optimal device.

## Dependencies

| Dependency | Version | Required | Notes |
|------------|---------|----------|-------|
| CMake | 3.15+ | Yes | Build system |
| C++17 compiler | -- | Yes | GCC 9+, Clang 12+, MSVC 2019+ |
| CUDA | 11.7+ | No | Optional GPU backend |
| OpenBLAS | any | No | CPU linear algebra |
| FFTW3 | 3.3+ | No | CPU FFT |
| OpenMP | -- | No | CPU multi-threading |
| Thrust | bundled | No | CUDA sorting/unique |
| GoogleTest | auto | -- | Automatically fetched |

## Test Status

**1140+ tests passing** -- CPU (630+, 27 suites) and CUDA (510+, 23 suites), plus 384 precision alignment tests

| Suite | CPU | CUDA | Notes |
|-------|-----|------|-------|
| cast | 9 | 9 | |
| complex | 22 | 22 | |
| creation | 27 | 27 | |
| csv | 1 | 1 | |
| dtype | 9 | 9 | Shared |
| elementwise | 28 | 28 | |
| fft | 19 | 19 | |
| indexing | 41 | 33 | |
| linalg | 43 | 43 | 15 native CUDA + 13 C_FALLBACK |
| manipulation | 42 | 42 | |
| operator | 50 | 50 | |
| print | 11 | 11 | |
| random | 31 | 31 | |
| reduction | 24 | 24 | |
| signal (core) | 10 | 10 | Composite ops |
| signal_windows | 30 | 30 | |
| signal_waveforms | 18 | 18 | |
| signal_bsplines | 13 | 13 | |
| signal_filter_design | 22 | 22 | |
| signal_convolution | 21 | 17 | |
| signal_filtering | 23 | 15 | |
| signal_spectral | 11 | -- | |
| signal_wavelets | 13 | -- | |
| signal_acoustics | 9 | -- | |
| signal_radar | 7 | -- | |
| signal_io | 11 | -- | |
| signal_peak_finding | 3 | -- | |
| signal_demod | 1 | -- | |
| signal_estimation | 1 | -- | |
| plot | 13 | -- | |
| unary | 27 | 27 | |
| audio | 9 | 9 | |
| **Total** | **630+** | **510+** | |

### Language Binding Tests

| Language | Test Framework | Tests |
|----------|---------------|-------|
| Python | pytest | 76 smoke + 54 numerical + 384 alignment |
| Lua | busted | 208 |
| Julia | Test stdlib | 74 |

## Demos

Example programs in `demos/` covering 4 languages and 6 scenarios:

| Demo | C++ | Python | Lua | Julia |
|------|-----|--------|-----|-------|
| basic_ops | ✅ | ✅ | ✅ | ✅ |
| fft_demo | ✅ | ✅ | ✅ | ✅ |
| gpu_transfer | ✅ | ✅ | ✅ | ✅ |
| linalg_demo | ✅ | ✅ | ✅ | ✅ |
| radar_task1 | ✅ | ✅ | ✅ | ✅ |
| sndfile_demo | ✅ | ✅ | ✅ | ✅ |

## License

[Apache License 2.0](LICENSE)

Copyright 2026 PlumBlossomMaid

## Contributing

Issues and pull requests are welcome. Please ensure:
1. Code follows `.clang-format` style
2. All existing tests pass
3. New features include corresponding tests
