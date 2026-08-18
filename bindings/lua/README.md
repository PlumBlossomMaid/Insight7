# Insight7 Lua Bindings

Lua bindings for the Insight7 scientific computing framework.
Built with [sol2](https://github.com/ThePhD/sol2), providing a dual calling convention
for ergonomic signal processing and scientific computing in Lua.

## Installation

### From luarocks (after CMake build)

```bash
# Build C++ backend + Lua binding first
mkdir -p build && cd build
cmake .. -DINSIGHT_WITH_CUDA=ON -DINSIGHT_USE_FFTW3=ON -DINSIGHT_USE_OPENBLAS=ON
cmake --build . -j$(nproc) --target insight_lua

# Install via luarocks
cd ..
# Lua 5.3
luarocks make bindings/lua/insight-1.0-1.rockspec LUA_DIR=/usr --local
# LuaJIT
luarocks make bindings/lua/insight-1.0-1.rockspec --local
```

### Prerequisites

- LuaJIT 2.1+ (recommended) or Lua 5.3+
- Insight7 built with Lua binding target

### Build

```bash
# From the Insight7 root directory
mkdir -p build && cd build
cmake .. -DINSIGHT_WITH_CUDA=ON
cmake --build . --target insight_lua -j$(nproc)
```

### Environment Setup

Set the following environment variables so Lua can find the Insight modules and native library:

```bash
export LUA_PATH="bindings/lua/?/init.lua;bindings/lua/?.lua;;"
export LUA_CPATH="build/bindings/lua/?.so;;"
export LD_LIBRARY_PATH="build/backends/cpu:$LD_LIBRARY_PATH"
```

For CUDA support:

```bash
export LD_LIBRARY_PATH="build/backends/cpu:build/backends/cuda:$LD_LIBRARY_PATH"
```

### Switching Lua Versions

The build system supports both LuaJIT and Lua 5.3+ via a single CMake target.
Switch by specifying `-DLUA_DIR`:

```bash
# Build for LuaJIT (default)
cmake .. -DLUA_DIR=/usr

# Build for Lua 5.4
cmake .. -DLUA_DIR=/usr/lib/lua5.4
```

> **Note:** When switching Lua versions, clear the CMake cache first:
> `rm -rf build/CMakeCache.txt build/CMakeFiles`

### Run Tests

```bash
# From the Insight7 root directory
LUA_PATH="bindings/lua/?/init.lua;;" \
LUA_CPATH="build/bindings/lua/?.so;;" \
LD_LIBRARY_PATH=build/backends/cpu \
    ~/.luarocks/bin/busted tests/bindings/test_lua_binding.lua -v
```

### Linux Installation Guide

#### 1. System Dependencies (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libfftw3-dev \
    libopenblas-dev \
    liblapacke-dev \
    libomp-dev
```

#### 2. CMake (3.15+)

```bash
# Option A: Install from apt (if version >= 3.15)
sudo apt-get install -y cmake

# Option B: Install from official release
wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.sh
sudo sh cmake-3.28.3-linux-x86_64.sh --prefix=/usr/local --skip-license
```

#### 3. Optional: CUDA Toolkit (11.7+)

```bash
# Install CUDA toolkit from NVIDIA
# See: https://developer.nvidia.com/cuda-downloads
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-3
```

#### 4. Clone and Build

```bash
git clone https://github.com/PlumBlossomMaid/Insight7.git
cd Insight7
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DINSIGHT_WITH_CUDA=ON \
    -DINSIGHT_USE_FFTW3=ON \
    -DINSIGHT_USE_OPENBLAS=ON \
    -DINSIGHT_BUILD_TESTS=ON
cmake --build . -j$(nproc)
```

#### 5. Lua Setup

```bash
# Install LuaJIT (recommended)
sudo apt-get install -y luajit libluajit-5.1-dev luarocks
sudo luarocks install busted  # for testing

# OR install Lua 5.3
sudo apt-get install -y lua5.3 liblua5.3-dev

# Build with Lua binding
cd build
cmake .. -DINSIGHT_BUILD_LUA_BINDING=ON
cmake --build . --target insight_lua -j$(nproc)

# Verify
export LUA_PATH="bindings/lua/?/init.lua;bindings/lua/?.lua;;"
export LUA_CPATH="build/bindings/lua/?.so;;"
export LD_LIBRARY_PATH=build/backends/cpu:$LD_LIBRARY_PATH
luajit -e "local ins = require('insight'); print('Insight7 loaded')"
```

### Windows Installation Guide

#### 1. Prerequisites

- Visual Studio 2022+ with C++ workload
- CMake 3.15+ and Ninja (install via `winget install Kitware.CMake Ninja-build.Ninja`)
- LuaJIT 2.1+ or Lua 5.3+

#### 2. Dependencies (via vcpkg)

```powershell
# Install vcpkg if not already installed
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg && bootstrap-vcpkg.bat

# Install FFTW3 and OpenBLAS
.\vcpkg install fftw3:x64-windows openblas:x64-windows
```

#### 3. Install Lua

```powershell
# Option A: LuaJIT (recommended)
# Download from https://luajit.org/download.html and extract to C:\LuaJIT
# Add C:\LuaJIT to PATH

# Option B: Install via scoop
scoop install luajit
scoop install luarocks
```

#### 4. Clone and Build

```powershell
# Open VS Developer Command Prompt (x64)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

git clone https://github.com/PlumBlossomMaid/Insight7.git
cd Insight7
cmake -S . -B build -G Ninja ^
    -DCMAKE_C_COMPILER=cl.exe ^
    -DCMAKE_CXX_COMPILER=cl.exe ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DINSIGHT_WITH_CUDA=ON ^
    -DCMAKE_PREFIX_PATH="C:/vcpkg/installed/x64-windows"
cmake --build build -j %NUMBER_OF_PROCESSORS%
```

#### 5. DLL Discovery

On Windows, native DLLs must be discoverable at runtime. Add the backend directories to `PATH`:

```powershell
$env:PATH = "$PWD\build\backends\cpu;$PWD\build\backends\cuda;$PWD\build\bindings\lua;$env:PATH"
$env:LUA_PATH = "bindings/lua/?/init.lua;bindings/lua/?.lua;;"
$env:LUA_CPATH = "build/bindings/lua/?.dll;;"
luajit -e "local ins = require('insight'); print('Insight7 loaded')"
```

#### 6. Run Tests

```powershell
$env:PATH = "$PWD\build\backends\cpu;$PWD\build\bindings\lua;$env:PATH"
$env:LUA_PATH = "bindings/lua/?/init.lua;;"
$env:LUA_CPATH = "build/bindings/lua/?.dll;;"
luajit tests/bindings/test_lua_binding.lua
```

> **Note:** For plot functionality, install [gnuplot](http://www.gnuplot.info/) and ensure it is in your `PATH`.

## Quick Start

```lua
local ins = require("insight")
-- Backend auto-detected (GPU when available, no manual init needed)

print(ins.get_device())       -- "gpu:0" or "cpu:0"
print(ins.gpu_version())      -- GPU runtime version, 0 if no GPU

-- Array creation (default: current device)
local a = ins.zeros({2, 3}, ins.float32)
local b = ins.ones({2, 3}, ins.float32)
local c = a + b

-- NumPy-style indexing (1-based for Lua)
local row = a[1]              -- partial indexing → shape (3,)
local val = a[1][2]           -- scalar extraction

-- Device management
local a_gpu = a:to(ins.GPUPlace(0)) -- cpu:0 → gpu:0, active backend selected by init/env
local a_cpu = a_gpu:to(ins.CPUPlace()) -- gpu:0 → cpu:0

-- Reduction
local s = ins.sum(c)

-- Linear algebra
local m = ins.matmul(a, ins.transpose(b))

-- Signal processing (dual calling convention)
local w = ins.signal.hann(256)
local w2 = ins.signal.hann{n=256}
```

## Dual Calling Convention

All Insight functions support both **positional** and **named-table** argument styles,
powered by a shared `_wrap.lua` utility:

```lua
-- Positional arguments
local c = ins.add(a, b)
local w = ins.signal.hann(256)

-- Named-table arguments
local c2 = ins.add{a=a, b=b}
local w2 = ins.signal.hann{n=256}

-- Mixed positional + named
local y = ins.signal.filtfilt{b=b, a=a, x=signal}
local y2 = ins.signal.filtfilt(b, a, signal)
```

The `_wrap` utility (in `insight/_wrap.lua`) automatically detects whether the caller
is using positional or table-style arguments by checking for non-numeric keys.

## Module Structure

```
insight/
├── init.lua            # Main entry point, loads all submodules
├── _wrap.lua           # Shared dual calling convention utility
├── elementwise.lua     # add, sub, mul, div, pow, comparisons, logical ops
├── unary.lua           # abs, sqrt, exp, log, sin, cos, floor, ceil, ...
├── reduction.lua       # sum, mean, max, min, prod, argmax, argmin, var, std, ...
├── manipulation.lua    # concat, stack, split, reshape, squeeze, permute, ...
├── linalg.lua          # matmul, det, inv, solve, svd, eigh, cholesky, qr, ...
├── fft.lua             # fft, ifft, rfft, irfft, fft2, ifft2, fftfreq, ...
├── complex.lua         # is_complex, to_complex, as_complex, real, imag, ...
├── random.lua          # rand, randn, randint, normal, uniform, seed, ...
├── cast.lua            # cast (type conversion)
├── indexing.lua        # take, nonzero, sort, argsort, gather, scatter, ...
├── signal.lua          # Signal top-level (re-exports signal/)
└── signal/             # Signal processing submodule (ins.signal.*)
    ├── init.lua        # Loads all signal submodules
    ├── windows.lua     # hann, hamming, blackman, kaiser, gaussian, ...
    ├── waveforms.lua   # sawtooth, square_wf, gausspulse, chirp, ...
    ├── bsplines.lua    # gauss_spline, cubic, quadratic
    ├── filter_design.lua # firwin, firwin2, kaiser_beta, kaiser_atten, ...
    ├── convolution.lua # fftconvolve, correlate, convolve2d, correlate2d, ...
    ├── filtering.lua   # hilbert, detrend, lfilter, filtfilt, decimate, ...
    ├── spectral.lua    # welch, periodogram, csd, coherence, spectrogram, stft
    ├── wavelets.lua    # morlet, ricker, morlet2, cwt
    ├── acoustics.lua   # mel2hz, hz2mel, mel_frequencies, hz2bark, bark2hz
    ├── radar.lua       # pulse_compression, pulse_doppler, cfar_alpha, mvdr, ambgfun
    ├── demod.lua       # fm_demod
    ├── peak_finding.lua # argrelextrema, argrelmax, argrelmin
    ├── estimation.lua  # KalmanFilter
    └── io.lua          # read_bin, write_bin, unpack_bin, pack_bin, read_sigmf, write_sigmf
```

## API Overview

### Types and Places

```lua
ins.float32, ins.float64, ins.float16, ins.bfloat16
ins.int8, ins.int16, ins.int32, ins.int64
ins.uint8, ins.uint16, ins.uint32, ins.uint64
ins.bool, ins.complex64, ins.complex128

ins.CPUPlace()       -- public CPU place (cpu:0)
ins.GPUPlace(0)      -- public GPU place (gpu:0, active backend selected by init/env)
```

### Array Properties

```lua
arr.shape           -- table of dimension sizes
arr.dtype           -- DType enum value
arr.place           -- CPUPlace or GPUPlace
arr.numel           -- total number of elements
arr.ndim            -- number of dimensions
arr.is_contiguous   -- whether data is contiguous
```

### Array Methods

```lua
arr:contiguous()      -- return contiguous copy if needed
arr:reshape(shape)    -- return view with new shape
arr:transpose()       -- return transposed view
arr:squeeze(axis)     -- remove size-1 dimensions
arr:unsqueeze(axis)   -- insert size-1 dimension
arr:to(place)         -- transfer to device
arr:copy()            -- deep copy
arr:get(index)        -- get scalar value at index (1-based)
```

### Indexing (1-based)

Lua bindings use 1-based indexing, automatically converted to 0-based for the C++ backend:

```lua
local val = arr:get(1)        -- first element
arr["1:3, :"]                 -- string-based slicing
```

### Creation

```lua
ins.zeros({2, 3}, ins.float32)
ins.ones({4, 4}, ins.float64)
ins.full({3, 3}, 7.0, ins.float32)
ins.arange(0, 10, 1, ins.float64)
ins.linspace(0.0, 1.0, 100, ins.float64)
ins.eye(5, ins.float64)
ins.from_table({{1, 2, 3}, {4, 5, 6}})  -- from nested Lua table
```

### Signal Processing

```lua
-- Windows
local w = ins.signal.hann(256)
local w = ins.signal.kaiser{length=256, beta=14.0}
local w = ins.signal.get_window("hamming", 256)

-- Filtering
local b, a = ins.signal.firwin(101, 0.5)
local filtered = ins.signal.filtfilt{b=b, a=a, x=signal}
local resampled = ins.signal.resample{signal, num=1000}

-- Spectral Analysis
local f, Pxx = ins.signal.welch(signal, 1000.0)

-- Wavelets
local coeffs = ins.signal.cwt(signal, ins.signal.morlet, widths)

-- Radar
local compressed = ins.signal.pulse_compression(tx, rx)
```

### Arithmetic Operators

Arrays support Lua operator overloading:

```lua
local c = a + b       -- elementwise add
local c = a - b       -- elementwise subtract
local c = a * b       -- elementwise multiply
local c = a / b       -- elementwise divide
local c = -a          -- negation
local eq = (a == b)   -- elementwise equality
```

### GPU Support

```lua
-- Transfer arrays between public devices
-- Concrete GPU backend is selected by INSIGHT_GPU_BACKEND or initialization.
local a_cpu = ins.zeros({100, 100}, ins.float32)
local a_gpu = a_cpu:to(ins.GPUPlace(0))   -- cpu:0 -> gpu:0
local a_back = a_gpu:to(ins.CPUPlace())   -- gpu:0 -> cpu:0

-- Operations auto-dispatch to the active backend
local c = ins.matmul(a_gpu, a_gpu)  -- runs on active GPU backend
```

## LDoc Documentation

All Lua modules include LDoc-compatible annotations. Generate documentation with:

```bash
ldoc bindings/lua/insight/ -d docs/lua
```

## License

Part of the Insight7 framework. See [LICENSE](../../LICENSE).
