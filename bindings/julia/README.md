# Insight7 Julia Bindings

Julia bindings for the Insight7 scientific computing framework.
Built using Julia's `ccall()` to interface with a C ABI wrapper library (`libinsight_julia.so`),
providing a PaddlePaddle-style API with idiomatic Julia conventions.

## Installation

### Prerequisites

- Julia 1.6+
- Insight7 built with Julia binding target

### Build

```bash
# From the Insight7 root directory
mkdir -p build && cd build
cmake .. -DINSIGHT_WITH_CUDA=ON
cmake --build . --target insight_julia -j$(nproc)
```

### Environment Setup

Add the Julia bindings directory to Julia's load path:

```julia
push!(LOAD_PATH, "/path/to/Insight7/bindings/julia")
using Insight
```

Or set the environment variable before launching Julia:

```bash
export JULIA_LOAD_PATH="/path/to/Insight7/bindings/julia:$JULIA_LOAD_PATH"
export LD_LIBRARY_PATH="build/backends/cpu:$LD_LIBRARY_PATH"
```

For CUDA support:

```bash
export LD_LIBRARY_PATH="build/backends/cpu:build/backends/cuda:$LD_LIBRARY_PATH"
```

### Run Tests

```bash
# From the Insight7 root directory
LD_LIBRARY_PATH=build/backends/cpu julia tests/bindings/test_binding.jl
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

#### 5. Julia Setup

```bash
# Install Julia 1.6+
# Option A: From apt
sudo apt-get install -y julia

# Option B: From official release
wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.2-linux-x86_64.tar.gz
tar xzf julia-1.11.2-linux-x86_64.tar.gz
export PATH="$PWD/julia-1.11.2/bin:$PATH"

# Build with Julia binding
cd build
cmake .. -DINSIGHT_BUILD_JULIA_BINDING=ON
cmake --build . --target insight_julia -j$(nproc)

# Copy Julia module files
cp bindings/julia/Insight.jl build/bindings/julia/
cp -r bindings/julia/modules build/bindings/julia/

# Verify
export LD_LIBRARY_PATH=build/backends/cpu:$LD_LIBRARY_PATH
julia -e 'push!(LOAD_PATH, "bindings/julia"); using Insight; println("Insight7 loaded")'
```

### Windows Installation Guide

#### 1. Prerequisites

- Visual Studio 2022+ with C++ workload
- CMake 3.15+ and Ninja (install via `winget install Kitware.CMake Ninja-build.Ninja`)
- Julia 1.6+

#### 2. Dependencies (via vcpkg)

```powershell
# Install vcpkg if not already installed
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg && bootstrap-vcpkg.bat

# Install FFTW3 and OpenBLAS
.\vcpkg install fftw3:x64-windows openblas:x64-windows
```

#### 3. Install Julia

```powershell
# Option A: Install via winget
winget install julia -s msstore

# Option B: Download from https://julialang.org/downloads/
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
$env:PATH = "$PWD\build\backends\cpu;$PWD\build\backends\cuda;$env:PATH"
$env:JULIA_LOAD_PATH = "$PWD\bindings\julia;$env:JULIA_LOAD_PATH"
julia -e 'using Insight; println("Insight7 loaded")'
```

#### 6. Run Tests

```powershell
$env:PATH = "$PWD\build\backends\cpu;$env:PATH"
$env:JULIA_LOAD_PATH = "$PWD\bindings\julia;$env:JULIA_LOAD_PATH"
julia tests/bindings/test_binding.jl
```

> **Note:** For plot functionality, install [gnuplot](http://www.gnuplot.info/) and ensure it is in your `PATH`.

## Quick Start

```julia
push!(LOAD_PATH, "/path/to/Insight7/bindings/julia")
using Insight

# Backend auto-detected (GPU when available)
dt, id = Insight.get_device()  # public kind id + device id; GPU means gpu:0
println(Insight.active_gpu_backend_name())  # "cuda", "sdaa", ... when GPU is active

# --- Array Creation ---
a = Insight.zeros(Int64[2, 3], Insight.float32)
b = Insight.ones(Int64[2, 3], Insight.float32)
c = Insight.arange(0.0, 10.0, 1.0, Insight.float64)
e = Insight.randn(Int64[3, 3], Insight.float32)

# --- Arithmetic ---
f = a + b               # elementwise add
g = Insight.mul(f, b)   # elementwise multiply
h = Insight.matmul(a, Insight.transpose(b))

# --- NumPy-style Indexing (1-based Julia) ---
row = a[1]              # partial indexing → shape (3,)
val = a[1, 2]           # scalar extraction

# --- Device Management ---
Insight.set_device(1, 0)   # switch to public gpu:0; active backend selected by init/env
Insight.set_device(0, 0)   # switch to public cpu:0

# --- Reductions ---
s = Insight.sum(a)
m = Insight.mean(a)

# --- Linear Algebra ---
det_val = Insight.det(e)
x = Insight.solve(e, b)

# --- FFT ---
fft_result = Insight.fft(c)

# --- Signal Processing ---
w = Insight.hann(Int64(256))
f, Pxx = Insight.welch_jl(c, 1000.0)
analytic = Insight.hilbert(d)
filtered = Insight.filtfilt(b_coeffs, a_coeffs, signal)
```

## Module Structure

The Julia binding is organized into per-module files under `modules/`, with a `signal/`
sub-namespace for signal processing functions.

```
Insight.jl              # Main module, exports all functions
├── modules/
│   ├── types.jl        # DType, Place definitions
│   ├── cast.jl         # cast (type conversion)
│   ├── complex.jl      # is_complex, to_complex, as_complex, real, imag, ...
│   ├── elementwise.jl  # add, sub, mul, div, pow, comparisons, logical ops
│   ├── fft.jl          # fft, ifft, rfft, irfft, fft2, ifft2, fftfreq, ...
│   ├── indexing.jl     # take, nonzero, sort, argsort, gather, scatter, ...
│   ├── linalg.jl       # matmul, dot, det, inv, solve, svd, eigh, cholesky, ...
│   ├── manipulation.jl # concat, stack, split, reshape, squeeze, permute, ...
│   ├── random.jl       # rand, randn, randint, normal, uniform, seed, ...
│   ├── reduction.jl    # sum, mean, max, min, prod, argmax, argmin, var, std, ...
│   ├── unary.jl        # abs, sqrt, exp, log, sin, cos, floor, ceil, ...
│   └── signal/         # Signal processing sub-namespace (Insight.signal.*)
│       ├── windows.jl
│       ├── waveforms.jl
│       ├── bsplines.jl
│       ├── filter_design.jl
│       ├── convolution.jl
│       ├── filtering.jl
│       ├── spectral.jl
│       ├── wavelets.jl
│       ├── acoustics.jl
│       ├── radar.jl
│       ├── demod.jl
│       ├── peak_finding.jl
│       ├── estimation.jl
│       └── io.jl
├── insight_julia_capi.cpp  # C ABI wrapper (compiled to libinsight_julia.so)
└── CMakeLists.txt
```

## API Reference Overview

### Types and Places

```julia
Insight.float32, Insight.float64, Insight.float16, Insight.bfloat16
Insight.int8, Insight.int16, Insight.int32, Insight.int64
Insight.uint8, Insight.uint16, Insight.uint32, Insight.uint64
Insight.bool, Insight.complex64, Insight.complex128

Insight.CPU    # Int32(0)
Insight.GPU    # Int32(1)
```

### Array Properties

```julia
Insight.shape(arr)         # Vector{Int64} of dimension sizes
Insight.dtype(arr)         # DType enum value
Insight.device_type(arr)   # Int32 (0=CPU, 1=GPU)
Insight.numel(arr)         # total number of elements
Insight.ndim(arr)          # number of dimensions
```

### Array Conversion

```julia
# Insight Array -> Julia Array
julia_arr = Insight.to_array(insight_arr)

# Julia Array -> Insight Array
insight_arr = Insight.from_data(julia_arr, Insight.float32)
```

### Creation

`zeros`, `ones`, `full`, `eye`, `arange`, `linspace`, `from_data`

### Elementwise

`add`, `sub`, `mul`, `div`, `pow` (plus operator overloading: `+`, `-`, `*`, `/`)

### Unary

`abs_fn`, `sqrt_fn`, `exp_fn`, `log_fn`, `sin_fn`, `cos_fn`, `tan_fn`, `floor_fn`, `ceil_fn`, `round_fn`, `exp2_fn`, `expm1_fn`, `log1p_fn`, `cbrt_fn`, `reciprocal_fn`, `asinh_fn`, `acosh_fn`, `atanh_fn`, `trunc_fn`, `deg2rad_fn`, `rad2deg_fn`, `negative`

### Reduction

`sum`, `mean`, `max`, `min`, `prod`, `argmax`, `argmin`, `any`, `all`, `var`, `std`, `cumsum`, `cumprod`, `cummax`, `cummin`, `sem`, `count_nonzero`, `median`, `quantile`, `percentile`, `nansum`, `nanmean`, `nanmax`, `nanmin`, `nanstd`, `nanvar`

### Manipulation

`concat`, `stack`, `vstack`, `hstack`, `split`, `tile`, `repeat`, `flip`, `pad`, `reshape`, `flatten`, `ravel`, `squeeze`, `unsqueeze`, `roll`, `permute`, `swapaxes`, `moveaxis`, `fliplr`, `flipud`, `rot90`, `diag_fn`, `diagonal`, `tril`, `triu`, `diff_fn`

### Linear Algebra

`matmul`, `dot`, `inner`, `outer`, `det`, `slogdet`, `inv`, `pinv`, `solve`, `lstsq`, `svd`, `svdvals`, `eigh`, `eigvalsh`, `eig`, `eigvals`, `cholesky`, `qr`, `lq`, `lu`, `norm`, `cond_fn`, `matrix_rank`, `matrix_power`, `trace`, `cov`, `transpose`

### FFT

`fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`, `next_fast_len`, `hfft`, `ihfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`

### Complex

`is_complex`, `has_complex_shape`, `to_complex`, `as_complex`, `as_real`, `real_part`, `imag_part`

### Random

`rand`, `randn`, `randint`, `normal`, `uniform`, `randperm`, `seed`, `get_seed`, `rand_like`, `randn_like`, `exponential`, `gamma_dist`, `beta_dist`, `binomial_dist`, `poisson_dist`

### Signal Processing (89 functions)

Signal functions are available both at the top level (`Insight.hann(...)`) and via the `Insight.signal` sub-namespace:

```julia
using Insight

# Windows
w = Insight.hann(Int64(256))
w = Insight.signal.windows.hann(Int64(256))

# Filtering
b, a = Insight.firwin(Int64(101), 0.5)
filtered = Insight.filtfilt(b, a, signal)
resampled = Insight.resample(signal, Int64(1000))

# Spectral Analysis
f, Pxx = Insight.welch_jl(signal, 1000.0)
f, t, Sxx = Insight.spectrogram(signal, 1000.0)

# Wavelets
coeffs = Insight.cwt(signal, Insight.morlet, widths)

# Radar
compressed = Insight.pulse_compression(tx, rx)
```

Signal submodules: `windows`, `waveforms`, `bsplines`, `filter_design`, `convolution`, `filtering`, `spectral`, `wavelets`, `acoustics`, `radar`, `demod`, `peak_finding`, `estimation`, `io`

### KalmanFilter

The `KalmanFilter` type provides multi-point Kalman filtering for state estimation.
All matrices are stacked on the `points` dimension for batched processing.

```julia
using Insight

# Create a 2-state, 1-measurement Kalman filter
kf = Insight.KalmanFilter(2, 1)

# Set system matrices (as InsightArrays)
kf.F = Insight.from_data([1.0 0.0; 1.0 1.0], Insight.float64)  # state transition
kf.H = Insight.from_data([1.0 0.0], Insight.float64)            # measurement function

# Set noise covariances
kf.Q = Insight.from_data([0.01 0.0; 0.0 0.01], Insight.float64)  # process noise
kf.R = Insight.from_data([0.1], Insight.float64)                  # measurement noise

# Initialize state
kf.x = Insight.from_data([0.0, 0.0], Insight.float64)            # initial state
kf.P = Insight.from_data([1.0 0.0; 0.0 1.0], Insight.float64)    # initial covariance

# Run predict-update loop
z = Insight.from_data([1.05], Insight.float64)  # measurement
Insight.predict(kf)         # x = F*x, P = F*P*F' + Q
Insight.update(kf, z)       # compute Kalman gain, update x and P

# Read estimated state
state = Insight.to_array(kf.x)  # [x1, x2]
```

Batched processing (multiple filters in parallel):

```julia
# 10 independent filters, each 2-state, 1-measurement
kf = Insight.KalmanFilter(2, 1; points=10)

# Set matrices with batched dimension [points, dim_x, dim_x]
# ... set F, H, Q, R as 3D arrays ...

Insight.predict(kf)
Insight.update(kf, measurements)  # measurements: [10, 1, 1]
```

### Device Information

```julia
Insight.device_name("gpu", 0)          # active backend device name
Insight.active_gpu_backend_name()       # "cuda", "sdaa", ... when GPU is active
Insight.active_gpu_backend_version()    # backend runtime/driver version string
Insight.gpu_version()                   # backend numeric version, 0 if no GPU
Insight.driver_version()                # backend driver version when available
Insight.compute_capability(0)           # CUDA-style capability where available
Insight.device_memory(0)                # (total=85899345920, free=...)
Insight.gpu_count()                     # active backend device count
Insight.get_device()                    # public kind id + device id
Insight.set_device(0, 0)                # switch to public cpu:0
```

## Implementation Notes

- The binding uses Julia's `ccall()` to call into `libinsight_julia.so`, which wraps
  Insight7's C++ API with a C ABI.
- `InsightArray` is a Julia mutable struct holding a `Ptr{Cvoid}` to the C++ `Array` object.
  A `finalizer` ensures the C++ memory is freed when the Julia object is garbage collected.
- All functions use `Int64` for dimensions and `Int32` for dtype/device enums.
- Julia's `size()` returns a `Tuple`; use `collect()` to convert to `Vector{Int64}` when
  passing to Insight functions.
- DType values are `Int32` constants (not Julia `DataType` objects) for ccall compatibility.

## License

Part of the Insight7 framework. See [LICENSE](../../LICENSE).
