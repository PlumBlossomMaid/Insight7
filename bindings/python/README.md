# Insight7 Python Bindings

NumPy-compatible Python bindings for the Insight7 scientific computing framework.
Built with [pybind11](https://github.com/pybind/pybind11), providing a PaddlePaddle-style API.

## Installation

### From pip (after CMake build)

```bash
# Build C++ backend + Python binding first
mkdir -p build && cd build
cmake .. -DINSIGHT_WITH_CUDA=ON -DINSIGHT_USE_FFTW3=ON -DINSIGHT_USE_OPENBLAS=ON
cmake --build . -j$(nproc) --target insight_python

# Install the Python package
cd .. && pip install .
```

### Prerequisites

- Python 3.8+
- Insight7 built with Python binding target

### Build

```bash
# From the Insight7 root directory
mkdir -p build && cd build
cmake .. -DINSIGHT_WITH_CUDA=ON
cmake --build . --target insight_python -j$(nproc)
```

### Run Tests

```bash
# From the Insight7 root directory
PYTHONPATH=bindings/python LD_LIBRARY_PATH=build/backends/cpu \
    python3 -m pytest tests/bindings/ -v

# Precision alignment tests (vs NumPy)
PYTHONPATH=bindings/python LD_LIBRARY_PATH=build/backends/cpu \
    python3 -m pytest tests/python_align/ -v
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

#### 5. Python Setup

```bash
# Install Python 3.8+ and dependencies
sudo apt-get install -y python3-dev python3-pip python3-numpy
pip3 install pytest

# Build with Python binding
cd build
cmake .. -DINSIGHT_BUILD_PYTHON_BINDING=ON
cmake --build . --target insight_python -j$(nproc)

# Verify
PYTHONPATH=bindings/python LD_LIBRARY_PATH=build/backends/cpu \
    python3 -c "import insight as ins; print('Insight7 loaded:', ins.float32)"
```

### Windows Installation Guide

#### 1. Prerequisites

- Visual Studio 2022+ with C++ workload
- CMake 3.15+ and Ninja (install via `winget install Kitware.CMake Ninja-build.Ninja`)
- Python 3.8+

#### 2. Dependencies (via vcpkg)

```powershell
# Install vcpkg if not already installed
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg && bootstrap-vcpkg.bat

# Install FFTW3 and OpenBLAS
.\vcpkg install fftw3:x64-windows openblas:x64-windows
```

#### 3. Clone and Build

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

#### 4. Python Setup

```powershell
pip install numpy pytest

# Build with Python binding
cmake -S . -B build -G Ninja -DINSIGHT_BUILD_PYTHON_BINDING=ON ^
    -DCMAKE_PREFIX_PATH="C:/vcpkg/installed/x64-windows"
cmake --build build --target insight_python -j %NUMBER_OF_PROCESSORS%

# Install the Python package
pip install .
```

#### 5. DLL Discovery

On Windows, native DLLs must be discoverable at import time. Choose one of:

```powershell
# Option A: Add backend directories to PATH
$env:PATH = "$PWD\build\backends\cpu;$PWD\build\backends\cuda;$env:PATH"
python -c "import insight as ins; print('Insight7 loaded:', ins.float32)"

# Option B: Use os.add_dll_directory() in your script
python -c "import os; os.add_dll_directory(r'E:\code\insight\Insight7\build\backends\cpu'); import insight as ins; print(ins.float32)"
```

#### 6. Run Tests

```powershell
$env:PYTHONPATH = "bindings/python"
$env:PATH = "$PWD\build\backends\cpu;$PWD\build\backends\cuda;$env:PATH"
python -m pytest tests/bindings/ -v
```

> **Note:** For plot functionality, install [gnuplot](http://www.gnuplot.info/) and ensure it is in your `PATH`.

## Quick Start

```python
import insight as ins

# Backend auto-detected (GPU when available, PaddlePaddle behavior)
print(ins.get_device())               # "gpu:0" or "cpu:0"
print(ins.active_gpu_backend_name())  # "cuda", "sdaa", ... when GPU is active

# --- Array Creation ---
a = ins.zeros([2, 3], ins.float32)
b = ins.ones([2, 3], ins.float32)
c = ins.arange(0, 10, 1, ins.float64)
d = ins.linspace(0.0, 1.0, 100, ins.float64)
e = ins.randn([3, 3], ins.float32)

# --- Operators: +, -, *, /, //, %, **, @ ---
f = a + b               # elementwise add
g = a @ b.T             # matrix multiplication (@)
h = a ** 2              # elementwise power
i = a // 3.0            # floor division
j = a % 2.0             # modulo

# --- NumPy-style Indexing ---
row = a[1]              # partial indexing → shape (3,)
val = a[1, 2]           # scalar extraction
sub = a[1:, ::2]        # mixed slice indexing

# --- Reductions ---
s = ins.sum(a, axis=0)
m = ins.mean(a)
mx = ins.max(a, axis=1)

# --- Linear Algebra ---
det_val = ins.det(e)
u, s, vh = ins.svd(e)
x = ins.solve(e, b[:, 0])

# --- FFT ---
fft_result = ins.fft(c)
rfft_result = ins.rfft(d)

# --- Signal Processing ---
w = ins.signal.hann(256)
f, Pxx = ins.signal.welch(c, fs=1000.0)
analytic = ins.signal.hilbert(d)
filtered = ins.signal.filtfilt(b_coeffs, a_coeffs, signal)

# --- NumPy Interop ---
import numpy as np
arr = a.numpy()              # Insight Array -> NumPy ndarray
d = ins.from_numpy(arr)      # NumPy ndarray -> Insight Array
```

## Module Structure

The Python binding is organized into per-module wrapper files, each providing docstrings, type hints, and argument validation on top of the native C++ extension.

```
insight/
├── __init__.py          # Main entry point, re-exports all modules
├── types.py             # DType, Place, Shape, Slice, CPUPlace, GPUPlace
├── array.py             # Array class with NumPy interop
├── creation.py          # zeros, ones, full, eye, arange, linspace, from_numpy, ...
├── elementwise.py       # add, sub, mul, div, pow, mod, comparisons, logical ops
├── unary.py             # abs, sqrt, exp, log, sin, cos, tan, floor, ceil, ...
├── reduction.py         # sum, mean, max, min, prod, argmax, argmin, var, std, ...
├── manipulation.py      # concat, stack, split, reshape, squeeze, permute, ...
├── linalg.py            # matmul, det, inv, solve, svd, eigh, cholesky, qr, ...
├── fft.py               # fft, ifft, rfft, irfft, fft2, ifft2, fftfreq, ...
├── complex.py           # is_complex, to_complex, as_complex, real, imag, ...
├── random.py            # rand, randn, randint, normal, uniform, seed, ...
├── cast.py              # cast (type conversion)
├── indexing.py          # take, nonzero, sort, argsort, gather, scatter, ...
├── signal/              # Signal processing submodule (ins.signal.*)
│   ├── __init__.py      # Re-exports all signal functions
│   ├── windows.py       # hann, hamming, blackman, kaiser, gaussian, ...
│   ├── waveforms.py     # sawtooth, square_wf, gausspulse, chirp, ...
│   ├── bsplines.py      # gauss_spline, cubic, quadratic
│   ├── filter_design.py # firwin, firwin2, kaiser_beta, kaiser_atten, ...
│   ├── convolution.py   # fftconvolve, correlate, convolve2d, correlate2d, ...
│   ├── filtering.py     # hilbert, detrend, lfilter, filtfilt, decimate, ...
│   ├── spectral.py      # welch, periodogram, csd, coherence, spectrogram, stft
│   ├── wavelets.py      # morlet, ricker, morlet2, cwt
│   ├── acoustics.py     # mel2hz, hz2mel, mel_frequencies, hz2bark, bark2hz
│   ├── radar.py         # pulse_compression, pulse_doppler, cfar_alpha, mvdr, ambgfun
│   ├── demod.py         # fm_demod
│   ├── peak_finding.py  # argrelextrema, argrelmax, argrelmin
│   ├── estimation.py    # KalmanFilter
│   └── io.py            # read_bin, write_bin, unpack_bin, pack_bin, read_sigmf, write_sigmf
└── _insight.so          # Native C++ extension (loaded at runtime)
```

## API Reference Overview

### Types and Places

```python
ins.float32, ins.float64, ins.float16, ins.bfloat16
ins.int8, ins.int16, ins.int32, ins.int64
ins.uint8, ins.uint16, ins.uint32, ins.uint64
ins.bool, ins.complex64, ins.complex128

ins.CPUPlace()       # public CPU place (cpu:0)
ins.GPUPlace(0)      # public GPU place (gpu:0, active backend selected by init/env)
```

### Array Properties

```python
arr.shape           # tuple of dimension sizes
arr.dtype           # DType enum value
arr.place           # CPUPlace or GPUPlace
arr.numel           # total number of elements
arr.ndim            # number of dimensions
arr.numpy()         # convert to NumPy ndarray
arr.to(place)       # transfer to device
```

### Creation (15+ functions)

`zeros`, `ones`, `full`, `eye`, `arange`, `linspace`, `logspace`, `from_numpy`, `from_array`, `zeros_like`, `ones_like`

### Elementwise (20+ functions)

`add`, `sub`, `mul`, `div`, `pow`, `mod`, `maximum`, `minimum`, `equal`, `not_equal`, `greater`, `less`, `greater_equal`, `less_equal`, `logical_and`, `logical_or`, `logical_xor`, `logical_not`

### Unary (35+ functions)

`abs`, `negative`, `square`, `sqrt`, `exp`, `log`, `log2`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `round`, `sign`, `isnan`, `isinf`, `isfinite`, `exp2`, `expm1`, `log1p`, `cbrt`, `reciprocal`, `asinh`, `acosh`, `atanh`, `trunc`, `deg2rad`, `rad2deg`, `where`

### Reduction (25+ functions)

`sum`, `mean`, `max`, `min`, `prod`, `argmax`, `argmin`, `any`, `all`, `var`, `std`, `cumsum`, `cumprod`, `cummax`, `cummin`, `sem`, `count_nonzero`, `median`, `quantile`, `percentile`, `nansum`, `nanmean`, `nanmax`, `nanmin`, `nanstd`, `nanvar`

### Manipulation (25+ functions)

`concat`, `stack`, `vstack`, `hstack`, `split`, `tile`, `repeat`, `flip`, `pad`, `reshape`, `flatten`, `ravel`, `squeeze`, `unsqueeze`, `roll`, `permute`, `swapaxes`, `moveaxis`, `fliplr`, `flipud`, `rot90`, `diag`, `diagonal`, `tril`, `triu`, `diff`

### Linear Algebra (20+ functions)

`matmul`, `dot`, `inner`, `outer`, `det`, `slogdet`, `inv`, `pinv`, `solve`, `lstsq`, `svd`, `svdvals`, `eigh`, `eigvalsh`, `eig`, `eigvals`, `cholesky`, `qr`, `lq`, `lu`, `norm`, `cond`, `matrix_rank`, `matrix_power`, `trace`, `cov`

### FFT (10+ functions)

`fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `fftfreq`, `rfftfreq`

### Complex (8 functions)

`is_complex`, `has_complex_shape`, `to_complex`, `as_complex`, `as_real`, `real`, `imag`

### Random (15+ functions)

`rand`, `randn`, `randint`, `normal`, `uniform`, `randperm`, `seed`, `get_seed`, `rand_like`, `randn_like`, `exponential`, `gamma`, `beta`, `binomial`, `poisson`

### Cast & Indexing (15+ functions)

`cast`, `take`, `nonzero`, `flatnonzero`, `argsort`, `sort`, `masked_select`, `searchsorted`, `unique`, `topk`, `gather`, `scatter`, `scatter_add`, `scatter_reduce`, `interp`, `indices`, `ix_`

### Signal Processing (89 functions via `ins.signal.*`)

Access all signal functions through the `ins.signal` namespace:

```python
import insight as ins

# Windows
w = ins.signal.hann(256)
w = ins.signal.kaiser(256, beta=14.0)
w = ins.signal.get_window("hamming", 256)

# Filtering
b, a = ins.signal.firwin(101, 0.5)
filtered = ins.signal.filtfilt(b, a, signal)
resampled = ins.signal.resample(signal, num=1000)

# Spectral Analysis
f, Pxx = ins.signal.welch(signal, fs=1000.0)
f, t, Sxx = ins.signal.spectrogram(signal, fs=1000.0)

# Wavelets
coeffs = ins.signal.cwt(signal, ins.signal.morlet, widths)

# Radar
compressed = ins.signal.pulse_compression(tx, rx)
```

Signal submodules: `windows`, `waveforms`, `bsplines`, `filter_design`, `convolution`, `filtering`, `spectral`, `wavelets`, `acoustics`, `radar`, `demod`, `peak_finding`, `estimation`, `io`

### Plot (optional, requires `INSIGHT_USE_MATPLOT=ON`)

```python
# Available as ins.plot.* if built with matplotplusplus support
# 70+ plotting functions
```

## GPU Support

```python
# Transfer arrays between public devices
# Concrete GPU backend is selected by INSIGHT_GPU_BACKEND or initialization.
a_cpu = ins.zeros([100, 100], ins.float32)
a_gpu = a_cpu.to(ins.GPUPlace(0))   # cpu:0 -> gpu:0
a_back = a_gpu.to(ins.CPUPlace())   # gpu:0 -> cpu:0

# Operations auto-dispatch to the active backend
c = ins.matmul(a_gpu, a_gpu)  # runs on active GPU backend
```

## License

Part of the Insight7 framework. See [LICENSE](../../LICENSE).
