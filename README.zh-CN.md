[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17/20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-1140%2B%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Insight

**轻量级、工业级张量计算框架，专为信号处理与 GPU 加速设计。**

Insight 是一个 C++ 张量库，设计理念深受 **PaddlePaddle**（算子注册机制、设备 HAL）、**Torch7**（简洁的 C API、TH/THC 架构精神）和 **NumPy/CuPy**（Python 侧 API 习惯）启发。提供 CPU/GPU 统一计算、零拷贝视图、动态算子派发和完整的信号处理支持。

## 特性

- **统一 API** -- `insight::Array` 无缝在 `CPUPlace` / `GPUPlace` 间切换
- **零拷贝视图** -- 通过 `strides` + `offset` 实现 `reshape`、`transpose`、`slice`
- **动态算子注册** -- `ops()["add"][CPU][F32](args)` 派发（Paddle 风格）
- **设备 HAL** -- 通过 `Device` 基类 + `extern "C"` 工厂实现 ABI 稳定的插件系统
- **信号处理** -- 89 个函数，覆盖 14 个子模块（窗函数、波形生成、B 样条、滤波器设计、卷积、滤波、频谱分析、小波、声学、雷达、解调、峰值检测、参数估计、I/O），全部配有 CPU 和 CUDA 后端 kernel
- **半精度支持** -- fp16/bf16 支持，通过 `half_utils.h` / `half_utils.cuh` 实现，116 个 kernel 文件包含半精度覆盖
- **语言绑定** -- Python（pybind11）、Lua（sol2）、Julia（ccall），按模块拆分的 wrapper，信号子命名空间
- **现代 C++** -- C++17/20，OpenMP 并行，FFTW3，OpenBLAS，cuBLAS，cuFFT
- **无自动求导** -- 保持库轻量、专注

## 架构

```
insight/
├── include/insight/
│   ├── core/           # Array, Shape, Strides, DType, Place
│   ├── ops/            # 前端 API（elementwise, fft, signal, linalg 等）
│   ├── io/             # I/O（csv, print, sndfile）
│   ├── c_api/          # C ABI 接口（array, kernel, dtype, place）
│   └── plugin/         # 算子注册 + 设备 HAL
├── src/
│   ├── core/           # Array 实现、内存管理
│   ├── ops/            # 前端算子逻辑
│   └── internal/       # 内部工具
├── backends/
│   ├── cpu/kernels/    # CPU kernel（OpenMP + FFTW + OpenBLAS）
│   └── cuda/kernels/   # CUDA kernel（cuBLAS + cuFFT + Thrust）
├── bindings/
│   ├── python/insight/ # pybind11 绑定（按模块拆分的 wrapper）
│   ├── lua/insight/    # sol2 绑定（双调用约定）
│   └── julia/          # ccall 绑定（Insight.jl）
├── tests/
│   ├── cpu/            # CPU 测试（630+ 测试，27 个套件）
│   ├── cuda/           # CUDA 测试（510+ 测试，23 个套件）
│   └── python_align/   # NumPy 精度对齐测试
└── demos/              # 示例程序（C++, Python, Lua, Julia）
```

## 快速开始

### 从源码编译

**Linux / macOS：**

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

**Windows (MSVC)：**

```powershell
# 前置要求：Visual Studio 2022+（C++ 工作负载）、CMake 3.15+、Ninja
# 通过 vcpkg 安装依赖（推荐）：
#   vcpkg install fftw3:x64-windows openblas:x64-windows
# 或从 https://github.com/OpenMathLib/OpenBLAS/releases 下载 OpenBLAS
#   解压到如 C:\deps\OpenBLAS-0.3.33-x64

# 打开 VS 开发者命令提示符（x64）
# 请根据你的 Visual Studio 安装路径调整：
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

> **注意：** 如需绘图功能，请安装 [gnuplot](http://www.gnuplot.info/) 并确保其在系统 `PATH` 中。

### 安装语言绑定

**Python**（需先完成 CMake 构建）：
```bash
pip install .
```

**Lua**（通过 luarocks，需先完成 CMake 构建）：
```bash
# Lua 5.3
luarocks make bindings/lua/insight-1.0-1.rockspec LUA_DIR=/usr --local

# LuaJIT
luarocks make bindings/lua/insight-1.0-1.rockspec --local
```

**Julia**：
```julia
push!(LOAD_PATH, "/path/to/Insight7/bindings/julia")
using Insight
```

## 示例

### C++

```cpp
#include "insight/insight.h"
using namespace insight;

int main() {
    // 创建数组（有 GPU 时自动选择 GPU）
    Array a = ones({1000, 1000}, F32);
    Array b = randn({1000, 1000}, F32);

    // 矩阵乘法
    Array c = matmul(a, b);

    // NumPy 风格部分索引
    Array row = c.at({0});     // shape (1000,)
    Array val = c.at({0, 0});  // 标量

    // 信号处理
    Array w = signal::hann(256);
}
```

### Python

```python
import insight as ins

# 有 GPU 时自动选择 GPU（PaddlePaddle 行为）
print(ins.get_device())  # GPUPlace(0)

a = ins.rand([1000, 1000])
b = ins.randn([1000, 1000])

# 运算符：+, -, *, /, //, %, **, @
c = a @ b                # 矩阵乘法
d = a ** 2               # 逐元素幂
e = a // 3.0             # 地板除

# NumPy 风格索引
row = a[1]               # 部分索引 → shape (1000,)
val = a[1, 2]            # 标量提取
sub = a[1:, ::2]         # 混合切片

# 信号处理
w = ins.signal.hann(256)
f, Pxx = ins.signal.welch(x, fs=1000)
```

### Lua

```lua
local ins = require("insight")
-- 后端自动检测，有 GPU 时自动选择

print(ins.get_device())       -- "gpu:0" 或 "cpu:0"
print(ins.gpu_version())      -- GPU runtime 版本，无 GPU 时为 0

local a = ins.rand({1000, 1000})
local b = ins.randn({1000, 1000})
local c = ins.matmul(a, b)

-- 1-based 索引（Lua 约定）
local row = a[1]              -- 部分索引 → shape (1000,)

-- 双调用约定
local w = ins.signal.hann(256)
local w2 = ins.signal.hann{n=256}
```

### Julia

```julia
using Insight

dt, id = Insight.get_device()  # (1, 0) 表示 GPU

a = Insight.rand(Int64[1000, 1000], Insight.float32)
b = Insight.randn(Int64[1000, 1000], Insight.float32)
c = Insight.matmul(a, b)

# 1-based 索引（Julia 约定）
row = a[1]                     -- 部分索引 → shape (1000,)
val = a[1, 2]                  -- 标量提取
```

## GPU 性能基准（A800-SXM4-80GB）

在百度 AI Studio 上测试，24 核 CPU + NVIDIA A800-SXM4-80GB：

| 测试 | CPU (24核) | GPU (A800) | 加速比 |
|------|-----------|------------|--------|
| add (2000万元素) | 226ms | 601μs | **376x** |
| mul (2000万元素) | 229ms | 609μs | **376x** |
| sin (2000万元素) | 278ms | 771μs | **361x** |
| sum (2000万元素) | 26ms | 8.8μs | **2,962x** |
| max (2000万元素) | 42ms | 8.4μs | **4,976x** |
| matmul 256×256 | 19ms | 38μs | **503x** |
| matmul 1024×1024 | 3.6s | 110μs | **32,348x** |
| rfft2 512×512 | 4.5ms | 1.2ms | **3.7x** |
| randn (2000万) | 766ms | 82ms | **9.4x** |
| sort (200万) | 206ms | 187ms | **1.1x** |

> GPU 擅长大规模并行运算。小规模 FFT 和 SVD 有 kernel launch 开销，CPU 更优。框架自动选择最优设备。

## 依赖

| 依赖 | 版本 | 必需 | 说明 |
|------|------|------|------|
| CMake | 3.15+ | 是 | 构建系统 |
| C++17 编译器 | -- | 是 | GCC 9+, Clang 12+, MSVC 2019+ |
| CUDA | 11.7+ | 否 | 可选 GPU 后端 |
| OpenBLAS | 任意 | 否 | CPU 线性代数 |
| FFTW3 | 3.3+ | 否 | CPU FFT |
| OpenMP | -- | 否 | CPU 多线程 |
| Thrust | 捆绑 | 否 | CUDA 排序/去重 |
| GoogleTest | 自动 | -- | 自动获取 |

## 测试状态

**1140+ 测试全部通过** -- CPU（630+, 27 个套件）和 CUDA（510+, 23 个套件），另有 384 个精度对齐测试

### 语言绑定测试

| 语言 | 测试框架 | 测试数 |
|------|---------|--------|
| Python | pytest | 76 基础 + 54 数值 + 384 对齐 |
| Lua | busted | 208 |
| Julia | Test stdlib | 74 |

## 示例程序

`demos/` 目录下提供 4 种语言、6 个场景的示例程序：

| 示例 | C++ | Python | Lua | Julia |
|------|-----|--------|-----|-------|
| basic_ops | ✅ | ✅ | ✅ | ✅ |
| fft_demo | ✅ | ✅ | ✅ | ✅ |
| gpu_transfer | ✅ | ✅ | ✅ | ✅ |
| linalg_demo | ✅ | ✅ | ✅ | ✅ |
| radar_task1 | ✅ | ✅ | ✅ | ✅ |
| sndfile_demo | ✅ | ✅ | ✅ | ✅ |

## 许可证

[Apache License 2.0](LICENSE)

版权所有 2026 PlumBlossomMaid

## 贡献指南

欢迎提交 Issue 和 Pull Request。请确保：
1. 代码遵循 `.clang-format` 风格
2. 所有现有测试通过
3. 新功能包含对应测试
