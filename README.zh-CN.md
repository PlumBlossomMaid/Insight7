[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17/20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-1324%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Insight

**轻量级、工业级 C++ Array 计算框架，专注信号处理与 GPU 加速。**

Insight 是一个语言无关的 Array 库，设计理念来自 **PaddlePaddle**（算子注册、后端 HAL、公开设备行为）、**Torch7**（干净 C API、TH/THC 精神）以及 **NumPy/CuPy**（数组语义、strided view、互操作与 GPU 执行模型）。用户侧设备模型保持简单：`cpu:0` / `gpu:0`；CUDA、ROCm、IXUCA、SDAA 等具体 GPU 后端由初始化阶段选择，并只在诊断或显式后端选择时暴露。

## 设计方向

Insight 的目标是成为语言无关的 NumPy/CuPy 核心，而不是 Python 优先再补绑定的库。核心契约如下：

- **Array，不是 Tensor** -- 核心对象是 `ins::Array`，专注数值数组、信号处理和后端执行，不引入 autograd。
- **公开设备简单稳定** -- 用户选择 `cpu:0` 或 `gpu:0`；`cuda`、`rocm`、`ixuca`、`sdaa` 是诊断和显式选择用的后端名，不是公开 place 字符串。
- **单一活跃 GPU 后端** -- 一个进程只选择一个 GPU 实现挂在 `gpu:*` 后面，可通过 `INSIGHT_GPU_BACKEND` 或 init options 指定。
- **显式协议** -- Array storage/layout、dtype promotion、OpSchema、ArrayIterator、结构化 fallback、互操作都要显式建模，不再靠 raw pointer 猜测。
- **Lua 代码生成** -- 重复的算子 schema/dispatch metadata 由 Lua 生成，schema 使用后端无关的 `host` / `device` adapter，由 CUDA/ROCm/SDAA 等后端消费。
- **跨语言一致性** -- C++、Python、Lua、Julia 共用 C++/C ABI 核心，语言差异（索引、axis、shape 约定）在绑定边界处理。

完整架构契约见 [`docs/architecture.md`](docs/architecture.md)，Windows CUDA 继续适配见 [`docs/windows-cuda.md`](docs/windows-cuda.md)。

## 特性

- **统一 API** -- `ins::Array` 在当前 `CPUPlace()` / `GPUPlace(0)` 上运行，公开设备字符串为 `cpu:0` / `gpu:0`。
- **后端选择** -- CUDA、ROCm、IXUCA、SDAA 和未来 GPU 插件都挂在统一 GPU place 后面；用 `INSIGHT_GPU_BACKEND=cuda|rocm|ixuca|sdaa` 强制选择。
- **零拷贝视图** -- `reshape`、`transpose`、slice、partial indexing 共享 storage、strides、offset。
- **结构化调度** -- OpSchema 和 schema-aware kernel launch 显式区分 scalar attrs、arrays、fallback、writeback。
- **ArrayIterator** -- broadcast、contiguous fast path、非连续 strides、offset、reduction 统一表达，供 kernel 复用。
- **设备 HAL** -- ABI 稳定插件系统，覆盖 allocation、copy、stream、event、profiler、diagnostics、kernel registration。
- **信号处理** -- 89 个函数，覆盖 14 个子模块：窗函数、波形、B 样条、滤波器设计、卷积、滤波、频谱、小波、声学、雷达、解调、峰值检测、估计、I/O。
- **半精度支持** -- fp16/bf16 通过 `half_utils.h` / `half_utils.cuh` 支持，在可用 CPU/GPU 路径覆盖。
- **语言绑定** -- Python（pybind11）、Lua（sol2）、Julia（ccall），按模块拆分 wrapper 和 signal 子命名空间。
- **互操作** -- Python NumPy protocol 与 DLPack 方向属于核心设计，host/device copy 语义保持显式。
- **无自动求导** -- 保持轻量，聚焦数组计算。

## 架构

```
insight/
├── include/insight/
│   ├── core/           # Array, Shape, Strides, DType, Place, OpSchema, ArrayIterator
│   ├── ops/            # 前端 API（elementwise, fft, signal, linalg 等）
│   ├── io/             # I/O（csv, print, sndfile）
│   ├── c_api/          # C ABI（array, kernel, dtype, place, profiler）
│   └── generated/      # Lua 生成的算子 manifest 和 schema metadata
├── src/
│   ├── core/           # Array 实现、内存、schema launch、device init
│   ├── ops/            # 前端算子逻辑
│   └── internal/       # 内部工具
├── backends/
│   ├── cpu/            # CPU runtime + kernels（OpenMP + FFTW + OpenBLAS）
│   ├── cuda/           # CUDA 后端（cuBLAS + cuFFT + Thrust）
│   ├── rocm/           # ROCm/HIP 后端（环境可用时）
│   ├── ixuca/          # IXUCA 后端
│   └── sdaa/           # Tecorigin SDAA runtime 后端
├── bindings/
│   ├── python/insight/ # pybind11 绑定（按模块拆分）
│   ├── lua/insight/    # sol2 绑定（dual calling convention）
│   └── julia/          # ccall 绑定（Insight.jl）
├── tools/codegen/      # Lua 5.1 兼容的后端无关生成器
├── tests/
│   ├── cpu/            # CPU 测试
│   ├── cuda/           # CUDA/SDAA 兼容 GPU 测试
│   └── python_align/   # NumPy 精度对齐测试
└── demos/              # C++ / Python / Lua / Julia 示例
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
    -DINSIGHT_WITH_ROCM=OFF \
    -DINSIGHT_WITH_SDAA=OFF \
    -DINSIGHT_USE_FFTW3=ON \
    -DINSIGHT_USE_OPENBLAS=ON
cmake --build . -j$(nproc)
```

运行时选择 GPU 后端：

```bash
export INSIGHT_GPU_BACKEND=cuda   # 也可以是 rocm, ixuca, sdaa
```

**Windows CUDA（MSVC + Ninja）：**

```powershell
# 前置要求：Visual Studio 2022+ C++ workload、CMake、Ninja、CUDA Toolkit 11.7+
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"

cmake -S . -B build-cuda-win -G Ninja ^
    -DCMAKE_C_COMPILER=cl.exe ^
    -DCMAKE_CXX_COMPILER=cl.exe ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DINSIGHT_WITH_CUDA=ON ^
    -DINSIGHT_WITH_ROCM=OFF ^
    -DINSIGHT_WITH_SDAA=OFF ^
    -DINSIGHT_BUILD_TESTS=ON ^
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" ^
    -DCMAKE_PREFIX_PATH="C:/vcpkg/installed/x64-windows"
cmake --build build-cuda-win -j %NUMBER_OF_PROCESSORS%

$env:INSIGHT_GPU_BACKEND = "cuda"
$env:PATH = "$PWD\build-cuda-win\backends\cpu;$PWD\build-cuda-win\backends\cuda;$env:CUDA_PATH\bin;$env:PATH"
ctest --test-dir build-cuda-win -R "GPU|CUDA" --output-on-failure -j 8
```

更多 Windows CUDA 适配细节见 [`docs/windows-cuda.md`](docs/windows-cuda.md)。

> **注意：** 如需绘图功能，请安装 [gnuplot](http://www.gnuplot.info/) 并确保其在系统 `PATH` 中。

### Lua Codegen

```bash
cmake --build build --target insight_codegen
lua tools/codegen/gen.lua build/generated/codegen
```

生成器兼容 Lua 5.1、5.2、5.3、5.4 和 LuaJIT。输出是确定性的，会生成 `include/insight/generated/` 和 `docs/` 下的 manifest；schema 只使用后端无关的 `host` / `device` adapter，不把 CUDA/ROCm/SDAA 写死在 IR 里。

### 安装语言绑定

**Python**（需先完成 CMake 构建）：
```bash
pip install .
```

**Lua**（通过 luarocks，需先完成 CMake 构建）：
```bash
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
using namespace ins;

int main() {
    init();
    set_device(GPUPlace(0));  // 公开 place: gpu:0，具体后端由 init/env 选择

    Array a = ones({1000, 1000}, DType::F32);
    Array b = randn({1000, 1000}, DType::F32);
    Array c = matmul(a, b);

    Array row = c.at({0});
    Array w = signal::hann(256);
}
```

### Python

```python
import insight as ins

print(ins.get_device())               # "gpu:0" or "cpu:0"
print(ins.active_gpu_backend_name())  # "cuda", "sdaa", ... when GPU is active

a = ins.rand([1000, 1000])
b = ins.randn([1000, 1000])
c = a @ b

row = a[1]
sub = a[1:, ::2]
w = ins.signal.hann(256)
```

### Lua

```lua
local ins = require("insight")

print(ins.get_device())       -- "gpu:0" 或 "cpu:0"
print(ins.gpu_version())      -- active GPU runtime 版本，无 GPU 时为 0

local a = ins.rand({1000, 1000})
local b = ins.randn({1000, 1000})
local c = ins.matmul(a, b)
local row = a[1]
local w = ins.signal.hann{n=256}
```

### Julia

```julia
using Insight

dt, id = Insight.get_device()  # public kind id + device id；GPU 表示 gpu:0

a = Insight.rand(Int64[1000, 1000], Insight.float32)
b = Insight.randn(Int64[1000, 1000], Insight.float32)
c = Insight.matmul(a, b)
row = a[1]
```

## 依赖

| 依赖 | 版本 | 必需 | 说明 |
|------|------|------|------|
| CMake | 3.15+ | 是 | 构建系统 |
| C++17 编译器 | -- | 是 | GCC 9+, Clang 12+, MSVC 2019+ |
| CUDA | 11.7+ | 否 | 可选 active GPU backend |
| ROCm/HIP | -- | 否 | 可选 active GPU backend |
| SDAA runtime | -- | 否 | 可选 active GPU backend |
| OpenBLAS | 任意 | 否 | CPU 线性代数 |
| FFTW3 | 3.3+ | 否 | CPU FFT |
| OpenMP | -- | 否 | CPU 多线程 |
| GoogleTest | 自动 | -- | 自动获取 |
| Lua | 5.1+ / LuaJIT | 否 | 仅 developer codegen target 需要 |

## 测试状态

当前 feature branch 最新本地验证：

| Suite | Result | Notes |
|-------|--------|-------|
| Full available C++ tests | 1324 / 1324 passed | CPU 加当前可用 GPU/SDAA-compatible suites |
| SDAA validation build | 1324 / 1324 passed | 结构化 fallback 与 runtime backend validation |
| Lua codegen | generated 23 ops | 已用本机 Lua interpreter 验证 |

CUDA、ROCm、Julia 应在有对应 runtime 的机器上继续验证。Windows CUDA 继续适配请按 [`docs/windows-cuda.md`](docs/windows-cuda.md) 执行。

## Demos

`demos/` 提供 C++、Python、Lua、Julia radar/signal 工作流，通用 CLI flags：

```bash
--device cpu|gpu|all
--seed N
--iterations N
--timer
--profiler
--info
```

## 许可证

[Apache License 2.0](LICENSE)

版权所有 2026 PlumBlossomMaid

## 贡献指南

欢迎提交 Issue 和 Pull Request。请确保：
1. 代码遵循 `.clang-format` 风格
2. 所有现有测试通过
3. 新功能包含对应测试
4. 未来工作依赖的架构/构建决策必须提交到仓库文档
