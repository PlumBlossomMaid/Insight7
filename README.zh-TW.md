[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17/20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-1324%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Insight

**輕量級、工業級 C++ Array 計算框架，專注訊號處理與 GPU 加速。**

Insight 是一個語言無關的 Array 函式庫，設計理念來自 **PaddlePaddle**（算子註冊、後端 HAL、公開裝置行為）、**Torch7**（乾淨 C API、TH/THC 精神）以及 **NumPy/CuPy**（陣列語義、strided view、互操作與 GPU 執行模型）。使用者側裝置模型保持簡單：`cpu:0` / `gpu:0`；CUDA、ROCm、IXUCA、SDAA 等具體 GPU 後端由初始化階段選擇，並只在診斷或明確後端選擇時暴露。

## 設計方向

Insight 的目標是成為語言無關的 NumPy/CuPy 核心，而不是 Python 優先再補繫結的函式庫。核心契約如下：

- **Array，不是 Tensor** -- 核心物件是 `ins::Array`，專注數值陣列、訊號處理和後端執行，不引入 autograd。
- **公開裝置簡單穩定** -- 使用者選擇 `cpu:0` 或 `gpu:0`；`cuda`、`rocm`、`ixuca`、`sdaa` 是診斷和明確選擇用的後端名，不是公開 place 字串。
- **單一活躍 GPU 後端** -- 一個行程只選擇一個 GPU 實作掛在 `gpu:*` 後面，可透過 `INSIGHT_GPU_BACKEND` 或 init options 指定。
- **顯式協定** -- Array storage/layout、dtype promotion、OpSchema、ArrayIterator、結構化 fallback、互操作都要顯式建模，不再靠 raw pointer 猜測。
- **Lua 程式碼生成** -- 重複的算子 schema/dispatch metadata 由 Lua 生成，schema 使用後端無關的 `host` / `device` adapter，由 CUDA/ROCm/SDAA 等後端消費。
- **跨語言一致性** -- C++、Python、Lua、Julia 共用 C++/C ABI 核心，語言差異（索引、axis、shape 約定）在繫結邊界處理。

完整架構契約見 [`docs/architecture.md`](docs/architecture.md)，Windows CUDA 繼續適配見 [`docs/windows-cuda.md`](docs/windows-cuda.md)。

## 特性

- **統一 API** -- `ins::Array` 在目前 `CPUPlace()` / `GPUPlace(0)` 上執行，公開裝置字串為 `cpu:0` / `gpu:0`。
- **後端選擇** -- CUDA、ROCm、IXUCA、SDAA 和未來 GPU 外掛都掛在統一 GPU place 後面；用 `INSIGHT_GPU_BACKEND=cuda|rocm|ixuca|sdaa` 強制選擇。
- **零拷貝視圖** -- `reshape`、`transpose`、slice、partial indexing 共用 storage、strides、offset。
- **結構化調度** -- OpSchema 和 schema-aware kernel launch 顯式區分 scalar attrs、arrays、fallback、writeback。
- **ArrayIterator** -- broadcast、contiguous fast path、非連續 strides、offset、reduction 統一表達，供 kernel 複用。
- **裝置 HAL** -- ABI 穩定外掛系統，覆蓋 allocation、copy、stream、event、profiler、diagnostics、kernel registration。
- **訊號處理** -- 89 個函數，覆蓋 14 個子模組：窗函數、波形、B 樣條、濾波器設計、卷積、濾波、頻譜、小波、聲學、雷達、解調、峰值偵測、估計、I/O。
- **半精度支援** -- fp16/bf16 透過 `half_utils.h` / `half_utils.cuh` 支援，在可用 CPU/GPU 路徑覆蓋。
- **語言繫結** -- Python（pybind11）、Lua（sol2）、Julia（ccall），按模組拆分 wrapper 和 signal 子命名空間。
- **互操作** -- Python NumPy protocol 與 DLPack 方向屬於核心設計，host/device copy 語義保持顯式。
- **無自動微分** -- 保持輕量，聚焦陣列計算。

## 架構

```
insight/
├── include/insight/
│   ├── core/           # Array, Shape, Strides, DType, Place, OpSchema, ArrayIterator
│   ├── ops/            # 前端 API（elementwise, fft, signal, linalg 等）
│   ├── io/             # I/O（csv, print, sndfile）
│   ├── c_api/          # C ABI（array, kernel, dtype, place, profiler）
│   └── generated/      # Lua 生成的算子 manifest 和 schema metadata
├── src/
│   ├── core/           # Array 實作、記憶體、schema launch、device init
│   ├── ops/            # 前端算子邏輯
│   └── internal/       # 內部工具
├── backends/
│   ├── cpu/            # CPU runtime + kernels（OpenMP + FFTW + OpenBLAS）
│   ├── cuda/           # CUDA 後端（cuBLAS + cuFFT + Thrust）
│   ├── rocm/           # ROCm/HIP 後端（環境可用時）
│   ├── ixuca/          # IXUCA 後端
│   └── sdaa/           # Tecorigin SDAA runtime 後端
├── bindings/
│   ├── python/insight/ # pybind11 繫結（按模組拆分）
│   ├── lua/insight/    # sol2 繫結（dual calling convention）
│   └── julia/          # ccall 繫結（Insight.jl）
├── tools/codegen/      # Lua 5.1 相容的後端無關生成器
├── tests/
│   ├── cpu/            # CPU 測試
│   ├── cuda/           # CUDA/SDAA 相容 GPU 測試
│   └── python_align/   # NumPy 精度對齊測試
└── demos/              # C++ / Python / Lua / Julia 範例
```

## 快速開始

### 從原始碼編譯

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

執行時選擇 GPU 後端：

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

更多 Windows CUDA 適配細節見 [`docs/windows-cuda.md`](docs/windows-cuda.md)。

> **注意：** 如需繪圖功能，請安裝 [gnuplot](http://www.gnuplot.info/) 並確保其在系統 `PATH` 中。

### Lua Codegen

```bash
cmake --build build --target insight_codegen
lua tools/codegen/gen.lua build/generated/codegen
```

生成器相容 Lua 5.1、5.2、5.3、5.4 和 LuaJIT。輸出是確定性的，會生成 `include/insight/generated/` 和 `docs/` 下的 manifest；schema 只使用後端無關的 `host` / `device` adapter，不把 CUDA/ROCm/SDAA 寫死在 IR 裡。

### 安裝語言繫結

**Python**（需先完成 CMake 建置）：
```bash
pip install .
```

**Lua**（透過 luarocks，需先完成 CMake 建置）：
```bash
luarocks make bindings/lua/insight-1.0-1.rockspec --local
```

**Julia**：
```julia
push!(LOAD_PATH, "/path/to/Insight7/bindings/julia")
using Insight
```

## 範例

### C++

```cpp
#include "insight/insight.h"
using namespace ins;

int main() {
    init();
    set_device(GPUPlace(0));  // 公開 place: gpu:0，具體後端由 init/env 選擇

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
print(ins.gpu_version())      -- active GPU runtime 版本，無 GPU 時為 0

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

## 依賴

| 依賴 | 版本 | 必需 | 說明 |
|------|------|------|------|
| CMake | 3.15+ | 是 | 建置系統 |
| C++17 編譯器 | -- | 是 | GCC 9+, Clang 12+, MSVC 2019+ |
| CUDA | 11.7+ | 否 | 可選 active GPU backend |
| ROCm/HIP | -- | 否 | 可選 active GPU backend |
| SDAA runtime | -- | 否 | 可選 active GPU backend |
| OpenBLAS | 任意 | 否 | CPU 線性代數 |
| FFTW3 | 3.3+ | 否 | CPU FFT |
| OpenMP | -- | 否 | CPU 多執行緒 |
| GoogleTest | 自動 | -- | 自動取得 |
| Lua | 5.1+ / LuaJIT | 否 | 僅 developer codegen target 需要 |

## 測試狀態

目前 feature branch 最新本地驗證：

| Suite | Result | Notes |
|-------|--------|-------|
| Full available C++ tests | 1324 / 1324 passed | CPU 加目前可用 GPU/SDAA-compatible suites |
| SDAA validation build | 1324 / 1324 passed | 結構化 fallback 與 runtime backend validation |
| Lua codegen | generated 23 ops | 已用本機 Lua interpreter 驗證 |

CUDA、ROCm、Julia 應在有對應 runtime 的機器上繼續驗證。Windows CUDA 繼續適配請按 [`docs/windows-cuda.md`](docs/windows-cuda.md) 執行。

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

## 授權條款

[Apache License 2.0](LICENSE)

版權所有 2026 PlumBlossomMaid

## 貢獻指南

歡迎提交 Issue 和 Pull Request。請確保：
1. 程式碼遵循 `.clang-format` 風格
2. 所有現有測試通過
3. 新功能包含對應測試
4. 未來工作依賴的架構/建置決策必須提交到倉庫文件
