[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17/20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-1140%2B%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Insight

**輕量級、工業級張量計算框架，專為訊號處理與 GPU 加速設計。**

Insight 是一個 C++ 張量庫，設計理念深受 **PaddlePaddle**（算子註冊機制、裝置 HAL）、**Torch7**（簡潔的 C API、TH/THC 架構精神）和 **NumPy/CuPy**（Python 端 API 習慣）啟發。提供 CPU/GPU 統一計算、零拷貝視圖、動態算子派發和完整的訊號處理支援。

## 特性

- **統一 API** -- `insight::Array` 無縫在 `CPUPlace` / `GPUPlace` 間切換
- **零拷貝視圖** -- 透過 `strides` + `offset` 實現 `reshape`、`transpose`、`slice`
- **動態算子註冊** -- `ops()["add"][CPU][F32](args)` 派發（Paddle 風格）
- **裝置 HAL** -- 透過 `Device` 基底類別 + `extern "C"` 工廠實現 ABI 穩定的外掛系統
- **訊號處理** -- 89 個函數，涵蓋 14 個子模組（窗函數、波形產生、B 樣條、濾波器設計、卷積、濾波、頻譜分析、小波、聲學、雷達、解調、峰值偵測、參數估計、I/O），全部配有 CPU 與 CUDA 後端 kernel
- **半精度支援** -- fp16/bf16 支援，透過 `half_utils.h` / `half_utils.cuh` 實現，116 個 kernel 檔案包含半精度覆蓋
- **語言繫結** -- Python（pybind11）、Lua（sol2）、Julia（ccall），按模組拆分的 wrapper，訊號子命名空間
- **現代 C++** -- C++17/20，OpenMP 平行，FFTW3，OpenBLAS，cuBLAS，cuFFT
- **無自動微分** -- 保持函式庫輕量、專注

## 架構

```
insight/
├── include/insight/
│   ├── core/           # Array, Shape, Strides, DType, Place
│   ├── ops/            # 前端 API（elementwise, fft, signal, linalg 等）
│   ├── io/             # I/O（csv, print, sndfile）
│   ├── c_api/          # C ABI 介面（array, kernel, dtype, place）
│   └── plugin/         # 算子註冊 + 裝置 HAL
├── src/
│   ├── core/           # Array 實作、記憶體管理
│   ├── ops/            # 前端算子邏輯
│   └── internal/       # 內部工具
├── backends/
│   ├── cpu/kernels/    # CPU kernel（OpenMP + FFTW + OpenBLAS）
│   └── cuda/kernels/   # CUDA kernel（cuBLAS + cuFFT + Thrust）
├── bindings/
│   ├── python/insight/ # pybind11 繫結（按模組拆分的 wrapper）
│   ├── lua/insight/    # sol2 繫結（雙呼叫約定）
│   └── julia/          # ccall 繫結（Insight.jl）
├── tests/
│   ├── cpu/            # CPU 測試（630+ 測試，27 個套件）
│   ├── cuda/           # CUDA 測試（510+ 測試，23 個套件）
│   └── python_align/   # NumPy 精度對齊測試
└── demos/              # 範例程式（C++, Python, Lua, Julia）
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
    -DINSIGHT_USE_FFTW3=ON \
    -DINSIGHT_USE_OPENBLAS=ON
cmake --build . -j$(nproc)
```

**Windows (MSVC)：**

```powershell
# 前置要求：Visual Studio 2022+（C++ 工作負載）、CMake 3.15+、Ninja
# 透過 vcpkg 安裝依賴（建議）：
#   vcpkg install fftw3:x64-windows openblas:x64-windows
# 或從 https://github.com/OpenMathLib/OpenBLAS/releases 下載 OpenBLAS
#   解壓縮至如 C:\deps\OpenBLAS-0.3.33-x64

# 開啟 VS 開發者命令提示字元（x64）
# 請根據你的 Visual Studio 安裝路徑調整：
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

> **注意：** 如需繪圖功能，請安裝 [gnuplot](http://www.gnuplot.info/) 並確保其在系統 `PATH` 中。

### 安裝語言繫結

**Python**（需先完成 CMake 建置）：
```bash
pip install .
```

**Lua**（透過 luarocks，需先完成 CMake 建置）：
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

## 範例

### C++

```cpp
#include "insight/insight.h"
using namespace insight;

int main() {
    // 建立陣列（有 GPU 時自動選擇）
    Array a = ones({1000, 1000}, F32);
    Array b = randn({1000, 1000}, F32);

    // 矩陣乘法
    Array c = matmul(a, b);

    // NumPy 風格部分索引
    Array row = c.at({0});     // shape (1000,)
    Array val = c.at({0, 0});  // 標量

    // 訊號處理
    Array w = signal::hann(256);
}
```

### Python

```python
import insight as ins

# 有 GPU 時自動選擇（PaddlePaddle 行為）
print(ins.get_device())  # GPUPlace(0)

a = ins.rand([1000, 1000])
b = ins.randn([1000, 1000])

# 運算子：+, -, *, /, //, %, **, @
c = a @ b                # 矩陣乘法
d = a ** 2               # 逐元素冪
e = a // 3.0             # 地板除

# NumPy 風格索引
row = a[1]               # 部分索引 → shape (1000,)
val = a[1, 2]            # 標量提取
sub = a[1:, ::2]         # 混合切片

# 訊號處理
w = ins.signal.hann(256)
f, Pxx = ins.signal.welch(x, fs=1000)
```

### Lua

```lua
local ins = require("insight")
-- 後端自動偵測，有 GPU 時自動選擇

print(ins.get_device())       -- "gpu:0" 或 "cpu:0"
print(ins.gpu_version())      -- GPU runtime 版本，無 GPU 時為 0

local a = ins.rand({1000, 1000})
local b = ins.randn({1000, 1000})
local c = ins.matmul(a, b)

-- 1-based 索引（Lua 約定）
local row = a[1]              -- 部分索引 → shape (1000,)

-- 雙呼叫約定
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

-- 1-based 索引（Julia 約定）
row = a[1]                     -- 部分索引 → shape (1000,)
val = a[1, 2]                  -- 標量提取
```

## GPU 效能基準（A800-SXM4-80GB）

在百度 AI Studio 上測試，24 核 CPU + NVIDIA A800-SXM4-80GB：

| 測試 | CPU (24核) | GPU (A800) | 加速比 |
|------|-----------|------------|--------|
| add (2000萬元素) | 226ms | 601μs | **376x** |
| mul (2000萬元素) | 229ms | 609μs | **376x** |
| sin (2000萬元素) | 278ms | 771μs | **361x** |
| sum (2000萬元素) | 26ms | 8.8μs | **2,962x** |
| max (2000萬元素) | 42ms | 8.4μs | **4,976x** |
| matmul 256×256 | 19ms | 38μs | **503x** |
| matmul 1024×1024 | 3.6s | 110μs | **32,348x** |
| rfft2 512×512 | 4.5ms | 1.2ms | **3.7x** |
| randn (2000萬) | 766ms | 82ms | **9.4x** |
| sort (200萬) | 206ms | 187ms | **1.1x** |

> GPU 擅長大規模平行運算。小規模 FFT 和 SVD 有 kernel launch 開銷，CPU 更優。框架自動選擇最佳裝置。

## 依賴

| 依賴 | 版本 | 必需 | 說明 |
|------|------|------|------|
| CMake | 3.15+ | 是 | 建置系統 |
| C++17 編譯器 | -- | 是 | GCC 9+, Clang 12+, MSVC 2019+ |
| CUDA | 11.7+ | 否 | 可選 GPU 後端 |
| OpenBLAS | 任意 | 否 | CPU 線性代數 |
| FFTW3 | 3.3+ | 否 | CPU FFT |
| OpenMP | -- | 否 | CPU 多執行緒 |
| Thrust | 捆綁 | 否 | CUDA 排序/去重 |
| GoogleTest | 自動 | -- | 自動取得 |

## 測試狀態

**1140+ 測試全部通過** -- CPU（630+, 27 個套件）與 CUDA（510+, 23 個套件），另有 384 個精度對齊測試

### 語言繫結測試

| 語言 | 測試框架 | 測試數 |
|------|---------|--------|
| Python | pytest | 76 基礎 + 54 數值 + 384 對齊 |
| Lua | busted | 208 |
| Julia | Test stdlib | 74 |

## 範例程式

`demos/` 目錄下提供 4 種語言、6 個場景的範例程式：

| 範例 | C++ | Python | Lua | Julia |
|------|-----|--------|-----|-------|
| basic_ops | ✅ | ✅ | ✅ | ✅ |
| fft_demo | ✅ | ✅ | ✅ | ✅ |
| gpu_transfer | ✅ | ✅ | ✅ | ✅ |
| linalg_demo | ✅ | ✅ | ✅ | ✅ |
| radar_task1 | ✅ | ✅ | ✅ | ✅ |
| sndfile_demo | ✅ | ✅ | ✅ | ✅ |

## 授權條款

[Apache License 2.0](LICENSE)

版權所有 2026 PlumBlossomMaid

## 貢獻指南

歡迎提交 Issue 和 Pull Request。請確保：
1. 程式碼遵循 `.clang-format` 風格
2. 所有現有測試通過
3. 新功能包含對應測試
