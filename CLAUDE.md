# Insight7 - Array 科学计算框架

## 项目定位

Insight 是一个轻量级、工业级 C++ Array 计算框架，用于信号处理、数值计算和 GPU 加速。项目目标不是做 PyTorch 兼容层，也不引入 autograd，而是提供一个语言无关的 NumPy/CuPy 风格数组核心。

设计灵感：

- **PaddlePaddle** — 算子注册、设备 HAL、公开设备行为和后端 discipline。
- **Torch7** — 干净 C API、TH/THC 精神、Lua 生态亲和。
- **NumPy/CuPy** — Array 语义、strided view、dtype promotion、互操作、GPU 执行模型。

核心原则：

- Insight 里面叫 **Array**，不要在代码或文档里把核心对象写成 Tensor。
- 公开设备只保留 `cpu:0` / `gpu:0` / `gpu:N`；不要暴露 `cuda:0`、`sdaa:0`、`rocm:0` 这类 vendor place。
- 一个进程只选择一个活跃 GPU 后端，后端名仅用于诊断或显式选择。
- 新改动不需要优先兼容旧 API；当前用户很少，架构正确性优先。
- 未来依赖的架构、构建、适配经验必须进入 git 仓库文档，不靠手工下载或本地笔记。

## 当前重构状态

当前 feature branch 已完成一次架构收口并推送到远端：

- 分支：`refactor/insight-architecture-host-device`
- 公开设备模型：`cpu:0` / `gpu:0`
- 活跃 GPU 后端选择：`INSIGHT_GPU_BACKEND=cuda|rocm|ixuca|sdaa`
- IXUCA 命名已替换旧写法。
- SDAA 已作为 `gpu` 后端实现 runtime / registry / device HAL / fallback validation。
- OpSchema、ArrayIterator、axis helpers、dtype metadata、structured CPU fallback 已进入核心。
- Lua codegen 已建立 backend-neutral schema，使用 `host` / `device` adapter，不写死 CUDA/ROCm/SDAA。
- Python interop 增加 NumPy/DLPack 相关入口。
- Lua 绑定增加 table validation 与 backend copy helper。

最新本机可用环境验证：

- Full available C++ tests：`1324 / 1324 passed`
- SDAA validation build：`1324 / 1324 passed`
- Lua codegen：generated 23 ops

CUDA、ROCm、Julia 需要在有对应 runtime 的机器上继续验证；Windows CUDA 适配见 `docs/windows-cuda.md`。

## 架构文档

主架构契约：`docs/architecture.md`

Windows CUDA 继续适配：`docs/windows-cuda.md`

Lua codegen：`tools/codegen/README.md`

这些文件是后续工作的 source of truth。若代码实现改变了架构或平台假设，必须同步更新这些文档并提交。

## 目录结构

```text
insight/
├── include/insight/
│   ├── core/           # Array, Shape, Strides, DType, Place, OpSchema, ArrayIterator
│   ├── ops/            # 前端 API 声明
│   ├── io/             # I/O
│   ├── c_api/          # C ABI 接口
│   └── generated/      # Lua 生成的 manifest/schema metadata
├── src/
│   ├── core/           # Array 实现、内存、schema launch、device init
│   ├── ops/            # 前端算子逻辑
│   └── internal/       # 内部工具
├── backends/
│   ├── cpu/            # CPU runtime + kernels
│   ├── cuda/           # CUDA backend
│   ├── rocm/           # ROCm/HIP backend
│   ├── ixuca/          # IXUCA backend
│   └── sdaa/           # Tecorigin SDAA runtime backend
├── bindings/
│   ├── python/         # pybind11
│   ├── lua/            # sol2
│   └── julia/          # ccall
├── tools/codegen/      # Lua 5.1-compatible generator
├── tests/
│   ├── cpu/
│   ├── cuda/
│   └── python_align/
└── demos/
```

## 构建

### Linux / macOS

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DINSIGHT_WITH_CUDA=ON \
  -DINSIGHT_WITH_ROCM=OFF \
  -DINSIGHT_WITH_SDAA=OFF \
  -DINSIGHT_BUILD_TESTS=ON
cmake --build . -j$(nproc)
```

运行时指定活跃 GPU 后端：

```bash
export INSIGHT_GPU_BACKEND=cuda
```

### Windows CUDA

```powershell
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
```

## 测试

```bash
# 全量 CTest
ctest --test-dir build --output-on-failure -j$(nproc)

# CPU 单模块
./build/tests/insight_tests_cpu --gtest_filter="ElementwiseTestCPU.*"

# CUDA/SDAA GPU 测试需从 build/tests 运行，确保 backend 动态库可被找到
cd build/tests
INSIGHT_GPU_BACKEND=cuda ./insight_tests_cuda --gtest_filter="ElementwiseTestGPU.*"
```

Python / Lua / Julia 绑定测试按当前 README 与 binding 子目录文档执行。Julia 当前是可选项，重心优先 Insight 本体和可用平台。

## Lua Codegen

```bash
cmake --build build --target insight_codegen
lua tools/codegen/gen.lua build/generated/codegen
```

约束：

- 生成器必须兼容 Lua 5.1、5.2、5.3、5.4、LuaJIT。
- schema 必须 backend-neutral，只使用 `host` / `device` adapter。
- 不要为 CUDA、ROCm、SDAA、IXUCA 各写一套 codegen。
- 生成输出必须确定、可 review，并纳入仓库管理。

## 添加或迁移算子

优先路径：

1. 查 `tools/codegen/schema/` 是否适合加入 schema。
2. 对简单 elementwise/cast/reduction，优先推进 codegen/ArrayIterator 路径。
3. 对复杂 linalg/FFT/signal，可保留手写实现，但 shape/dtype/fallback 仍应走显式协议。
4. CPU kernel 和 GPU kernel 都必须尊重 strides、offset、非连续 view。
5. 可能返回 `C_FALLBACK` 的 GPU 路径必须使用 schema-aware launch，不要再用 raw `void**` 猜类型。

## 常见陷阱

1. **非连续视图**：必须使用 strides/offset，不能假设连续。
2. **公开设备名**：用户文档和 CLI 用 `cpu` / `gpu`，后端名只做 diagnostics/selection。
3. **SDAA/IXUCA 命名**：不要写旧后端名；天数智芯写作 IXUCA。
4. **CUDA 测试悬垂指针**：`.to(CPUPlace()).data<T>()` 会悬垂，必须用命名变量持有返回的 Array。
5. **cuFFT 异步**：相关 kernel 返回前必须同步，避免后续 op 读到未完成数据。
6. **Julia 列主序**：Julia 绑定转换要集中处理，不要在算法里散落补丁。
7. **Lua 异常**：raw `lua_pushcfunction` 必须捕获 C++ 异常。
8. **CMake GLOB**：新增 build-time 产物时不要依赖 configure-time `file(GLOB)` 自动发现。
9. **文档管理**：未来适配 Windows/CUDA/ROCm/SDAA 的结论必须进入 git-tracked docs。
