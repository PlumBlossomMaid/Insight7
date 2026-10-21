# IXUCA 后端待验证项（2026-10-18 记录）

> 本文件记录 CODE_REVIEW P0-2 中**仍存活**、且因本机无 CoreX SDK
> 无法编译验证的 IXUCA 后端问题。已由本分支架构重构解决的 P0-2 项
> （iluvatar→ixuca 改名、CUDA 测试恢复 `ins::init()`、无条件 GTEST_SKIP
> 改运行时条件判断、arange/logspace/bincount 注销走 C_FALLBACK）不在
> 本清单内。
>
> 待 CoreX SDK 环境到位后，按下列顺序修复并在该机器上跑
> `insight_tests_cuda`（`INSIGHT_GPU_BACKEND=ixuca`）验证。

## 1. 错误字符串双存储（kernel 错误信息全部显示 "no error"）

**位置**：
- `backends/ixuca/registry/ixuca_registry.cpp:39` —
  `static thread_local std::string last_error;`
- `backends/ixuca/registry/ixuca_registry.h:23` —
  `#define gpu_set_last_error ixuca_set_last_error`（写前者）
- `backends/ixuca/device/ixuca_device.cpp:30` —
  `static thread_local std::string gpu_last_error_str;`
- `backends/ixuca/device/ixuca_device.cpp:822` —
  `iface->get_last_error = ixuca_get_last_error_ext;`（读后者）

**根因**：所有 IXUCA kernel 的 `gpu_set_last_error(...)` 经宏写入
registry 的 `last_error`，但注册给 HAL 的 getter 读的是 device 的
`gpu_last_error_str`——两者是不同存储，错误信息永远读不到。

**修法**（参照 `backends/cuda/device/cuda_device.cpp:25` 单一存储）：
统一为一份 thread_local 错误字符串，`gpu_set_last_error` /
`gpu_get_last_error` 与 HAL getter 读写同一份。

## 2. CMake `DEPENDS` 缺 `common.cuh`（增量编译失效）

**位置**：`backends/ixuca/CMakeLists.txt:33` —
`DEPENDS ${_src}` 只依赖 `.cu` 自身。

**根因**：kernel 都 include 各自目录的 `common.cuh`，改 `.cuh` 不会
触发对应 `.cu` 重编，容易拿到陈旧对象。

**修法**：`DEPENDS ${_src} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common.cuh`
（或按目录 glob 对应头文件）。

## 3. 编译命令无 `-O2` / `-DNDEBUG`（Release 也按 -O0 编）

**位置**：`backends/ixuca/CMakeLists.txt:27-33` 的 `add_custom_command`
只有 `-fPIC -std=c++17` 与架构/包含参数。

**根因**：CoreX clang++ 不走 CMake 的 `CMAKE_BUILD_TYPE` 默认优化
开关，Release 构建实际是 -O0。

**修法**：在 `add_custom_command` 里按 `CMAKE_BUILD_TYPE` 显式追加
`-O2 -DNDEBUG`（Release/Debug 分别处理）。

## 4. CoreX 路径硬编码不一致

**位置**：
- `backends/ixuca/CMakeLists.txt:13` —
  `set(IXUCA_LIB_DIR "/usr/local/corex-4.3.8/lib64")`
- 根 `CMakeLists.txt:479-485` 检测 `/usr/local/corex/bin/clang++` 与
  `/usr/local/corex/include/cuda_runtime.h`

**根因**：lib 路径写死带版本号 `corex-4.3.8`，与根 CMake 的
`/usr/local/corex` 检测不一致；SDK 升级换目录即失效。

**修法**：与根 CMake 一致统一从 `/usr/local/corex` 推导 lib 目录，
或引入缓存变量 `IXUCA_COREX_DIR` 允许显式指定。
