# Insight7 全项目 Code Review

审查日期：2026-07-27 · 基线 commit：`85c44d6` (main)
覆盖范围：core / ops 前端 / CPU backend / CUDA backend / Iluvatar backend / 语言绑定 / 构建·CI·测试·文档

---

## 结论摘要

框架的骨架（算子注册、设备 HAL、C ABI、四语言绑定、95k 行规模）是扎实的，但存在**两个系统性缺陷**和**一批静默返回错误结果的单点 bug**。共计约 80 项确认问题，其中 P0 级 2 类、P1 级 27 项。

最重要的判断：**当前的 630+/510+ 测试全绿并不能说明数值正确性**。绝大多数 bug 恰好落在测试没覆盖的角落——非连续视图、axis≠最后一维、b[0]≠1 的 IIR、广播的 batched matmul、F32 路径的 upfirdn。多处 bug 是"测试只验了返回值的一部分"才存活至今（例如 SVD 只断言奇异值 S，不验 U/VT）。

---

## P0-1：view offset / strides 契约缺失（跨所有层的系统性问题）

这是本次 review 影响面最大的一类，单一根因派生出至少 20 个 bug。

**根因**：`insight_array_is_contiguous`（`src/core/array.cpp:172-182`）只检查 strides，**不检查 offset**。切片产生的视图（如 `A["1:3, :"]`）strides 连续、offset≠0，因此通过了"连续"检查，而 kernel 直接从 `x->data` 基址线性读取。前端也没有统一的契约：有的 kernel 是 stride-aware 的（elementwise、flip），有的只检查输出（roll、pad、tril），有的完全盲读（where、put、lexsort）。`Array::data<T>()`（`array.h:138`）是无检查的 `static_cast`，让 dtype 错误也无法被发现。

### 已确认的实例

| 位置 | 问题 |
|---|---|
| `backends/cpu/kernels/unary/common.h:66,92,125` | **三个 unary 宏都漏加 `x->offset`/`out->offset`**，而同目录的 `elementwise/common.h:88-93` 是加了的。影响全部 ~40 个 CPU unary 算子：`sin(a["1:"])` 计算的是 `a[0..n-1]` |
| `backends/cuda/kernels/unary/*.cu` | 40 个 kernel 中 **39 个漏加 offset**，唯独 `exp.cu:19-20` 是正确的——证明这是疏漏而非约定 |
| `backends/cpu/kernels/cast/common.h:17-31` | `CAST_LOOP` 线性读源数据（忽略 strides+offset），同时 `copy_layout()` 把源视图的 strides 和 offset 盖到全新的连续输出上 → 输出数组声称有 offset，但缓冲区没有这段余量，**下游越界读**。`a[":, ::2"].to(f64)` 直接坏掉 |
| `backends/cpu/kernels/linalg/common.h:127` + 全部 linalg kernel | `cpu_ensure_contiguous` 只校验 strides，`inv(A["1:3,:2"])` 静默算的是父数组左上角 |
| `src/core/kernel.cpp:96-138` | **`do_cpu_fallback` 从基址 `a->data` 拷 `numel*dtype_size` 字节**，忽略 offset 和 stride 跨度。对带 offset 的 GPU 视图调用任何 C_FALLBACK 算子（cholesky/qr/lu/pinv/lstsq/eig/cov）→ CPU kernel 随后按 `host_buf + offset` 寻址，**堆越界读写**；回拷也写错位置 |
| `backends/cuda/kernels/indexing/where.cu:14-19` | 前端 `broadcast_arrays` 产出 stride-0 视图，kernel 按连续线性读 → **设备端越界读**。`where(mask, x, 标量)` 这种最常见用法即触发 |
| `backends/cuda/kernels/indexing/masked_select.cu:35-38` | 对 broadcast 后的 mask 视图做 `cudaMemcpy(n bytes)`，源缓冲区实际更小 |
| `backends/cuda/kernels/indexing/nonzero.cu:53-76` | 拷 `numel` 个元素到主机，却用 stride 派生的偏移去索引该缓冲区。4×4 数组的列切片 `a[:,0]`：偏移 0/4/8/12 索引进 4 元素的分配 → **主机堆越界** |
| `backends/cuda/kernels/linalg/matmul.cu:107-145` | batched 分支假设批次密集排布，前端却在 `broadcast_to` 之后才交给它（`src/ops/linalg.cpp:165-174`）→ `matmul(A(2,3,4), B(4,5))` 读到 B 缓冲区之外 |
| `src/ops/manipulation.cpp:404-493` | `roll`/`tril`/`triu`/`diag`/`pad` 前端不调 `.contiguous()`，对应 kernel 线性索引输入。`flip` 是唯一处理了输入 strides 的 |
| `backends/cpu/kernels/elementwise/mod.cpp:217-262`, `pow.cpp` | 这两个文件手写了循环而没用共享宏，因此漏了 offset |
| `bindings/lua/insight_lua.cpp:473-591`（`get`/`table`）<br>`bindings/python/insight_py.cpp:797-887`（`list()`） | CPU 数组上 `to(CPUPlace())` 是 no-op，随后按内存序线性读 → `a:transpose():table()` 返回错序。Python `numpy()` 是对的（先 `contiguous()`） |

### 建议

按此顺序修，成本从低到高：

1. **让 `insight_array_is_contiguous` 把 `offset != 0` 也判为非连续**（或新增 `is_dense_from_base()`），这一处改动能立刻堵住 linalg / fft / cast 一大片。
2. 给 `unary/common.h` 三个宏和 39 个 CUDA unary kernel 补 `offset`——机械改动，可脚本化。
3. 修 `do_cpu_fallback`：改为按 strides 逐元素搬运，或先在设备侧 `contiguous()` 再传输。顺带修掉它原地改写调用方 `InsightArray` 的设计（见下文"设计建议"）。
4. 在 `launch()` 入口加 debug 断言：kernel 声明自己需要连续输入时校验之，`data<T>()` 加 `dtype_of<T>() == dtype()` 断言。**这一条能在未来自动拦截整个 bug 类**。

---

## P0-2：PR #51（Iluvatar 合并）摧毁了 CUDA 测试覆盖

两个独立 agent 各自发现，证据一致。

`backends/iluvatar/` 是 `backends/cuda/` 的近乎逐字节 fork（226 文件 vs 226，211 个 `.cu` vs 211，语义差异只有 3 个文件）。问题不在 fork 本身，而在于**它是通过就地修改共享的 CUDA 测试源码来"接入"的**：

- **`tests/cuda/cpp/` 下 33 个测试文件中的 30 个被改成 `ins::init({"cpu", "iluvatar"})`**（如 `test_linalg.cpp:24`、`test_fft.cpp:26`、`test_cast.cpp:91`）。而 `src/init.cpp:325-330` 在指名的后端加载失败时**抛异常**。`tests/CMakeLists.txt:5,9,70` 让 `insight_tests_cuda` 和 `insight_tests_iluvatar` 用同一份源码 glob。
  → **在任何没有 CoreX SDK 的 NVIDIA 机器上，整个 CUDA 测试套件在 `SetUpTestSuite` 就崩掉**。如果两个后端都构建了，`insight_tests_cuda` 加载的是 Iluvatar 的 .so，根本没测 CUDA。
- **65 处无条件 `GTEST_SKIP() << "Iluvatar: ..."`**，且有 16 个 suite 级跳过写在 `SetUpTestSuite` 里（`test_fft.cpp:28` 干掉全部 19 个 FFT 测试；全部 15 个 `test_signal*.cpp`）。这些跳过**不是按运行时设备判断的**，所以在 NVIDIA 上也一样跳过。CPU 侧的可选依赖跳过是正确用 `#ifdef` 守卫的（`tests/cpu/cpp/test_signal_radar.cpp:117-120`），说明团队知道正确写法。
- 为此准备的 `tests/cuda/cpp/iluvatar_skip.h` 被**零个文件 include**，而且它自己就编译不过：`#pragma once` 开头，第 19 行却有个孤立的 `#endif`。
- **三个已知算错的 kernel 仍然全 dtype 注册**：`arange`、`logspace`（`tests/cuda/cpp/test_creation.cpp:176,284` 注明 "produces wrong results"）、`bincount`（`test_reduction.cpp:421,441`）。用户调用 `ins::arange(..., GPUPlace(0))` 拿到垃圾数据，无报错无警告无回退。
- **最可能的根因**：`backends/iluvatar/CMakeLists.txt:28` 给所有 `.cu` 加了 `-mllvm --skip-double`，但 `device/`、`registry/` 和整个 `insight_core` 由 host 编译器编译、没有这个 flag，同时 184 个 F64 registration 依然存在。16 个 kernel 通过 `void**` ABI 接收 host 侧传来的 `double` 标量——其中就包括 `arange.cu:28,59` 和 `logspace.cu`，即"唯二既接收 double 又拿它乘索引"的 creation kernel。

### 其他 Iluvatar 问题

- **`iluvatar_registry.cpp:39` 和 `iluvatar_device.cpp:30` 各自定义了一个 thread_local 错误字符串**。651 处 `gpu_set_last_error` 全写进前者，而注册给 HAL 的 getter（`:822`）读的是后者。→ **所有 Iluvatar kernel 错误信息都显示为 `Backend: no error`**。CUDA 侧是单一存储，正确的。
- `backends/iluvatar/CMakeLists.txt:22-33` 的 `add_custom_command` 只 `DEPENDS ${_src}`，改 `common.cuh` 不会触发 kernel 重编 → 静默的陈旧 `.o`。
- 同文件只传了 `-fPIC -std=c++17 --cuda-gpu-arch=...`，**没有 `-O2`/`-DNDEBUG`，完全忽略 `CMAKE_BUILD_TYPE`** → Release 构建里 Iluvatar kernel 全是 -O0。
- 检测路径 `/usr/local/corex`（根 `CMakeLists.txt:451`）与链接路径硬编码的 `/usr/local/corex-4.3.8/lib64`（`backends/iluvatar/CMakeLists.txt:13`）不一致 → 非 4.3.8 版本 configure 成功但链接失败。
- 注册到既有的 `INSIGHT_DEVICE_GPU`，**没有新增 Place/设备枚举** → CUDA 与 Iluvatar 无法共存，且自动发现按字母序（`src/init.cpp:164`）让 "cuda" 静默压过 "iluvatar"。
- `demos/CMakeLists.txt:31` 的 GPU 后端拷贝只守 `INSIGHT_WITH_CUDA`，demos 在纯 Iluvatar 构建下拿不到后端。
- `INSIGHT_WITH_ILUVATAR` 宏定义了但全项目无人使用；README/CLAUDE.md/QWEN.md **零处提及该后端**。

### 建议

1. **立即恢复 CUDA 可测性**：把 `tests/cuda/cpp` 的后端选择改成运行时条件式（先试 cuda，失败再试 iluvatar），把 65 处无条件 skip 换成按实际设备判断的宏——先把 `iluvatar_skip.h` 的 `#endif` 修好并真正用起来。
2. **注销已知算错的 kernel**：`arange`/`logspace`/`bincount` 的 Iluvatar registration 应删除，让它们走 C_FALLBACK 到 CPU，而不是返回垃圾。同理，若 F64 确实不被硬件支持，就不该注册 184 个 F64 kernel。
3. **消除 fork**：211 个文件为了 3 个文件的差异而复制，未来每个 CUDA 修复都要手工镜像（两边的 `common.cuh` 已经开始分叉）。宏间接层已经就位，改成同一份 kernel 源码用不同 registry 宏编译两次即可根除这类漂移。
4. 修错误字符串双存储、加 header 依赖、加 `-O2`、统一 CoreX 路径检测。

---

## P1：静默返回错误结果的单点 bug

以下每一项都会返回看起来合理但数值错误的结果，且都不报错。

### 数值 / 信号处理

| 位置 | 问题 |
|---|---|
| `backends/cpu/kernels/signal/filtering/lfilter.cpp:54,84` | **DF2T 漏了 `b[0]` 系数**：`val = xi[i] + z[0]`，正确应为 `b[0]*xi[i] + z[0]`。前端只按 `a[0]` 归一化（`src/ops/signal/filtering.cpp:358`），`b[0]` 一般≠1。任何真实 IIR 设计（Butterworth 等）输出量级全错。**测试没发现是因为所有 IIR 测试都用 `b=[1.0]`** |
| `src/ops/signal/filtering.cpp:652` | **`resample_poly` 丢弃了 `put` 的返回值**——`put` 是非原地的（`indexing.cpp:176` 内部 `prepared.copy()`）。`up_arr` 始终全零，整个函数对零信号做滤波，**返回值恒为 ~0** |
| `src/ops/signal/filtering.cpp:887,893,963,972` | `upfirdn` / `channelize_poly` 在 work_dtype 为 **F32** 时用 `data<double>()` 读取 → 重解释 + 读出分配 2 倍长度。F32 输入全坏 |
| `backends/cpu/kernels/linalg/svd.cpp:48,90` | **U 和 VT 以列主序直接写入行主序输出缓冲区**，无转换（对比 `eigh.cpp:42-46` 是正确转换了的）。`U @ diag(S) @ VT` 无法重建 A。**测试只断言 S**（C++ 与 Python 测试里 `_U_np`/`_Vt_np` 都故意没用） |
| `backends/cuda/kernels/linalg/eigh.cu:33-65` | CUDA 的 `syevd` 特征向量列主序写出且无转换，**相对 CPU/NumPy 是转置的**。注释"row-major = column-major (A = A^T)"对输入成立、对特征向量矩阵 Q 不成立。对应测试恰好被 `GTEST_SKIP` 了 |
| `backends/cpu/kernels/random/common.h:35-48` | **`set_seed` 是空操作**：getter 里是 `static seed{0}`，setter 里是另一个 `static global_seed{0}`。getter 永远返回 0，随机数永不可复现。现有测试 `DifferentSeedsProduceDifferentSequences` 是恒过的（RNG 本来就在推进），缺"同 seed 可复现"测试 |
| `src/ops/signal/spectral_analysis.cpp:146-148` | `csd` 只累加 `real(Pxy)`，丢弃虚部。互谱对 x≠y 是复数；`coherence`（`:208`）随后把 `\|Pxy\|²` 算成 `(Re Pxy)²`，有相位差时结果全错 |
| `src/ops/signal/spectral_analysis.cpp:189-193` | `periodogram` 硬编码转发 `nperseg=256` 给 `welch`。短于 256 点的信号**直接抛异常**，长信号被做了分段平均 |
| `src/ops/signal/filtering.cpp:541-604` | `resample` 构造的频谱有误：负频块 flip 后未 flip 回来就拼接；偶数 n 上采样时 Nyquist bin 被计入三次，数组长 `num+2`，末尾的"保证尺寸"截断切掉了真实负频内容。`axis` 参数被忽略 |
| `src/ops/fft.cpp:398-404` | **`hfft` = `irfft`，`ihfft` = `rfft`，逐字别名**。NumPy 定义为 `hfft(a,n)=irfft(conj(a),n)*n`，缺共轭和 n 倍缩放 |
| `src/ops/signal/filtering.cpp:460-486` | `filtfilt` 没有 scipy 的奇反射边缘延拓（`padlen=3*max(len(a),len(b))`）和 `lfilter_zi` 初始条件——同文件里 `lfilter_zi` 存在却没被用。IIR 场景边界瞬态很大 |
| `src/ops/signal/filtering.cpp:345-347` | `lfilter` 在 `a.numel()==1` 时转发 `firfilter`，返回 **full 卷积长度 `n+nb-1`**；IIR 路径返回 `n`。scipy 恒返回 `len(x)`。这个不一致进而驱动了 `filtfilt` 里的条件裁剪 hack |

### 形状 / 归约 / 索引

| 位置 | 问题 |
|---|---|
| `src/ops/reduction.cpp:266-275` | **`mean(x, axis, keepdim=false)` squeeze 错维度**——算出了 `ax` 却去 squeeze 最后一维。`mean((2,3,4), axis=1)` 返回 `(2,1,4)` 而非 `(2,4)`。只在 axis 就是最后一维时正确 |
| `src/ops/reduction.cpp:147-192` | `quantile`/`nanquantile`/`nansum`/`nanmean`/`nanvar` 的 `transpose_output` 是多余的（kernel 已按原相对序写出），keepdim=true 且 axis≠last 时返回 `(2,4,1)` 而非 `(2,1,4)`；keepdim=false 时 `post_process_keepdim` 又去 squeeze 一个已被丢弃的轴位 |
| `src/ops/indexing.cpp:815-830` | **`lexsort` 从不做字典序排序**：`batch_size` 把 keys 维也乘进去了，`nkeys` 恒为 1，退化成逐行独立 argsort。返回形状也是 `(k,n)` 而非 NumPy 的 `(n,)` |
| `src/ops/linalg.cpp:88-93` vs `:165-174` | **广播的 batched matmul 必然失败**：`contiguous()` 在 `broadcast_to` 之前调用，stride-0 视图到 kernel 后 `cpu_ensure_contiguous` 只检查不修复 → `(2,3,4) @ (4,5)` 抛 "non-contiguous array not supported" |
| `src/ops/linalg.cpp:29-37` | **linalg 对复数输入静默丢虚部**：`get_working_dtype` 把 C32/C64 映射成 F64，cast kernel（`cast_c64.cpp:47`）取 `.real()`。`matmul`/`det`/`inv`/`solve`/`svd` 对复数数组悄悄只算实部并返回实数 |
| `src/ops/indexing.cpp:189-191` | `put` 标量 values 广播成 stride-0 视图但 kernel 按 `v[i]` 读 → 越界 n-1 个元素；且 `put` **从不做 dtype 转换**（对比 `put_along_axis:238` 是做了的）→ `put(f32_x, idx, f64_vals)` 按错 dtype 重解释 |
| `src/ops/reduction.cpp:741-792` | `bincount` 不校验非负。kernel（`bincount.cpp:36-48`）直接 `dst[src[i]]++` → 含 -1 的输入**在输出缓冲区之前写入**，而且是 OpenMP 多线程写。NumPy 此处抛 `ValueError` |
| `src/ops/fft.cpp:504,548,594` | `fftn`/`ifftn`/`rfftn` 用原始（可能为负）的 axis 算 `pad_width[2*axis+1]`。`fft2` 默认 `axes={-2,-1}`，故 `fft2(x, s={8,8})` 对更小输入时**写 `pad_width[-3]`**（vector 下标下溢，UB） |
| `src/ops/fft.cpp:640-642` | `irfftn` 用 axis 的**值**而非位置去索引 `s`：应为 `s.back()`。`irfft2` 带 `s` 时 `s` 被静默忽略 |
| `backends/cpu/kernels/indexing/take.cpp:92,180` + `src/ops/indexing.cpp:111` | 前端不归一化用户传入的负 axis → kernel 读 `x->dims[-1]`（越界结构体读）；kernel 内负索引又用 `out_dims[axis]`（索引个数）而非源轴长度归一化 → `take(x_len5, [-1], axis=0)` 返回 `x[0]` 而非 `x[4]` |
| `src/ops/manipulation.cpp:430-437` + `roll.cpp:134` | `roll` 的 `axis=-1` 被当作"展平"哨兵，用户按 NumPy 习惯传 -1 表示最后一维时得到的是展平 roll。`fftshift`/`ifftshift` 同类陷阱（`-1` = 全部轴） |
| `src/ops/manipulation.cpp:474-477` | `diagonal` 接收 `axis1`/`axis2` 后直接丢弃，转发 `diag(x, offset)` |
| `backends/cpu/kernels/reduction/common.h:71-314` | `max`/`min`/`argmax`/`argmin` 静默跳过 NaN（NumPy 传播 NaN）；`nanmax`/`nanmin` 对全 NaN 切片返回 0（NumPy 返回 NaN） |
| `backends/cpu/kernels/indexing/searchsorted.cpp:103-119` | f16/bf16 按 `uint16_t`、f8 按 `uint8_t` 比较原始位模式，负浮点的排序是反的 |

### 错误状态未检查

| 位置 | 问题 |
|---|---|
| `backends/cpu/kernels/linalg/inv.cpp:66-108` | `inv_f32/f64` 在 LAPACK 失败时设置错误串并早退（`dst` 未初始化），但 `inv_kernel_cpu` **不检查错误、无条件返回 `C_SUCCESS`**（对比 `eigh.cpp:142` / `svd.cpp:169` 是检查了的）→ `inv(奇异矩阵)` 返回未初始化内存并报成功 |
| `backends/cuda/kernels/linalg/*.cu` | **`cusolver_check`/`cublas_check`（`linalg/common.cuh:58-98`）定义了但全项目零调用**；`d_info` 在 det/eigh/eigvalsh/inv/slogdet/solve/svd/svdvals 里分配又释放，**从不回读**。`solve(奇异A)` 返回垃圾并报成功；宽矩阵 `svd`（m<n，前端不转置）被 `gesvd` 拒绝后返回未初始化的 U/S/VT |

---

## P2：资源泄漏与线程安全

| 位置 | 问题 |
|---|---|
| `backends/cpu/kernels/fft/common_impl.cpp:25-27,38-40` | **FFTW planner 无互斥保护**，全项目未调用 `fftw_make_planner_thread_safe`。plan cache 是 thread_local 并不保护共享的 planner → 两线程并发 `fft` 会破坏 planner 状态 |
| 同上 `:20-44` + `fft_c2c.cpp:48-59` | **C2C plan 每次调用都新建且从不 destroy**（只有 R2C/C2R 的 cache 条目会 destroy）→ 每次 `fft`/`ifft` 泄漏一个 plan，长跑内存无界增长 |
| `backends/cpu/kernels/fft/irfft.cpp:44,53` | `fftw_execute_dft_c2r` **直接在 `x->data` 上执行**，c2r 默认销毁输入（plan 未加 `FFTW_PRESERVE_INPUT`），而前端 `prepare_input` 对已连续的输入不拷贝 → `spec = rfft(x); y = irfft(spec);` 之后 `spec` 变垃圾 |
| `backends/cuda/kernels/fft/irfft.cu:54-56,72-74` | 同样问题：cuFFT 的 out-of-place C2R 会覆写输入缓冲区 |
| `src/core/array.cpp:102,326-336`, `array.h:72-78` | **引用计数是裸 `int32_t*`**，且析构里"我是最后一个吗"的判断（`:328` 读）与递减（`:331`）是分离的 → 两线程并发析构同一数组的副本：要么双双认为不是最后一个（泄漏），要么计数撕裂后归零两次（**双重释放**）。（错误串本身是 `thread_local`，这块是对的） |
| `src/core/kernel.cpp:15-21` | kernel 注册表是无锁的 `unordered_map`；`load_backend()` 是用户可运行时调用的 API（绑定层暴露了），一边加载后端一边在另一线程 launch → rehash 期间读取会崩 |
| `backends/cuda/kernels/random/normal.cu:61-64` 等 | 每元素分配一个 `curandState`（~48 B/元素，1e8 元素需 4.8 GB），且 `cudaMalloc` 返回值被忽略 |
| `backends/cuda/kernels/elementwise/common.cuh:38-39,64-66` | `alloc_elementwise_metadata` 的 `cudaMalloc`/`cudaMemcpy` 返回值不检查，失败后所有 elementwise kernel 解引用未初始化的设备指针 |
| CUDA 全局 | **263/398 处 `blockIdx.x * blockDim.x + threadIdx.x` 缺 `(int64_t)` 转换**，乘积在 `unsigned int` 里算完才拓宽 → numel ≥ 2³² 时索引回绕。正确写法见 `signal/spectral/spectrogram_accum.cu:18` |

---

## P3：核心层的切片与视图语义 bug

| 位置 | 问题 |
|---|---|
| `src/core/array.cpp:830-856` | **`view(const Shape&)` 单参重载只更新 C++ 成员 `shape_`/`strides_`，不更新 `layout_.ndim/dims/strides`**，而所有 kernel 是从 `InsightArray` 读形状的。`cumsum(a.view({3,2}), 0)` 前端按 (3,2) 分配输出、kernel 按 {2,3} 遍历 → 静默错值。双参的 `view(Shape, DType)`（`:903-918`）是正确重写了 layout 的——单参这个纯属遗漏 |
| `src/core/array.cpp:888-922` + `src/ops/complex.cpp:54,69` | `layout_.offset` 以旧 dtype 的元素为单位，dtype 换掉后 `data()` 和 kernel 都按**新** dtype 解释它，却从未换算。`as_complex(a.at({1}))`：offset=4 个 float(16 B) 被当成 4 个 complex(32 B)。`real()`/`imag()`/`as_real()` 对称地按一半偏移寻址 |
| `src/core/slice.cpp:78` | 正步长时 `start` 被 clamp 到 `[0, dim_size-1]`，NumPy 是 `[0, dim_size]` → `a["10:"]` 在长度 5 的数组上返回 `[a[4]]` 而非空。同一行在 `dim_size==0` 时是 `std::clamp(start, 0, -1)`（`hi<lo` 是 UB） |
| `src/core/array.cpp:652` | 字符串索引里整数 `-1` 被转成 `Slice(-1, 0, 1)` → 长度 0 → 随后 `squeeze` 抛异常。**`a[":,-1"]` 取最后一列会报 "cannot squeeze"**；-2、-3 都正常，只有 -1 坏（因为 `-1+1=0` 没有回绕到 dim） |
| `src/core/array.cpp:580-583` | `operator[](const Slice&)` 用 `value_or(0)` / `value_or(dim)` 抹掉了 nullopt，负步长的默认值应是 `dim-1`/`-1` → `a[Slice(nullopt,nullopt,-1)]`（头文件里注明是"整体反转"）返回**空数组**。字符串路径 `a["::-1"]` 是对的，两条路径不一致 |
| `src/core/kernel.cpp:63-87` | `do_cpu_fallback` 的 `consider()` 把每个输入指针都当 `InsightArray*` 读 ~200 字节字段，但部分 launch 传的是裸标量指针（如 `fill_` 传栈上 `double` 的地址，`array.cpp:1037`）→ 越界读 |
| `src/core/shape.cpp:51-59` | 无参 `Shape::squeeze()` 遍历整个定长 `dims_[10]` 而非 `ndim_`，尾部的 0 因为 `0 != 1` 被保留 → `Shape({2,1,3}).squeeze()` 得到 9 维、numel=0 的畸形形状。目前无内部调用者，但是公开 API |
| `src/core/shape.cpp:22,34`, `array.cpp:75` | numel 累乘无溢出检查、不拒绝负维度（只有 `reshape()` 校验）→ `Shape({1<<32, 1<<32})` 有符号溢出，`Shape({-2,3})` 得到 numel=-6 |
| `src/ops/broadcast.cpp:27` | 空数组广播错误：`(0,)` vs `(1,)` 取 max 得 1（NumPy 得 0），随后给空操作数 stride 0 → 从零元素缓冲区读元素 0 |

---

## P4：类型提升策略碎片化

`src/ops/operator.cpp:11-13`、`src/ops/linalg.cpp:183` 等处存在四套互不一致的策略：

- `scalar_to_array` 按数组 dtype 构造标量 → **`int32_array / 2.5` 实际除以 2**，`int_array + 0.5` 加 0
- linalg 计算完 `to(a.dtype())` 强转回来 → `solve(A_f64, B_int32)` 把解截断成 int32；`matmul(f32, f64)` 静默降为 F32
- 整数 `div` 保持整数语义（C 风格），与项目宣称的 NumPy 对齐相悖（NumPy 真除法出浮点）
- `mod` 用 C 的 `%`/`fmod`（符号随被除数）：`mod(-7,3)` 得 -1，NumPy 得 2
- 整数 `div`/`mod` 除零直接 SIGFPE 崩进程（NumPy 返回 0 + 警告）
- `promote_types` 本身对 U16/U32/U64 和 I32+F16 直接抛异常

**建议**：抽一个统一的 promotion 模块，所有算子走同一条路径。这能一次性消灭一整类静默截断 bug。

---

## P5：工程 / CI / 文档

| 问题 | 说明 |
|---|---|
| **CI 从不编译 CUDA 或 Iluvatar 后端** | 6 个 workflow 的 configure 全部带 `-DINSIGHT_WITH_CUDA=OFF`，`backends/cuda/**`（116+ kernel 文件）和新合入的 `backends/iluvatar/**` **零编译检查**。破坏 CUDA 编译的 PR 会绿灯合入。README badge "tests-1140+ passed" 有一半没有 CI 支撑 |
| **`tests/python_align`（394+391 个测试）不被任何 workflow 运行** | `language_bindings.yml:93` 只跑 `tests/cpu/python/`。而且 `test_numerical_cpu.py:19-23` 在 ImportError 时做模块级 `pytest.skip` —— 即便接上 CI，导入坏掉也是静默绿 |
| **仓库里有 40 MB 商业音乐** | `audio/大梦何时了-叶炫清-210356449-2000.flac`（40,864,855 字节，已 tracked），被 `demos/cpp/sndfile_demo.cpp:50` 引用。永久的 history 膨胀 + Apache-2.0 仓库里再分发商业录音的版权风险。建议换成生成的或 CC0 素材并清理 history |
| **`INSIGHT_DYNAMIC_*_BACKEND=OFF` 是个自毁开关** | `CMakeLists.txt:19-20` 声明了选项，但两个后端的 CMakeLists 无条件 `SHARED`；该选项唯一的作用是跳过把 DLL 拷到测试目录旁 → `ins::init()` 的 dlopen 失败，全部测试运行时挂掉。CLAUDE.md 的构建选项表把它当正常选项介绍 |
| **提交的 Find 模块硬编码了开发机路径** | `cmake/FindOpenBLAS.cmake:6-7,21-22`（`E:/code/insight/deps/...`、`E:/code/vcpkg/...`）、`cmake/FindFFTW3.cmake:11,22,33`。换台机器"同样的构建"什么都找不到 |
| **音频 codec 的 FetchContent 回退逻辑是错的** | `CMakeLists.txt:215-222` 在还不知道是本地还是 fetch 之前，就把 `OGG_INCLUDE_DIR` 等 FORCE 缓存成 `third_party/ogg/...`。而 `third_party/` 里只有 googletest/pybind11/sol2 → 全新 checkout（含所有 CI runner）上这些路径不存在。正是 CLAUDE.md 陷阱 #9，只修了一半 |
| **PR 的 path filter 比 push 窄** | `cpu_tests.yml` 的 push 触发含 `cmake/**`、`third_party/**`、`tests/CMakeLists.txt`，PR 触发（`:18-24`）三个都没有 → 只改 `cmake/FindOpenBLAS.cmake` 的 PR 不跑 CPU 测试。讽刺的是 `.qwen/skills/fix-ci-workflow-path-triggers` 记录的正是这类失败 |
| **全项目零 `install()` 规则** | `cmake --install` 是空操作，无法通过 `find_package` 被下游消费 |
| **文档与实际脱节** | 实际 `TEST(` 数：cpu 684、cuda 619、python_align 394/391，文档写 630+/510+/194+190。**Iluvatar 后端在 README/CLAUDE.md/QWEN.md 里零处提及**，而"CUDA 完全对齐/全部通过"的说法在 PR #51 之后已不成立 |
| **CLAUDE.md 的模板会误导后续开发** | ① CUDA kernel 模板要求 `extern "C"` 包裹，真实 CUDA kernel 一个都没这么写（`.qwen/skills/add-cuda-op.md` 说的"不需要 extern C"才是对的）；② "添加新算子流程"给的测试路径是 `tests/cpu/test_<module>.cpp`，而 `tests/CMakeLists.txt:4-5` 只 glob `cpu/cpp/*.cpp` → **按文档新建的测试文件永远不会被编译，静默不运行** |
| `.gitignore` 忽略 `.qwen/` 但 84 个 skill 文件已 tracked | 新增/重命名的 skill 会静默不入版本控制 |
| `bindings/julia/Insight.jl` 与 `bindings/julia/src/Insight.jl` 字节相同（各 88 KB） | 两份副本必然漂移，应只留 `Project.toml` 实际使用的 `src/` 那份 |

### 关于 `.qwen/skills/`（抽样 11/84）

**建议保留。** 这些是高密度的项目特定事后总结，与代码实际情况吻合（悬垂指针、可选依赖守卫、atomicAdd 指导都技术正确）。两处需修正：`add-cuda-test/SKILL.md` 给的测试路径是过期的（同上，文件会静默不参与构建）；`add-cuda-op.md` 与 CLAUDE.md 在 `extern "C"` 上矛盾——**skill 是对的，该改 CLAUDE.md**。

---

## P6：语言绑定

| 位置 | 问题 |
|---|---|
| `bindings/julia/insight_julia_capi.cpp:332-478` 等 | **Julia 侧绝大多数算子没有 try/catch**（只有 cast/to_device/slice/take 等十来个有）。`ins::add` 形状不匹配时 `INS_CHECK` 抛出的 C++ 异常穿过 `ccall`，Julia 无法捕获 → **整个 Julia 进程 abort**。`Insight.zeros([2]) + Insight.zeros([3])` 就能杀掉会话 |
| `insight_julia_capi.cpp:188-201` vs `:161-183` | **列主序转换只做了一半**：`from_data` 反转维度、`shape_reversed` 反转回来，但 `zeros/ones/full/rand/randn` 和 Julia 侧 `reshape` 都不反转（辅助函数 `julia_to_insight_dims` 定义了从未被调用）。后果：`shape(zeros([2,3])) == [3,2]`；`getindex` 不反转索引 → `a[1,3]` 越界报错；**`matmul(a,b)` 在 Julia 自己的约定下算的是 `b*a`**。Julia 测试只断言 `numel`/`ndim`，全被掩盖 |
| `bindings/lua/insight_lua.cpp:215-240` | **`lua_spec_to_cpp` 对 spec 里每个正整数都减 1，包括 step**。`a["1:5:2"]` → `"0:4:1"`，步长 2 变 1，静默返回错数据。负步长 `::-1` 因为不动负数才幸存。Lua 测试从不测正步长 |
| `bindings/lua/insight/indexing.lua:74-107` | gather/scatter/scatter_add/scatter_reduce 的 wrapper 参数顺序与原生绑定（`insight_lua.cpp:2587-2602`）完全不符（wrapper 传 `(x, indices, axis)`，原生要 `(x, dim, index)`）→ **凡走 `require("insight.indexing")` 的调用运行时必挂**。同文件 `M.indices`/`M.ix_` 也有 table vs Shape usertype 的类型错配 |
| `bindings/lua/insight/init.lua:181-192` | `a[i] = v` 的整数赋值路径调用 `self:at(idx)`，但 usertype 上根本没注册 `at` 方法 → 落到字符串 `__index` 处理器，把 `"at"` 当切片 spec 解析 → 抛莫名其妙的切片错误 |
| `bindings/lua/insight/signal/filtering.lua:44-52` | `wiener` 把 `{mysize=..., noise=...}` 当 kwargs 传给位置参数的原生函数 → 参数被**静默丢弃**，滤波器按默认值跑 |
| `bindings/lua/insight/_wrap.lua:6-30` | 文档声称支持 `func{x, y}` 全位置表形式，实际纯数字键的表会整个当成第一个参数传下去。这个 fallthrough 同时也是 shape 表参数能工作的原因——两种情况设计上不可区分 |
| `bindings/python/insight_py.cpp:777-786` | 比较/算术 dunder 用普通 lambda 定义，**全文件零处 `py::is_operator()`**，且只接受 `Array` 重载 → `arr == 3`、`arr == None`、`arr in list` 都抛 `TypeError` 而非返回 `NotImplemented` |
| `bindings/python/insight_py.cpp:901-921` | `eye`/`arange`/`linspace`/`logspace` 的 `py::arg("place") = get_device()` 在**模块导入时求值一次**。`creation.py` 重新包装了 `zeros/ones/full` 用调用时的 `get_device()`，但这四个是裸导入 → `set_device(GPUPlace(0))` 后 `zeros` 在 GPU 而 `arange` 还在 CPU |
| `insight_julia_capi.cpp:1390-1395` | `lombscargle` 返回 `static thread_local Array` 的地址 → 第二次调用覆盖第一次的结果存储 |
| `src/Insight.jl:1081-1119` | `cast`/`transpose`/`copy` 等对返回 nullptr 的 C 函数不做 `C_NULL` 检查 → 下一次元数据调用解引用空指针段错误（`slice`/`to`/`reshape` 是检查了的） |
| `bindings/python/insight_py.cpp:91-152` | dtype 映射漏了 F16/BF16，尽管框架有 116 个半精度 kernel → `arr.numpy()` 对 float16 数组直接抛 "Unsupported dtype" |
| `insight_py.cpp` 全局 | **零处 `gil_scoped_release`** → 长时间运行的 kernel 阻塞所有 Python 线程 |

**做对了的地方**（已验证）：三种语言的 `load_backend` 都正确 catch 了 CUDA .so 缺失的异常；Python `numpy()`/`from_numpy` 都是拷贝语义，无悬垂缓冲区问题；Julia 的 `finalizer` + 空指针守卫的所有权模型在正常路径上是健全的；sol2 按值持有 `ins::Array`。

---

## 设计层面的建议（非 bug）

1. **`do_cpu_fallback` 原地改写调用方的 `InsightArray`**（`kernel.cpp:112-138`）——它临时把 `data`/`device_type` 改成 CPU 的值，而这个结构体就是 C++ `Array::layout_` 成员。任何并发观察者（哪怕只是另一线程读 `.place()`）会看到一个伪装成 CPU、且指针即将被释放的 GPU 数组。改成构造影子 CPU `InsightArray` 传给 CPU kernel，可同时解决这个和前面的异常安全漏洞（拷贝抛异常时会留下一堆永久指向已释放内存的数组）。

2. **拷贝赋值与移动赋值在视图上语义不对称**（`array.cpp:340-369` vs `382-402`）：形状/dtype 匹配时 `view = other` 是**写穿**共享存储，而 `view = std::move(other)` 是重新绑定。且视图的普通拷贝会继承 `is_view=1` → `Array w = v; w = x;` 会静默改写原始父数组。这个别名行为极易误触发，且随值类别而变。

3. **哨兵值与合法值冲突**：`axis=-1` 在不同函数里分别表示"全部轴"（fftshift）、"展平"（roll、take）、"最后一维"（fft、argsort）；`n=0`/`noverlap=0` 表示"默认"。`reduction.cpp` 已经在用 `std::optional`，应统一。

4. **`thread_local` 的默认设备 + GPU 惰性提升**（`place.cpp:247-266`）：主线程 `set_device(CPUPlace())` 对工作线程无效，每个新线程会独立地在 CUDA 已加载时惰性默认到 `GPUPlace(0)`。与非原子引用计数叠加后风险更大。

5. **Iluvatar 应该有自己的设备枚举和 Place**，而不是挤占 `INSIGHT_DEVICE_GPU`。

---

## 建议的修复顺序

**第一批（阻断性，建议尽快）**
1. 恢复 CUDA 测试可运行性（P0-2 第 1 条）——目前 GPU 侧实际处于无回归保护状态
2. 注销 Iluvatar 上已知算错的 kernel（arange/logspace/bincount）
3. `lfilter` 补 `b[0]`、`resample_poly` 接住 `put` 返回值、`upfirdn`/`channelize_poly` 的 F32 路径、`set_seed`
4. `bincount` 负值校验（这是内存损坏，不只是错值）
5. Julia 绑定给所有 ccall 入口加 try/catch（进程 abort 级别）
6. Lua `lua_spec_to_cpp` 不要动 step

**第二批（系统性，需要一次成体系的改动）**
7. `is_contiguous` 纳入 offset + unary 宏/39 个 CUDA kernel 补 offset + `do_cpu_fallback` 按 strides 搬运
8. `view(Shape)` 更新 layout、dtype 视图的 offset 换算
9. CUDA linalg 检查 cuSOLVER status 与 `d_info`；CPU `inv` 检查错误状态
10. FFTW planner 加锁 + C2C plan 泄漏 + irfft 破坏输入
11. 引用计数改 `std::atomic`

**第三批（正确性对齐与工程化）**
12. 统一类型提升模块；SVD/eigh 的行列主序；csd 保留虚部；mean/quantile 的 squeeze
13. CI 增加 CUDA 编译检查（哪怕只是 compile-only）+ 接入 python_align
14. 修 CLAUDE.md 的两处错误模板（`extern "C"`、测试路径）、补 Iluvatar 文档、更新测试计数
15. 清理 40 MB FLAC 与 Find 模块里的硬编码路径
16. Julia 列主序约定二选一并贯彻到底

**可以不改的**：`.qwen/skills/` 整体保留（只修两处过期路径）；`shape.cpp` 无参 squeeze、Lua `a["0"]` 别名等纯粹 API 洁癖问题优先级最低。
