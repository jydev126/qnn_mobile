# QNN 心智模型：用 CUDA / TensorRT 的层级重新理解 Qualcomm HTP

这篇文档不是 QNN API 手册。它解决的是在开始 UniAD 之前最容易混乱的几个基础问题：**RF-DETR 在当前项目里扮演什么角色；QNN、QAIRT、HTP、Hexagon、HVX、HMX、Stub/Skel 分别在哪一层；C++ 程序到底跑在 CPU 还是 NPU；Finalize 与 TensorRT build 可以类比到什么程度。**

当前项目固定讨论这条已经实机验证的路径：

```text
RF-DETR DLC
→ QAIRT 2.45 / QNN
→ libQnnHtp.so
→ Snapdragon 8 Elite / SM8750 / HTP V79
→ Samsung Galaxy S25 Ultra
```

不要先把 QNN 当成“另一套 TensorRT”。先把硬件层、runtime 层、模型层分开。

---

## 1. RF-DETR 在这个项目里是什么

RF-DETR 是一个 DETR 系目标检测模型。对当前 QNN 生命周期实验，不需要先深入它内部的 Transformer 细节，只需要把它看成一个真实但边界清楚的神经网络载体：

```text
RGB image
[1,3,512,512] float32
        ↓
     RF-DETR
        ↓
300 个候选检测
        ↓
boxes   [1,300,4] float32
scores  [1,300]   float32
classes [1,300]   int32
```

这一步最重要的认识是：**我们当前已经解决的是“拿到一个 DLC 后怎么在 HTP 上准备和执行”，不是“RF-DETR 为什么能被转换成 DLC”。**

因此 RF-DETR 是学习：

```text
DLC → Compose → Finalize → Context Binary → Execute
```

的实验载体。后面 UniAD 才会把问题推进到：

```text
PyTorch source → export-friendly graph → QNN-compatible operators → DLC
```

---

## 2. 先画完整层级：不要把 HTP、QNN、HMX 混成一个词

最有用的一张图是：

```text
模型层
────────────────────────────────────────────
RF-DETR / UniAD
PyTorch / ONNX / DLC / QNN graph

宿主软件层：ARM CPU
────────────────────────────────────────────
qnn-net-run / qnn-context-runner
        │
        ├── QNN API
        ├── libQnnSystem.so
        └── libQnnHtp.so          ← HTP backend 软件
                 │
                 │ driver / RPC / runtime 协作
                 ▼

Hexagon NPU / HTP 硬件执行侧
────────────────────────────────────────────
        Scalar / control resources
        HVX  vector/SIMD resources
        HMX  matrix/tensor resources
        local/shared/on-chip memory
                 │
                 ▼

SoC memory system
────────────────────────────────────────────
System LPDDR + cache / local memory hierarchy
```

这里有三个必须分开的对象：

```text
QNN             软件 API / runtime 体系
libQnnHtp.so    ARM CPU 进程加载的 HTP backend 动态库
HTP/Hexagon NPU 芯片上的 AI 加速执行目标
```

所以：

```text
libQnnHtp.so != HTP
QNN != HTP
HMX != HTP
```

---

## 3. HTP 到底是 CPU、GPU、DSP 还是 NPU

先给项目里最实用的结论：

> **对于 AI 部署，把 HTP / Hexagon 这一侧理解成 Qualcomm NPU 最合适。它不是 GPU，也不是 ARM CPU。**

术语之所以容易乱，是因为 Hexagon 从传统 DSP 架构长期演进到今天的 AI/NPU 架构，Qualcomm 的资料中会同时看到：

```text
Hexagon DSP
Hexagon NPU
HTP = Hexagon Tensor Processor
HVX = Hexagon Vector eXtensions
HMX = Hexagon Matrix eXtensions
```

现代 Qualcomm 对 Hexagon NPU 的描述强调的是 scalar、vector、tensor accelerator 的融合。公开 SoC 资料也能看到 HTP 与 Hexagon DSP、HVX、HMX 协同出现。

因此不要做两个错误等式：

```text
DSP = HTP           ×
HTP = Tensor Core   ×
```

更好的层级理解是：

```text
Hexagon NPU / HTP execution subsystem
        │
        ├── scalar/control 类资源
        ├── HVX：宽向量/SIMD
        └── HMX：矩阵/tensor acceleration
```

不同 SoC 代际的具体组织方式会变化，所以这张图是“职责层级”，不是 SM 框图的精确一一映射。

参考：Qualcomm Hexagon NPU 官方介绍与 Qualcomm 公开 SoC data sheet。

---

## 4. 和 NVIDIA 怎么对齐才不容易错

### NVIDIA 心智模型

```text
TensorRT / CUDA runtime
        ↓
NVIDIA GPU
        ↓
SM
 ├── 普通 FP/INT/SIMD execution
 └── Tensor Core
```

### Qualcomm 心智模型

```text
QNN runtime / libQnnHtp.so
        ↓
Hexagon NPU / HTP
 ├── scalar/control
 ├── HVX vector
 └── HMX matrix/tensor
```

近似类比：

| NVIDIA / TensorRT | Qualcomm / QNN | 应该怎么理解 |
| --- | --- | --- |
| TensorRT / CUDA host runtime | QNN + HTP backend | host 侧软件栈，非硬件 |
| NVIDIA GPU device | Hexagon NPU / HTP 执行侧 | 整体 AI accelerator 层级的近似位置 |
| Tensor Core | HMX | 都承担矩阵/tensor-heavy 加速角色，但 ISA/架构不同 |
| SIMD/普通 execution resources | HVX + scalar 等 | 只能做职责类比，不能换算“几个 CUDA Core” |
| `trtexec` | `qnn-net-run` | 通用命令行 runner |
| 自写 TRT runtime | `qnn-context-runner` | 自己管理 runtime 生命周期和 I/O |
| serialized TRT engine | QNN context binary | 都是已准备执行资产的近似类比 |

特别记住：

> **HMX ≈ Tensor Core 的“角色”；HTP 不是 Tensor Core。**

也不要追问“一个 HTP 等于几个 SM”。两个架构的执行模型和公开抽象层级并不支持这种换算。

---

## 5. QNN 是不是只能跑 HTP

不是。

QNN 是一套 backend 化的推理 API/runtime 体系。不同 SDK/设备可以提供不同 backend；常见理解包括 CPU、GPU、HTP 等执行目标。**当前项目明确选的是 HTP，因为命令里写了：**

```bash
--backend /data/local/tmp/qnn_mobile/runtime/libQnnHtp.so
```

所以当前实验的含义不是：

```text
QNN = NPU
```

而是：

```text
QNN graph/runtime
        +
选择 HTP backend
        ↓
Hexagon NPU/HTP 执行
```

类似：

```text
PyTorch API
 ├── CPU
 ├── CUDA
 └── MPS
```

但不能反过来说“CUDA 可以跑 CPU/GPU/NPU”。CUDA 本身就是 NVIDIA GPU 软件栈；同理 `libQnnHtp.so` 就是在选 HTP 路径。

---

## 6. 当前 RF-DETR 到底哪里跑在 CPU，哪里跑在 HTP

“RF-DETR 跑在 NPU 上”可以说。

“整个 C++ 程序跑在 NPU 上”不对。

当前 `qnn-context-runner` 是一个 Android ARM64 executable，它首先运行在 CPU：

```text
ARM CPU
────────────────────────────────────────
qnn-context-runner
    │
    ├── dlopen(libQnnHtp.so)
    ├── dlopen(libQnnSystem.so)
    ├── 读 rf_detr.bin
    ├── backendCreate / deviceCreate
    ├── contextCreateFromBinary
    ├── graphRetrieve
    ├── 分配 input/output std::vector
    └── graphExecute(...)
                │
                │ QNN backend / RPC / driver
                ▼

Hexagon NPU / HTP
────────────────────────────────────────
已 finalize/restore 的 RF-DETR graph
Conv / MatMul / attention / elementwise / ...
由 backend 的执行计划映射到目标硬件资源
                │
                ▼

ARM CPU
────────────────────────────────────────
graphExecute 返回
    ├── 读取 output client buffer
    └── 写 boxes.raw / logits.raw / classes.raw
```

所以以后说性能时必须问：

```text
进程总时间？
context restore？
QnnGraph_execute CPU wall time？
还是 accelerator profiling time？
```

项目 C++ 源码已经在计时代码旁明确写了：`graphExecute` 那段是 **CPU 侧同步 API wall time，包含 RPC 等，不等于 HTP 内部 kernel 时间**。

---

## 7. `libQnnHtp.so` 为什么看起来像“HTP 本身”

因为 backend 名字里也有 HTP。

但两层完全不同：

```text
HTP hardware
    ↑ 被驱动
libQnnHtp.so
    ↑ 被加载
ARM C++ process
```

可以粗略类比：

```text
NVIDIA GPU               Qualcomm HTP/NPU
CUDA/TensorRT runtime     QNN HTP backend
host C++ app              qnn-context-runner
```

`libQnnHtp.so` 的职责包括向 QNN 提供 HTP backend API，并协调目标侧 runtime。具体每个算子最后如何分配到 HMX/HVX/scalar、怎样做 tiling/fusion/memory scheduling，不是当前 C++ runner 逐算子控制的。

你的代码里不会写：

```cpp
run_on_hmx(node_a);
run_on_hvx(node_b);
```

就像正常 TensorRT runtime 里也不会为每层手工指定某个 Tensor Core。

---

## 8. Stub / Skel 为什么存在

当前 Android CPU 和 Hexagon 侧是不同执行环境，需要跨处理器的软件配套。

项目部署：

```text
ARM64 side                       Hexagon V79 side
──────────────────               ──────────────────
libQnnHtpV79Stub.so   <------>   libQnnHtpV79Skel.so
```

你可以把它们先理解成跨 CPU ↔ Hexagon 调用链两边的配套组件：

- Stub 是 ARM 侧；
- Skel 是 Hexagon 侧；
- 两边架构不同，不能互换；
- `LD_LIBRARY_PATH` 主要影响 CPU 侧动态库查找；
- `ADSP_LIBRARY_PATH` 为 Hexagon/DSP 侧库查找提供路径。

这也解释了为什么只 push `libQnnHtp.so` 并不代表完整 HTP runtime 已经部署好。

---

## 9. Compose / Finalize / Execute：和 TensorRT 最值得建立的类比

### TensorRT

你熟悉的是：

```text
network / ONNX
   ↓
builder
   ↓
tactic / kernel / memory planning / optimization
   ↓
engine
   ↓
execute
```

### 当前 QNN DLC 路线

```text
DLC
 ↓
libQnnModelDlc.so 读取
 ↓
Compose
    创建 graph/tensor/node/connectivity
 ↓
QnnGraph_finalize()
    HTP backend 做目标相关 prepare / optimization
 ↓
Execute
```

这里最重要的是：

- Compose 是一段流程，不存在一个通用 `QnnGraph_compose()`；
- `QnnGraph_finalize()` 才是明确 API 边界；
- Finalize 之后的图才能用于 Execute；
- backend 负责把逻辑 graph 变成适合目标 HTP 的执行状态。

所以可以用 TensorRT builder 建立直觉，但不要假设 Finalize 内部就是 TRT builder 的同构实现。

---

## 10. Context Binary 为什么像 TensorRT engine，但又不完全一样

当前项目的两条路径：

### DLC online prepare

```text
DLC
→ Compose
→ Finalize
→ Execute × N
```

### Context Binary

build 阶段：

```text
DLC
→ Compose
→ Finalize
→ serialize context
→ rf_detr.bin
```

运行阶段：

```text
rf_detr.bin
→ contextCreateFromBinary
→ graphRetrieve
→ Execute × N
```

Qualcomm AI Hub 文档把 DLC 描述为更 SoC-agnostic 的模型表示，把 Context Binary 描述为面向特定 HTP/SoC prepare 后的表示。

因此最实用的类比仍然是：

```text
DLC                 ~ build 前模型资产
Context Binary      ~ serialized engine
```

但不要由这个类比推出：

```text
.bin 是 executable                 ×
.bin 跨所有 SoC/SDK 都兼容          ×
恢复 .bin 就没有初始化成本           ×
.bin 一定让 steady Execute 更快      ×
```

当前实测正好证明最后一点不成立：DLC、context + net-run、context + C++ 的 steady Execute 都在同一量级；context 的收益主要是把约 10 秒级 Finalize 从运行阶段移走。

---

## 11. 一张最终的 NVIDIA ↔ Qualcomm 层级图

```text
NVIDIA                                      Qualcomm
────────────────────────────────────────────────────────────
model / ONNX                                model / DLC
       │                                           │
TensorRT builder/runtime                         QNN
       │                                           │
CUDA runtime/driver                       libQnnHtp.so
       │                                           │
NVIDIA GPU                               Hexagon NPU / HTP
       │                                  │       │       │
       │                                scalar    HVX     HMX
       │                                         vector  matrix/tensor
       │
SM / execution resources
       │
CUDA / Tensor Core
```

这不是 1:1 架构图，而是层级图。

只允许记三句话时，记：

```text
HTP / Hexagon NPU = AI accelerator 执行硬件这一层
HMX               = 最接近 Tensor Core 角色的 matrix/tensor 单元
libQnnHtp.so      = ARM CPU 侧驱动/使用 HTP 的 QNN backend 软件
```

---

## 12. 对后续 UniAD 最直接的意义

UniAD 迁移时你不会自己指定：

```text
这个 MSDeformableAttention MatMul 去 HMX
这个 sampling 去 HVX
```

你真正控制的是更高一层：

```text
PyTorch graph 怎么改写
shape/layout 是否静态
用哪些 QNN 可接受的 primitives
哪些内容留在 DLC 内
哪些后处理留 CPU
```

然后：

```text
export / converter
→ QNN graph
→ Finalize / Hexagon compiler/backend optimization
→ HTP execution plan
```

所以“研究算子支持”的目标不是学习 HMX 指令，而是**把 UniAD 表达成 QNN/HTP backend 能高质量 lower 的 graph。**

---

## 13. 本项目里已经证明与尚未证明的边界

已经证明：

```text
[✓] ARM64 C++ 能加载 libQnnHtp.so / libQnnSystem.so
[✓] V79 Stub/Skel 配套能在 S25 Ultra 工作
[✓] RF-DETR DLC 可 Compose + Finalize + Execute
[✓] context binary 可恢复并 Execute
[✓] 自写 C++ 的 I/O 与 qnn-net-run 输出完全一致
```

尚未由当前项目证明：

```text
[ ] 某个具体 node 到底落在 HMX 还是 HVX
[ ] RAW client buffer 是否零拷贝给 HTP
[ ] HTP 内部 activation 的具体物理地址/tiling
[ ] UniAD 的特殊算子已经能被 HTP 接收
```

这四类问题不要从“模型成功运行”反推答案。

---

## 参考入口

- Qualcomm Hexagon NPU: https://www.qualcomm.com/processors/hexagon
- Qualcomm AI Hub FAQ（DLC / Context Binary 与 Hexagon NPU）: https://dev.aihub.qualcomm.com/docs/hub/faq.html
- Qualcomm 公开 SoC Data Sheet 可用于理解 HTP/HVX/HMX 的硬件层级，但不同 SoC 的具体单元数量和 memory 规格不能直接套到 SM8750。
- 本项目源码：`cpp/qnn_context_runner.cpp`
- 本项目实测：`report/lifecycle-results.md`
