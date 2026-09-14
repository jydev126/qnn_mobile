# QNN Tensor 与内存：从 `std::vector` 到 HTP，到底谁分配、谁拥有、谁搬运

这篇文档解决一个非常容易被 CUDA 经验带偏的问题：**QNN tensor 到底是什么；当前 `qnn-context-runner` 的 input/output buffer 真正分配在哪里；手机 SoC 为什么没有一个与独显 VRAM 完全等价的“HTP 显存”；weights、activation、workspace 又是谁管理。**

先给当前项目最重要的结论：

> **你的 C++ runner 没有做任何类似 `cudaMalloc()` 的“HTP 显存分配”。它只在 Android ARM 进程里分配普通 `std::vector<uint8_t>`，把地址作为 `QNN_TENSORMEMTYPE_RAW` client buffer 交给 QNN。之后这些数据如何变成 HTP 可访问的数据、是否 staging/copy/map、cache 如何同步，是 HTP backend/runtime 的职责；当前源码不能证明零拷贝。**

---

## 1. 先把 CUDA 独显心智模型拆掉一半

你熟悉的独立 NVIDIA GPU 通常是：

```text
CPU system RAM                      GPU VRAM
────────────────                    ────────────────
host malloc                          cudaMalloc
    │                                    ▲
    ├──── cudaMemcpy H2D ────────────────┤
    │                                    │
    │                                GPU kernel
    │                                    │
    └──── cudaMemcpy D2H ◄───────────────┤
```

这里物理上存在非常强的：

```text
host memory
vs
device VRAM
```

边界。

Snapdragon 是 SoC，更适合先画：

```text
                    Snapdragon SoC

          CPU        GPU       Hexagon/HTP
           │          │             │
           └──────────┴─────────────┤
                                    │
                             memory subsystem
                                    │
                                  LPDDR
```

CPU、GPU、NPU/HTP 共享 SoC memory subsystem，并不等于所有单元永远直接从同一地址、同一 cache 层读数据；HTP 自己还会有 cache/shared/local/on-chip memory 层级。Qualcomm 的公开资料也会描述 HTP/HVX/HMX 以及 GMEM/VTCM 或大块 shared memory 这类片上存储。

因此在这个项目里最好少说“显存”，多说：

```text
system LPDDR / DDR
application client buffer
HTP-accessible memory
HTP local/shared/on-chip memory
QNN tensor buffer
```

不同 SoC 的具体 GMEM/VTCM 大小不能从别的芯片直接套到 SM8750。

---

## 2. QNN tensor 本身不是一块“显存”

QNN runtime 里的 tensor 至少要分两层：

```text
Qnn_Tensor_t
├── name / id / tensor type
├── dataType
├── rank
├── dimensions
├── dataFormat
├── quantization params
├── memType
└── clientBuf / memory handle
        │
        ▼
真正的数据字节
```

前半部分是 **tensor contract / metadata**：

```text
这个 tensor 叫什么？
shape 是什么？
dtype 是什么？
布局/量化是什么？
```

后半部分才回答：

```text
这次 execute 的实际数据在哪里？
```

所以：

```text
Qnn_Tensor_t != device pointer
```

更不能看到 `Qnn_Tensor_t` 就脑补成 CUDA 的 `CUdeviceptr`。

---

## 3. 当前项目的 shape / dtype 从哪里来

当前 C++ runner 并没有把 RF-DETR I/O contract 全部硬编码在代码里。

恢复 context 后先通过 `libQnnSystem.so` 读取 binary metadata：

```text
rf_detr.bin
    ↓
QnnSystemContext
    ↓
BinaryInfo
    ↓
GraphInfo
    ↓
graphInputs / graphOutputs
```

代码里的 `singleGraph()` 最终拿到：

```cpp
Qnn_Tensor_t *inputs;
Qnn_Tensor_t *outputs;
```

然后 `TensorBuffers` 根据 tensor metadata 建立运行时 buffer。

当前 RF-DETR contract 是：

```text
input
image    float32 [1,3,512,512]

output
boxes    float32 [1,300,4]
logits   float32 [1,300]
classes  int32   [1,300]
```

这件事很重要：

> **DLC/context 定义模型 tensor contract，应用根据 metadata 准备数据；不要仅凭模型名字猜 dtype/shape。**

---

## 4. `TensorBuffers` 到底分配了什么

当前源码：

```cpp
using Bytes = std::vector<uint8_t>;

struct TensorBuffers {
    std::vector<Qnn_Tensor_t> tensors;
    std::vector<Bytes> buffers;
    std::vector<std::string> names;
};
```

注意 `Bytes`：

```cpp
std::vector<uint8_t>
```

也就是说你的应用 buffer 首先只是：

```text
Android ARM64 进程 heap
        ↓
用户态虚拟地址
        ↓
std::vector<uint8_t>
```

代码为每个 tensor 算字节数：

```cpp
size_t bytes = 4;
for (...) {
    bytes *= t.dimensions[d];
}
```

当前只允许：

```text
float32 = 4 bytes
aint32   = 4 bytes
```

然后真正分配：

```cpp
buffers[i].resize(bytes);
```

到这里为止，发生的完全是普通 C++ host process memory allocation。

没有：

```cpp
cudaMalloc(...)
htpMalloc(...)
qnnMallocDeviceMemory(...)
```

当前程序里不存在这种操作。

---

## 5. RF-DETR 输入在内存里是什么样

输入：

```text
image
shape = [1,3,512,512]
dtype = float32
```

逻辑 payload：

```text
1 × 3 × 512 × 512 × 4
= 3,145,728 bytes
```

所以 `TensorBuffers` 做的可以粗略画成：

```text
Android qnn-context-runner 进程

heap / process virtual address space
┌──────────────────────────────────────┐
│ buffers[i] = std::vector<uint8_t>     │
│                                      │
│ 3,145,728 bytes                     │
│                                      │
│ RF-DETR image tensor payload        │
└──────────────────────────────────────┘
               ▲
               │
        buffers[i].data()
```

然后它把地址交给 QNN：

```cpp
t.memType = QNN_TENSORMEMTYPE_RAW;
t.clientBuf = {
    buffers[i].data(),
    static_cast<uint32_t>(bytes)
};
```

这几行是理解当前内存模型最关键的源码。

---

## 6. `QNN_TENSORMEMTYPE_RAW` 在这里意味着什么

对当前 runner 来说，它表达的是：

```text
这个 QNN tensor 使用 client 提供的 raw buffer
地址 = buffers[i].data()
大小 = bytes
```

生命周期也是应用负责：

```text
TensorBuffers 构造
    ↓
buffers.resize()
    ↓
clientBuf 借用 vector 的 data 指针
    ↓
readInputs() 填数据
    ↓
graphExecute()
    ↓
writeOutputs() 读结果
    ↓
TensorBuffers 析构 / vector 释放
```

源码注释自己就写得很清楚：

```text
vector 拥有 raw buffer；Qnn_Tensor_t.clientBuf 仅借用其地址
```

因此不能让 vector 先 reallocate/free，再拿旧 `clientBuf.data` 去 Execute。

---

## 7. HTP 是不是直接读取这个 `std::vector`

**当前源码不能证明。**

这是最需要克制的地方。

应用能证明的只有：

```text
CPU process virtual buffer
        ↓
QNN_TENSORMEMTYPE_RAW + clientBuf
        ↓
QnnGraph_execute
```

之后更合理的边界描述是：

```text
CPU client buffer
        ↓
QNN HTP backend
        ↓
RPC / driver / memory mapping / staging / synchronization
        ↓
HTP 可访问的数据
```

但下面这些细节当前都没有证据：

```text
是否发生 memcpy？
是否把 client buffer 映射成共享 memory？
是否有 staging buffer？
DMA 怎么参与？
cache flush/invalidate 怎么做？
HTP 地址空间映射成什么样？
```

所以不要写：

```text
“std::vector 被 HTP 零拷贝直接读取”    ×
```

也不要反过来武断写：

```text
“graphExecute 一定先完整 memcpy 到 NPU 显存” ×
```

正确说法是：

> **应用负责 client raw buffer；QNN HTP backend 负责让数据满足 HTP 执行所需的 memory contract。当前 runner 不控制也不证明 backend 内部的数据移动策略。**

---

## 8. Input / Output、Weight、Activation 是三种不同内存责任

### 8.1 Graph Input / Output

当前 C++ 程序显式管理。

```text
image
boxes
logits
classes
```

数据 owner：

```text
TensorBuffers::buffers
```

应用负责：

```text
allocate
读 input raw
检查 byte count
Execute 前保持有效
Execute 后读取 output
写 raw
释放
```

### 8.2 Model Weights

当前代码没有：

```cpp
malloc(weight_size);
copy_weight_to_htp(...);
```

Context 路径做的是：

```text
rf_detr.bin
    ↓
readFile()
    ↓
contextCreateFromBinary(...)
```

之后 weight 的目标布局、常驻策略、内部 cache/packing 等属于 QNN context + HTP backend/runtime 的实现范围。

应用只保留 binary buffer 和 QNN context 生命周期。

### 8.3 Intermediate Activation / Workspace

比如：

```text
Conv output
Q/K/V
attention score
FFN hidden
sampling workspace
临时 transpose/reshape 结果
```

当前 C++ runner 同样没有逐层分配。

这部分主要由已经 Finalize 的 graph/backend memory planning 管理。

所以和 TensorRT 的职责边界很像：

```text
应用：输入输出 binding
runtime/engine：weights / activation / workspace planning
```

但 Snapdragon memory architecture 不是独显 VRAM 模型。

---

## 9. HTP 自己还有片上快速存储，但不是你当前手写管理

“共享 LPDDR”不意味着：

```text
所有 HMX/HVX 运算永远直接从 DDR 逐元素取数
```

Qualcomm 的公开 Hexagon/HTP 资料会描述 shared/local memory、GMEM、Vector TCM 等层级。概念上：

```text
                  大 / 慢 / 共享
                  System LPDDR
                       │
                       ▼
             HTP shared/local memory
               cache / GMEM / VTCM
                       │
                       ▼
               HVX / HMX / scalar
                  小 / 快 / 近
```

真正图执行时需要：

```text
tiling
buffer reuse
activation lifetime planning
memory movement
fusion
```

但当前应用代码没有类似：

```cpp
copy_to_vtcm();
allocate_hmx_scratch();
```

这层主要被 Finalize/compiler/backend 隐藏。

因此做 UniAD 时，你首先优化的是 graph/layout/shape，让 backend 更容易做 memory planning，而不是自己先写 VTCM allocator。

---

## 10. 为什么同样 1200 bytes 仍可能完全读错

当前：

```text
logits  [1,300] float32 → 300 × 4 = 1200 bytes
classes [1,300] int32   → 300 × 4 = 1200 bytes
```

文件大小完全相同。

如果把 `classes.raw` 当 float32：

```text
文件长度检查仍然 PASS
数值解释完全错误
```

所以 tensor contract 的检查顺序必须是：

```text
name
→ dtype
→ rank/shape
→ byte size
→ layout
→ quantization
```

不能只比较 raw 文件长度。

当前 `TensorBuffers` 专门限制 float32/int32，就是为了避免“未知 dtype 也按 4 字节悄悄跑”。

---

## 11. Context metadata 的生命周期也属于内存问题

当前 `Session` 有一个很容易忽略的注释：

```text
System context 拥有 metadata 指针；QNN context 拥有恢复出来的 graph。
保留 System context 与 binary buffer 直到结束，避免浅拷贝 tensor 的悬空指针。
```

这意味着 `Qnn_Tensor_t` 中某些 metadata pointer 并不是你的 `tensors.assign()` 深拷贝出来的新内存。

所以要区分：

```text
Qnn_Tensor_t struct copy
vs
struct 内部 pointer 所指 memory ownership
```

这和 `clientBuf` 又是第三套生命周期。

在自己扩展 runner 支持多图、动态 shape、quantized tensor 时，这个问题会越来越重要。

---

## 12. `graphExecute()` 前后数据流到底能证明到哪

当前程序真实顺序：

```text
host file image.raw
        │
        │ readInputs()
        ▼
std::vector<uint8_t>
        │
        │ clientBuf
        ▼
Qnn_Tensor_t input
        │
        │ graphExecute()
        ▼
QNN HTP backend / target execution
        │
        ▼
Qnn_Tensor_t output clientBuf
        │
        ▼
std::vector<uint8_t>
        │
        │ writeOutputs()
        ▼
boxes.raw / logits.raw / classes.raw
```

当前代码对 `graphExecute()` 做的是 CPU wall timer，而且源码明确说明它包含 RPC 等同步成本，**不是纯 accelerator 内部计算时间**。

所以看到 C++ Execute 160 ms，不能说：

```text
“HTP 算了 160 ms”
```

只能说：

```text
“QnnGraph_execute 同步 API 在 CPU 侧返回用了约 160 ms”
```

accelerator 时间要看 backend profiling。

---

## 13. CUDA / TensorRT 对照表

| CUDA / TensorRT | 当前 QNN / HTP | 备注 |
| --- | --- | --- |
| host `malloc/std::vector` | `TensorBuffers::buffers` | 当前明确存在 |
| `cudaMalloc` device VRAM | 当前 runner **没有直接对应操作** | 不要硬找一一对应 |
| `cudaMemcpy H2D/D2H` | 当前 runner **没有显式对应 API** | 数据可达性由 HTP backend 处理 |
| TensorRT binding | `Qnn_Tensor_t + clientBuf` | 很接近的应用层职责类比 |
| TRT engine deserialize | `contextCreateFromBinary` | 恢复已准备资产 |
| TRT workspace/activation planning | QNN/HTP backend graph memory planning | 具体实现不同 |
| GPU L2/shared memory/register | HTP cache/shared/local/GMEM/VTCM 等 | 只能做层级类比 |
| GPU VRAM | **没有完全对应的独立 HTP VRAM 概念** | SoC system memory architecture |

---

## 14. UniAD 为什么会把这个内存问题放大

RF-DETR 输入很简单：

```text
1 × 3 × 512 × 512 float32 ≈ 3 MB payload
```

UniAD 的 BEV tensor 本身就很大。以：

```text
BEV = 200 × 200
C = 256
float32
```

仅一个逻辑 tensor payload：

```text
200 × 200 × 256 × 4
= 40,960,000 bytes
≈ 39.1 MiB
```

如果同时保留：

```text
current BEV
prev BEV
query/value/projection temporaries
multi-camera features
attention workspace
```

峰值内存很容易远高于单个 tensor payload。

注意这 **不是在估计 HTP 实际峰值**；只是说明为什么 UniAD 必须关心：

```text
layout
activation lifetime
是否重复 Expand/Tile
camera rebatch
multi-level feature
FP32 vs FP16/quantized payload
```

这也是 Qualcomm BEVFormer patch 大量改 layout、减少 reshape/tile、把 Linear 改 1×1 Conv 的现实意义之一。

---

## 15. 做 UniAD 子模块实验时应该记录哪些内存事实

每个 standalone module 固定一张 contract 表：

```text
Input:
name
shape
dtype
layout
logical bytes
owner
是否 graph input

Output:
name
shape
dtype
layout
logical bytes

Constants/state:
reference points
spatial_shapes
level_start_index
prev_bev
camera calibration
```

另外分清三种指标：

```text
logical tensor payload
process RSS / host memory
QNN/HTP backend reported peak memory
```

三者不是同一个东西。

---

## 16. 当前项目已经证明与没有证明什么

### 已经证明

```text
[✓] input/output raw buffer 是应用 std::vector 分配
[✓] Qnn_Tensor 使用 QNN_TENSORMEMTYPE_RAW/clientBuf
[✓] graph metadata 提供 name/shape/dtype
[✓] app 检查 byte count
[✓] Execute 后直接从 output client buffer 落盘
[✓] float32/int32 输出按 native dtype 处理
```

### 没有证明

```text
[ ] RAW buffer 是否 zero-copy
[ ] HTP 实际物理地址映射
[ ] staging/DMA/cache 的具体策略
[ ] weight 哪些常驻 LPDDR / local memory
[ ] activation 的具体 VTCM/GMEM 分配
[ ] 某层实际占多少 HTP local memory
```

这就是以后读 Qualcomm profiling/memory report 时的边界。

---

## 17. 最后只记一句话

> **当前 C++ 程序管理的是 CPU 侧 QNN tensor client buffer，不是“NPU 显存”；权重、内部 activation 和 HTP 局部存储的具体安排主要由 context/Finalize 和 HTP backend 管。**

这句话比“手机是 unified memory，所以都是零拷贝”准确得多，也比硬套 `cudaMalloc/cudaMemcpy` 模型更接近你现在真正写的代码。

---

## 参考入口

- 本项目：`cpp/qnn_context_runner.cpp`
- 本项目：`report/lifecycle-results.md`
- Qualcomm Hexagon NPU: https://www.qualcomm.com/processors/hexagon
- Qualcomm AI Hub FAQ: https://dev.aihub.qualcomm.com/docs/hub/faq.html
- Qualcomm 公开 SoC Data Sheet 可帮助理解 HTP/HVX/HMX/local-memory 的层级；不要把其他 SoC 的具体容量直接当作 SM8750 参数。
