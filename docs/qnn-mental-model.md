# QNN 心智模型：从 CUDA / TensorRT 类比到 HTP

这篇文档只回答一个问题：**QNN、QAIRT、HTP、Hexagon、Stub/Skel 分别是什么，它们在一次手机推理里各自负责什么。**

## 1. 先给结论

可以先用下面这个近似类比建立直觉：

| Qualcomm/QNN 世界 | CUDA/TensorRT 世界里的近似类比 | 本质 |
| --- | --- | --- |
| QAIRT | CUDA Toolkit + TensorRT + 配套工具集合 | SDK/工具集合，不是硬件 |
| QNN API | CUDA Runtime / TensorRT Runtime 这类宿主侧 API | 软件 API |
| `libQnnHtp.so` | TensorRT/CUDA 的宿主侧 runtime/backend | CPU 进程中加载的 backend 动态库 |
| HTP | GPU 计算设备这一层的近似位置 | Snapdragon SoC 内的 AI/DSP 计算硬件与执行环境，不是一套 Python/C++ 协议 |
| Hexagon | Qualcomm DSP/向量处理器架构家族 | 指令集/处理器架构体系 |
| V79 | 某一代 Hexagon/HTP 架构版本 | 目标硬件代际 |
| Stub | host 侧跨处理器调用配套库 | ARM CPU 侧组件 |
| Skel | DSP/HTP 侧真正被加载的配套库 | Hexagon 侧组件 |
| DLC | TensorRT build 之前的模型资产的近似物 | 模型图、权重和相关 metadata，不是 executable |
| QNN context binary | TensorRT serialized engine 的近似物 | 已针对 backend/目标准备后序列化的 context |
| `qnn-net-run` | `trtexec` | Qualcomm 提供的通用命令行 runner |
| 自己的 QNN C++ 程序 | 自己写 TensorRT runtime | 自己负责加载 context、tensor buffer 和 execute |

这个表只是为了建立心智模型，不表示两套系统的文件格式、兼容规则或执行机制完全相同。

## 2. HTP 到底是不是硬件

把 HTP 理解成“一个软件协议”会越看越乱。对当前项目，更有用的理解是：

```text
Android ARM64 应用 / qnn-net-run
        |
        | 调 QNN API
        v
libQnnHtp.so                  <- CPU/host 侧 backend
        |
        | RPC / driver / runtime 协作
        v
V79 Stub / Skel
        |
        v
HTP / Hexagon 执行资源        <- 真正跑神经网络计算的目标侧
```

CPU 没有消失。进程启动、文件读取、动态库加载、QNN API 调用、tensor buffer 准备、context 恢复等仍由 ARM CPU 侧程序参与；被 HTP backend 接受的图才会在目标加速器侧执行。

因此不要把“整个 Android 进程运行时间”都叫作 HTP/NPU 推理时间。

## 3. `libQnnHtp.so` 和 HTP 的区别

`libQnnHtp.so` 是软件库，运行在 ARM CPU 进程中。HTP 是它面向的目标 backend/硬件执行环境。

因此：

```text
libQnnHtp.so != HTP
```

更像：

```text
CUDA runtime / TensorRT runtime != NVIDIA GPU
```

项目中 `LD_LIBRARY_PATH` 用来让 Android 进程找到 ARM64 动态库；`ADSP_LIBRARY_PATH` 用来让 DSP/HTP 侧找到对应 Hexagon 库。两侧 ABI/架构不同，不能互换。

## 4. Stub / Skel 是什么

可以先把它们看成“CPU 与 Hexagon 两边一对配套组件”。

```text
ARM CPU                          Hexagon / HTP
   |                                  |
V79 Stub  <------ 跨处理器调用 ----> V79 Skel
```

Stub 是 ARM64 侧，Skel 是 Hexagon 侧。项目当前目标是 Snapdragon 8 Elite / SM8750 / HTP V79，所以部署的是 V79 对应组件。

这也是为什么不能只 push 一个 `libQnnHtp.so` 就认为 HTP runtime 已完整部署。

## 5. Compose / Finalize / Execute 应该放在什么位置理解

一个 DLC 真正被执行前，不是“把文件扔给 NPU”这么简单。

```text
DLC
 |
 | loader 读取模型
 v
Compose
 |  创建 graph / tensor / node 并连接
 v
Finalize
 |  backend 对图做目标相关准备
 v
Executable graph
 |
 v
Execute x N
```

Compose 是流程，不是一个通用 `QnnGraph_compose()` API。模型 loader 会调用一系列 QNN graph/tensor/node API 把模型组织出来。

Finalize 才是非常重要的 API 边界。对 HTP backend，它会触发目标相关的图准备、验证与优化。Finalize 成功后图才进入可执行状态。

## 6. Context Binary 是什么

Finalize 后的 context 可以序列化：

```text
DLC
 -> Compose
 -> Finalize
 -> serialize context
 -> model.bin
```

以后新进程可以直接：

```text
model.bin
 -> contextCreateFromBinary
 -> graphRetrieve
 -> Execute
```

因此 context binary 的价值主要是：**把 Compose/Finalize 从每次应用启动路径中移走。**

它仍然：

- 不是 Android executable；
- 需要兼容的 QNN/HTP runtime；
- 受 SDK、SoC、HTP 架构等兼容条件约束；
- 恢复 context 本身仍有成本；
- 不等于稳态单次 Execute 一定更快。

## 7. 当前项目里一次推理到底是谁在干什么

以当前 RF-DETR context 路径为例：

```text
qnn-context-runner                         ARM CPU executable
    |
    | dlopen / QNN API
    v
libQnnSystem.so + libQnnHtp.so             ARM CPU libraries
    |
    | 恢复 context / graph metadata
    | 准备输入输出 buffer
    | graphExecute()
    v
HTP V79                                     加速器侧执行图
    |
    v
output buffers
    |
    v
ARM CPU 程序写 raw 文件 / 后处理
```

所以日后看任何性能数据，都先问一句：

> 这个时间是整个进程 wall time、QNN API 时间、Finalize 时间，还是 accelerator execute 时间？

这比先纠结“它到底叫 DSP、NPU 还是 HTP”更重要。