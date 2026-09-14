# QNN Tensor 与内存：shape、dtype、buffer 到底谁负责

这篇文档解决：**QNN tensor 是什么；shape/dtype/buffer 谁创建、谁拥有；CPU 和 HTP 之间的数据到底怎么走。**

## 1. 先把 tensor 分成两层

理解 QNN tensor 时最容易混淆的是“tensor 的描述”和“tensor 的数据”。

可以把运行时 tensor 想成：

```text
Qnn_Tensor
├── name / id / type
├── dataType
├── rank / dimensions
├── quantization params
└── client buffer / memory handle
       |
       v
   真正的数据字节
```

前半部分回答“逻辑上是什么”，后半部分回答“数据实际放在哪里”。

## 2. shape 和 dtype 从哪里来

对于已经存在的 QNN graph，输入输出 tensor contract 是 graph 的一部分。自己的 C++ runner 恢复 context 后，应该从 System/graph metadata 读取这些信息，而不是只凭经验硬编码。

例如当前 RF-DETR：

```text
image    [1,3,512,512] float32
boxes    [1,300,4]     float32
logits   [1,300]       float32
classes  [1,300]       int32
```

## 3. buffer 大小怎么算

最基本 native tensor：

```text
bytes = product(shape) * sizeof(dtype)
```

`[1,300] float32` 和 `[1,300] int32` 都是 1200 bytes，所以：

> **文件字节数相同，不代表 tensor dtype 相同。**

当前 RF-DETR 的 `classes` 就是典型例子：按 float32 读取 int32，尺寸不会报错，但数值会错。

## 4. `Qnn_Tensor` 不是“显存对象”

不要把 `Qnn_Tensor` 直接等同于 CUDA device pointer。它首先是 QNN API 的 tensor 描述结构；执行时还要告诉 backend 数据在哪里。

最简单的 client-buffer 路径可理解为：

```text
Qnn_Tensor
   |
   +-- clientBuf.data = 应用 buffer 指针
   +-- clientBuf.dataSize = N bytes
```

## 5. 谁拥有这块内存

如果应用使用 client buffer，生命周期通常是：

```text
应用分配 buffer
→ 填入 Qnn_Tensor
→ QnnGraph_execute
→ execute 返回
→ 应用读取 output
→ 应用释放 buffer
```

QNN metadata 不会替应用自动拥有这块 buffer，因此 metadata、Qnn_Tensor wrapper 和 data buffer 的生命周期要分清。

## 6. CPU buffer 是怎么让 HTP 用到的

应用层只需要遵守 QNN memory contract：

```text
应用 CPU 内存
   |
   | QNN runtime/backend
   v
HTP 可访问/转换后的执行数据
   |
   | HTP Execute
   v
结果
   |
   v
应用可读 output buffer
```

中间是否发生 copy、mapping、RPC transfer、shared memory 或 staging，取决于 backend、memory 类型和 runtime 实现。

所以不要因为代码里是 `void*` 就认为 HTP 直接把 Android heap 当本地 SRAM；也不要因为 HTP 执行就认为 CPU 完全没有搬运成本。

## 7. 为什么它不像 CUDA 那么显式

CUDA 常见：

```text
cudaMalloc
cudaMemcpy H2D
kernel launch
cudaMemcpy D2H
```

QNN 高层路径更常看到：

```text
准备 Qnn_Tensor
填 client buffer
QnnGraph_execute
```

跨 CPU/HTP 的 memory orchestration 被 backend/runtime 包了更多，所以不要强行寻找一个与 `cudaMalloc()` 一一对应的 API 才认为“内存管理开始了”。

## 8. Context Binary 里有没有本次输入输出

不要这样理解。Context Binary 保存的是序列化 QNN context / graph 准备结果，不是每次推理的输入输出 snapshot。

启动后仍然需要：

```text
读取 tensor contract
分配本次 buffer
填 input
Execute
读取 output
```

## 9. Graph 内部 tensor 和 graph I/O tensor

神经网络中有大量内部 activation。应用通常只显式准备 graph I/O，内部 activation 的目标侧 memory planning 主要由 backend 在 prepare/finalize 后处理。

这也是 Finalize 很重要的原因之一：backend 不只是检查语法，还要为目标执行环境准备整张图。

## 10. 内存问题分三层看

### 模型逻辑 tensor

```text
name / shape / dtype / layout / quantization
```

### 应用 I/O buffer

```text
分配多少字节
谁拥有
何时释放
文件如何读写
```

### backend / accelerator 内存

```text
activation 如何规划
哪里发生 copy/map
哪些数据常驻
峰值内存多少
```

第三层通常需要 profiling、backend 文档或工具信息，不能只靠应用源码得出。

## 11. 为什么 layout 对 UniAD 特别重要

attention 世界常见：

```text
[B, N, C]
```

而 accelerator 常对规则 4D tensor 有成熟路径：

```text
[B, C, H, W]
```

Qualcomm 的 BEVFormer 适配大量使用 NHWC/NCHW 转换，以及把空间位置上的 `Linear` 改成 1x1 Conv2d。数学含义不变，但 graph 更接近 CNN/HTP 友好形态。

所以 UniAD 的算子支持问题经常同时也是 shape/layout 问题。

## 12. 调试 tensor 的顺序

遇到输出异常时优先：

```text
1. tensor name
2. rank / shape
3. dtype
4. byte size
5. layout
6. quantization parameters
7. input preprocess
8. output interpretation
9. 最后才比较模型数值
```

## 13. 对 UniAD 的直接意义

UniAD 每改写一个特殊模块，都应该固定 tensor contract：

```text
输入 name / shape / dtype / layout
哪些维度必须静态
输出 shape / layout
```

特别是：

```text
BEV query
camera feature
reference points
sampling offsets
attention weights
prev_bev
```

如果 contract 不先固定，converter 报错时很难判断是算子不支持、动态 shape 无法解析，还是 layout/reshape 导出失败。

所以后续应该以“模块 + tensor contract”为最小验证单元，而不是整模型反复试编译。