# QNN Tensor 与内存：shape、dtype、buffer 到底谁负责

这篇文档解决：**QNN tensor 是什么；shape/dtype/buffer 谁创建、谁拥有；CPU 和 HTP 之间的数据到底怎么走。**

## 1. 先把 tensor 分成两层

理解 QNN tensor 时最容易混淆的是：

```text
“tensor 的描述”
和
“tensor 的数据”
```

不是一回事。

可以把一个运行时 tensor 想成：

```text
Qnn_Tensor
├── name
├── id
├── type
├── dataFormat
├── dataType
├── rank / dimensions
├── quantization params
└── client buffer / memory handle
       |
       v
   真正的数据字节
```

前半部分回答：

> 这块数据逻辑上是什么？

后半部分回答：

> 数据实际放在哪里？

## 2. shape 和 dtype 从哪里来

对于已经存在的 QNN graph，输入输出 tensor 的 contract 是 graph 的一部分。

例如当前 RF-DETR：

```text
image
shape: [1, 3, 512, 512]
dtype: float32

boxes
shape: [1, 300, 4]
dtype: float32

logits
shape: [1, 300]
dtype: float32

classes
shape: [1, 300]
dtype: int32
```

自己的 C++ runner 恢复 context 之后，应该从 QNN System metadata / graph metadata 读取这些信息，而不是在程序里只凭经验猜。

这也是为什么 context runner 会先完成：

```text
读取 context metadata
→ 找 graph
→ 找 input/output tensor
→ 读取 name / dtype / dimensions
```

然后才有资格分配 buffer。

## 3. buffer 大小怎么算

最基本的 native tensor：

```text
bytes = product(shape) * sizeof(dtype)
```

例如：

```text
[1, 300] float32
= 300 * 4
= 1200 bytes
```

但：

```text
[1, 300] int32
```

也是 1200 bytes。

所以：

> **文件字节数相同，不代表 tensor dtype 相同。**

这是当前 RF-DETR `classes` 很典型的坑。

如果把 int32 的 1200 bytes 按 float32 解读，文件尺寸完全正确，数值仍然会错。

## 4. `Qnn_Tensor` 不是“显存对象”

不要把 `Qnn_Tensor` 直接等同于：

```text
CUDA device pointer
```

更准确地说，它是 QNN API 用来描述 graph tensor / runtime tensor 的结构。

执行时还需要告诉 backend：

```text
这个 tensor 的数据在哪里
```

最简单的路径通常是 client buffer：

```text
Qnn_Tensor
   |
   +-- clientBuf.data = host buffer pointer
   +-- clientBuf.dataSize = N bytes
```

也就是说，应用自己拥有一块普通内存，把指针交给 QNN。

## 5. 谁拥有这块内存

如果自己的程序使用 client buffer，生命周期通常是：

```text
应用 malloc / vector / new
        |
        v
把地址填进 Qnn_Tensor
        |
        v
QnnGraph_execute(...)
        |
        v
execute 返回
        |
        v
应用读取 output buffer
        |
        v
应用释放内存
```

QNN graph metadata 不会替你自动拥有这块应用 buffer。

所以自己的 runner 需要非常清楚：

```text
metadata 的生命周期
Qnn_Tensor wrapper 的生命周期
真正 data buffer 的生命周期
```

必须至少覆盖 execute 调用。

## 6. CPU buffer 是怎么让 HTP 用到的

在最简单的 client-buffer 心智模型中：

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

中间是否发生：

```text
copy
mapping
RPC transfer
shared memory
backend 内部 staging
```

取决于 backend、memory 类型、runtime 实现和配置。

因此不要看到代码里是 `void*` 就得出：

> HTP 直接把普通 Android heap 当自己的本地 SRAM 在算。

也不要看到 HTP 执行就得出：

> CPU 完全没有数据搬运成本。

从应用层正确的表述是：

> 应用通过 QNN tensor/memory contract 把输入输出 buffer 提交给 backend；具体跨处理器映射和搬运由 QNN/HTP runtime 负责。

## 7. 为什么这和 CUDA 的“显存”感觉不一样

CUDA 常见开发路径非常显式：

```text
cudaMalloc(device_ptr)
cudaMemcpy(H2D)
kernel<<<>>>()
cudaMemcpy(D2H)
```

所以开发者天然会问：

> 显存在哪里？

QNN 高层 runtime 路径更常看到：

```text
准备 Qnn_Tensor
填 client buffer
QnnGraph_execute
```

跨 CPU/HTP 的具体 memory orchestration 被 backend/runtime 包了更多。

所以不要强行寻找一个与 `cudaMalloc()` 一一对应的 API 才认为“内存管理开始了”。

## 8. Context Binary 里有没有输入输出数据

一般不要这样理解。

Context Binary 保存的是序列化的 QNN context / graph 准备结果，而不是“把每次推理输入也保存进去”。

启动后仍需要：

```text
读取 graph tensor contract
分配本次运行 buffer
填 input
Execute
读取 output
```

因此：

```text
context binary
!=
input/output memory snapshot
```

## 9. Graph 内部 tensor 和 graph I/O tensor 的区别

一个神经网络图中有大量中间 tensor：

```text
input
  ↓
conv output
  ↓
attention q/k/v
  ↓
intermediate activation
  ↓
...
  ↓
output
```

自己的应用通常只需要显式准备 graph I/O。

图内部 activation 的 memory planning 主要由 backend 在 graph prepare/finalize 后处理。

这也是 Finalize 很重要的原因之一：backend 不只是“检查语法”，它还要为目标执行环境准备整张图。

## 10. QNN 内存问题应该分三层看

### 第一层：模型逻辑 tensor

回答：

```text
name?
shape?
dtype?
layout?
quantization?
```

### 第二层：应用 I/O buffer

回答：

```text
分配多少字节？
谁拥有？
什么时候释放？
文件如何读进来？
output 如何写出去？
```

### 第三层：backend/accelerator 内存

回答：

```text
HTP 内部如何规划 activation？
哪些数据常驻？
哪里发生 copy/map？
峰值内存是多少？
```

第三层通常不能只靠 C++ 应用源码直接看出来，需要 backend profiling / 文档 / 工具信息。

## 11. 为什么 layout 对 UniAD 特别重要

QNN 能接受一个算子，不代表它以任意 tensor layout 都高效。

例如 attention 世界常见：

```text
[B, N, C]
```

而 accelerator 对：

```text
[B, C, H, W]
```

这种 CNN-like layout 可能有更成熟的优化路径。

Qualcomm 的 BEVFormer 适配就大量使用：

```text
NHWC <-> NCHW
Linear -> 1x1 Conv2d
```

目的不是改变数学意义，而是把 graph 变成更适合目标 backend 的静态 tensor 形态。

这对 UniAD 很重要：

> 算子支持问题经常同时也是 shape/layout 问题。

## 12. 调试 tensor 最有效的顺序

遇到输出异常时，不要先看模型精度。

按这个顺序查：

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

当前 RF-DETR 已经证明一个典型错误：

```text
classes int32
```

如果只看 `.raw` 文件和 byte size，很容易错误地按 float32 解码。

## 13. 对后面 UniAD 的直接意义

UniAD 导出时，每改写一个特殊模块都应该写清楚自己的 tensor contract：

```text
输入 name
shape
layout
dtype
哪些维度必须静态
输出 shape/layout
```

尤其是：

```text
BEV query
camera feature
reference points
sampling offsets
attention weights
prev_bev
```

如果这些 contract 不先固定，后面遇到 ONNX/QNN converter 报错时很难判断究竟是：

```text
算子不支持
还是
动态 shape 无法解析
还是
layout/reshape 导出失败
```

所以 UniAD 的算子迁移应当以“模块 + tensor contract”为最小验证单元，而不是直接拿整模型反复试编译。