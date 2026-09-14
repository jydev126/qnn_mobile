# QNN 文件与产物地图

这篇文档解决：**`.pth / .onnx / .dlc / .bin / .so / executable` 分别是谁生成、谁读取、运行时出现在哪里。**

## 1. 一张总图

```text
训练 / PyTorch
    |
    | state_dict / checkpoint
    v
.pth / .pt
    |
    | PyTorch model + export
    v
.onnx / torch.export graph
    |
    | Qualcomm AI Hub / QAIRT converter + compiler
    v
.dlc
    |
    | libQnnModelDlc.so 读取
    | Compose + Finalize
    +------------------------------+
    |                              |
    | 直接 Execute                 | serialize context
    v                              v
QNN graph                        .bin
                                  |
                                  | runtime 恢复
                                  v
                              QNN graph
                                  |
                                  v
                              Execute
```

手机侧还需要另一组产物：

```text
Android ARM64 executable / .so        Hexagon .so
--------------------------------      ---------------------------
qnn-net-run                           libQnnHtpV79Skel.so
qnn-context-binary-generator
qnn-context-runner
libQnnHtp.so
libQnnModelDlc.so
libQnnSystem.so
libQnnHtpPrepare.so
libQnnHtpV79Stub.so
```

## 2. `.pth / .pt / .ckpt`

### 是什么

通常是 PyTorch 权重或 checkpoint。

它可能只保存：

```text
state_dict
```

也可能保存：

```text
model weights
optimizer state
epoch
训练配置
```

### 谁生成

训练代码 / PyTorch。

### 谁读取

PyTorch 模型源码。

### 能不能直接给 QNN 执行

不能。

QNN 不知道这个 checkpoint 对应什么 Python class，也不负责恢复 PyTorch 动态执行逻辑。

---

## 3. `.onnx`

### 是什么

一个框架中立的模型图交换格式。

典型链路：

```text
PyTorch model
   ↓ export
ONNX graph
```

### 谁生成

PyTorch ONNX exporter、模型仓库自己的 export 脚本等。

### 谁读取

QAIRT/QNN converter、ONNX Runtime、其他推理框架。

### 在 QNN 链路里的位置

对“自己部署一个开源模型”而言，ONNX 往往是重要中间层：

```text
PyTorch
→ export-friendly PyTorch
→ ONNX
→ QNN DLC
```

但并不是所有 QNN 流程都必须由用户手工经过一个 ONNX 文件；具体取决于使用的 Qualcomm 工具链入口。

---

## 4. `.dlc`

### 是什么

DLC 是 Qualcomm 工具链输出/使用的一种模型资产，包含图、权重以及相关 metadata。

当前项目的 RF-DETR 输入资产就是 DLC。

### 谁生成

Qualcomm AI Hub / QAIRT 模型转换编译流程。

### 谁读取

当前项目用：

```text
libQnnModelDlc.so
```

通过 `qnn-net-run` 或 `qnn-context-binary-generator` 加载 DLC。

### 它是不是 executable

不是。

不能：

```bash
./model.dlc
```

它需要 loader + backend 才能变成可执行 QNN graph。

### 当前项目里的命令关系

```text
qnn-net-run
  --model libQnnModelDlc.so
  --dlc_path model.dlc
  --backend libQnnHtp.so
```

其中：

- `--model` 是 DLC loader 动态库；
- `--dlc_path` 才是模型文件；
- `--backend` 决定图面向哪个 QNN backend。

---

## 5. QNN Context Binary：`.bin`

当前项目中的：

```text
rf_detr.bin
```

不是权重文件，也不是 Android executable。

它来自：

```text
DLC
→ Compose
→ Finalize
→ serialize QNN context
→ rf_detr.bin
```

所以可以近似类比 TensorRT：

```text
DLC                     ~ build 前模型资产
Context Binary          ~ serialized engine
```

但只是心智模型类比；QNN context 可以包含多个 graph，兼容规则也不同。

### 谁生成

当前实验：

```text
qnn-context-binary-generator
```

### 谁读取

可以是：

```text
qnn-net-run --retrieve_context
```

也可以是自己的 C++ 程序：

```text
QnnSystemContext / contextCreateFromBinary / graphRetrieve
```

### 它省掉什么

执行端不再需要重新：

```text
DLC load
Compose
Finalize
```

但仍需要：

```text
context restore
backend/device setup
input/output buffer
Execute
```

---

## 6. `.so`

`.so` 只是 Linux/Android 动态库后缀，不能只看扩展名判断职责。

当前项目至少有三类。

### 6.1 ARM CPU 侧 QNN runtime/backend

例如：

```text
libQnnHtp.so
libQnnSystem.so
libQnnHtpPrepare.so
libQnnModelDlc.so
libQnnHtpV79Stub.so
```

它们是 Android ARM64 动态库，由 CPU 侧进程加载。

### 6.2 Hexagon 侧库

例如：

```text
libQnnHtpV79Skel.so
```

这是 Hexagon V79 目标侧组件，不是 ARM64 `.so`。

即使后缀同为 `.so`，ELF 架构也不同。

### 6.3 模型自定义 op/plugin 动态库

某些 QNN 模型可能还会涉及用户自定义 op package，但**当前 RF-DETR 实验没有依赖一个单独的 `libMSDeformableAttention.so` 之类模型 plugin**。

因此不要看到特殊算子就默认需要生成 `.so`。优先判断能否将它改写为 QNN converter/backend 支持的普通图算子。

---

## 7. executable

例如：

```text
qnn-net-run
qnn-context-binary-generator
qnn-profile-viewer
qnn-context-runner
```

它们是真正能被 Android shell 启动的 ELF executable。

### Qualcomm 工具 executable

`qnn-net-run` 帮你做：

```text
参数解析
动态库加载
模型/context 加载
输入文件读取
tensor 分配
QNN execute
输出文件写入
profiling
```

### 自己的 executable

当前仓库：

```text
cpp/qnn_context_runner.cpp
```

自己负责：

```text
加载 backend/System API
恢复 context
查询 graph/tensor metadata
分配 buffer
填 Qnn_Tensor
graphExecute
写输出
释放资源
```

因此：

```text
Context Binary != executable
```

真正运行的是 runner；binary 是 runner 消费的模型资产。

---

## 8. `.raw`

QNN 工具经常使用 raw tensor 文件。

它不是图片文件格式，而是“tensor 内存按字节直接落盘”。

例如 RF-DETR：

```text
shape = [1, 3, 512, 512]
dtype = float32
bytes = 1 * 3 * 512 * 512 * 4
      = 3,145,728
```

文件本身通常没有：

```text
shape header
dtype header
layout header
```

所以读取者必须在文件外部已经知道 tensor contract。

---

## 9. 把这些产物放到“开发阶段”理解

### 模型训练阶段

```text
PyTorch source
+
.pth
```

### 模型导出阶段

```text
PyTorch
→ ONNX / export graph
```

### Qualcomm 转换阶段

```text
ONNX / export graph
→ DLC
```

### QNN prepare 阶段

```text
DLC
→ Compose
→ Finalize
→ Context Binary
```

### 手机产品运行阶段

```text
Android executable
+
QNN runtime .so
+
Context Binary
+
input tensor
→ Execute
```

对 UniAD 来说，真正困难的新工作主要发生在中间：

```text
PyTorch UniAD
→ 可静态导出的图
→ QNN 能接受的算子集合
→ DLC
```

而当前 RF-DETR 项目已经把 DLC 之后的执行链基本走通。