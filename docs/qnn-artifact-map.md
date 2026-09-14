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

通常是 PyTorch 权重或 checkpoint，由训练代码生成，由 PyTorch 模型源码读取。它不是一个 QNN graph，QNN 不知道这个 checkpoint 对应什么 Python class，因此不能直接执行。

## 3. `.onnx`

ONNX 是模型图交换格式，典型链路是：

```text
PyTorch model
   ↓ export
ONNX graph
```

对“自己部署一个开源模型”而言，它往往是很重要的中间层：

```text
PyTorch
→ export-friendly PyTorch
→ ONNX
→ QNN DLC
```

但具体 Qualcomm 工具链入口可能隐藏部分中间步骤，不要求用户永远手工保存 ONNX。

## 4. `.dlc`

DLC 是 Qualcomm 工具链使用的一种模型资产，包含模型图、权重以及相关 metadata。当前项目的 RF-DETR 就从 DLC 开始。

当前工具调用：

```text
qnn-net-run
  --model libQnnModelDlc.so
  --dlc_path model.dlc
  --backend libQnnHtp.so
```

其中：

- `--model` 是 DLC loader 动态库；
- `--dlc_path` 才是模型文件；
- `--backend` 决定 QNN graph 面向哪个 backend。

DLC 不是 executable，不能直接 `./model.dlc`。

## 5. QNN Context Binary：`.bin`

当前项目中的 `rf_detr.bin` 来自：

```text
DLC
→ Compose
→ Finalize
→ serialize QNN context
→ rf_detr.bin
```

可以近似类比：

```text
DLC                     ~ TensorRT build 前模型资产
Context Binary          ~ serialized engine
```

但只是心智模型类比；QNN context 可以包含多个 graph，兼容规则也不同。

Context binary 可以由 `qnn-context-binary-generator` 生成，由 `qnn-net-run --retrieve_context` 或自己的 C++ runner 恢复。

它省掉的是运行端重新 DLC load / Compose / Finalize，而不是省掉所有初始化、I/O 和 Execute。

## 6. `.so`

`.so` 只是 Linux/Android 动态库后缀，不能只看扩展名判断职责。

### ARM CPU 侧 QNN 库

```text
libQnnHtp.so
libQnnSystem.so
libQnnHtpPrepare.so
libQnnModelDlc.so
libQnnHtpV79Stub.so
```

### Hexagon 侧库

```text
libQnnHtpV79Skel.so
```

即使都叫 `.so`，ELF 架构不同，不能互换。

### 模型自定义 op/plugin

某些 QNN 项目可能使用用户自定义 op package，但当前 RF-DETR 实验**没有**依赖一个单独的 `libMSDeformableAttention.so` 之类 plugin。因此遇到特殊算子时，优先判断能否改写为 converter/backend 支持的普通图算子，不要默认必须生成 custom-op `.so`。

## 7. executable

例如：

```text
qnn-net-run
qnn-context-binary-generator
qnn-profile-viewer
qnn-context-runner
```

它们是真正能由 Android shell 启动的 ELF executable。

`qnn-net-run` 帮你完成动态库加载、模型/context 加载、输入读取、tensor 分配、execute、输出和 profiling。自己的 `qnn-context-runner` 则自己做 backend/System API 加载、context 恢复、graph/tensor metadata、buffer 和 `graphExecute`。

因此：

```text
Context Binary != executable
```

## 8. `.raw`

Raw tensor 文件不是图片格式，只是 tensor 内存直接落盘，通常没有 shape/dtype/layout header。

例如 RF-DETR 输入：

```text
shape = [1, 3, 512, 512]
dtype = float32
bytes = 1 * 3 * 512 * 512 * 4
      = 3,145,728
```

读取者必须从文件外部知道 tensor contract。

## 9. 把产物放回完整开发阶段

```text
模型训练阶段
PyTorch source + .pth
        ↓
模型导出阶段
ONNX / export graph
        ↓
Qualcomm 转换阶段
DLC
        ↓
QNN prepare 阶段
Compose → Finalize → Context Binary
        ↓
手机产品运行阶段
Android executable + QNN runtime + Context Binary + input
```

对 UniAD 来说，真正困难的新工作主要是：

```text
PyTorch UniAD
→ 可静态导出的图
→ QNN 能接受的算子集合
→ DLC
```

而当前 RF-DETR 项目已经把 DLC 之后的执行链基本走通。