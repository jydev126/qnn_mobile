# QNN 文件与产物地图：`.pth / .onnx / .dlc / .bin / .so / executable / .raw` 到底是谁的东西

这篇文档不按“文件后缀百科”讲，而是回答四个工程问题：

```text
谁生成？
谁读取？
在哪台机器/处理器侧使用？
它是否已经和目标 SoC/HTP 绑定？
```

当前 RF-DETR 项目已经验证的是从 `.dlc` 往后的链路；UniAD 需要补的是 `.pth → export → DLC` 这一段。

---

## 1. 先看当前项目的真实链路

```text
开发/训练世界
────────────────────────────────────────────
PyTorch source + .pth/.ckpt
        │
        │ export
        ▼
ONNX / export graph
        │
        │ Qualcomm AI Hub / QAIRT conversion
        ▼
.dlc

QNN 模型准备世界
────────────────────────────────────────────
.dlc
  │
  │ libQnnModelDlc.so 读取
  ▼
Compose QNN graph
  │
  ▼
QnnGraph_finalize() on HTP backend
  │
  ├────────────── Execute
  │
  └── serialize QNN context
              │
              ▼
         context .bin

产品运行世界
────────────────────────────────────────────
context .bin
  │
  │ contextCreateFromBinary
  ▼
QNN context / graph
  │
  │ graphExecute
  ▼
HTP execution
```

手机上还同时存在两套二进制：ARM64 侧和 Hexagon 侧。

```text
ARM64 Android side                    Hexagon V79 side
────────────────────────              ─────────────────────
qnn-net-run                           libQnnHtpV79Skel.so
qnn-context-binary-generator
qnn-context-runner
libQnnHtp.so
libQnnSystem.so
libQnnModelDlc.so
libQnnHtpPrepare.so
libQnnHtpV79Stub.so
```

扩展名都可能是 `.so`，但 CPU/Hexagon ELF 架构和职责完全不同。

---

## 2. 一张最实用的产物责任表

| 产物 | 典型生成者 | 典型读取者 | 主要在哪侧出现 | 当前项目含义 |
| --- | --- | --- | --- | --- |
| `.pth/.pt/.ckpt` | PyTorch 训练代码 | PyTorch model class | Host/开发环境 | 权重/checkpoint，不是 QNN graph |
| `.onnx` | PyTorch export | converter / ORT / 工具链 | Host/云端转换 | 图交换中间产物 |
| `.dlc` | Qualcomm conversion / AI Hub | `libQnnModelDlc.so` | Host 可保存；target 可 prepare | SoC-agnostic 模型表示，当前 RF-DETR 的起点 |
| context `.bin` | QNN context generator / AI Hub | QNN runtime | target runtime | HTP-specific、最好针对目标 SoC prepare |
| ARM64 `.so` | QAIRT/NDK | Android process linker | ARM CPU | backend/system/model loader/stub |
| Hexagon `.so` | QAIRT | Hexagon runtime | HTP/DSP side | Skel 等目标侧 runtime component |
| executable | QAIRT 或自己 NDK 编译 | Android kernel/shell | ARM CPU | 真正 `exec()` 运行的程序 |
| `.raw` | preprocess / runtime | QNN runner / postprocess | Host ↔ target 都可能 | 无 header 的 tensor payload |

这里最容易混的是：

```text
.dlc    不是 executable
.bin    不是 executable
.so     不等于“模型”
.raw    不知道 shape/dtype 就无法正确解释
```

---

## 3. `.pth / .pt / .ckpt`：权重不等于模型图

一个典型 PyTorch checkpoint 只保存：

```text
parameter tensors
optimizer state（可能有）
training metadata（可能有）
```

它通常还依赖 Python 代码重新构造网络：

```python
model = UniAD(...)
state = torch.load("model.pth")
model.load_state_dict(state)
```

QNN runtime 并不知道：

```text
`projects/mmdet3d_plugin/...` 里哪个 Python class
forward 怎么走
哪些 mmcv custom op 要调用
```

因此：

```text
.pth → QNN
```

中间必须先发生“把 Python 模型变成静态 tensor graph”这件事。

对 UniAD 来说，这就是后续最主要的工程工作。

---

## 4. `.onnx`：把 Python execution 变成 tensor graph

典型：

```text
PyTorch eager model
      ↓
固定 input contract + export patch
      ↓
ONNX graph
```

ONNX 的价值不是“QNN 必须先有一个 `.onnx` 文件”，而是它提供了一层很有用的检查面：

```text
Python 控制流是否已经消失？
shape 是否静态？
特殊 op 变成什么？
有没有 custom domain / ATen fallback？
输出 name/shape/dtype 是否正确？
```

Qualcomm AI Hub 的高级 API 可能把一些转换步骤封装起来，所以工程上不必强制“每条路线都永久保存 ONNX”。

但做 UniAD 调试时，ONNX/export graph 是非常好的问题定位边界。

---

## 5. `.dlc`：当前 RF-DETR 项目的真正起点

Qualcomm AI Hub 文档对 DLC 的重要定义是：它保存一个较 SoC-agnostic 的模型表示，用来驱动后续 QNN graph 构建和目标相关 prepare。

当前命令：

```bash
qnn-net-run \
  --backend /data/local/tmp/qnn_mobile/runtime/libQnnHtp.so \
  --model /data/local/tmp/qnn_mobile/runtime/libQnnModelDlc.so \
  --dlc_path /data/local/tmp/qnn_mobile/models/rf_detr/model.dlc \
  ...
```

三个参数不要混：

```text
--backend   = 谁执行 QNN graph
--model     = 谁读取/Compose 这个模型资产
--dlc_path  = 真正的 RF-DETR 模型数据
```

也就是说：

```text
libQnnModelDlc.so    软件 loader
model.dlc            模型资产
libQnnHtp.so         HTP backend
```

DLC 本身不能：

```bash
./model.dlc
```

它不是 ELF executable。

---

## 6. DLC 还没有完成目标 SoC prepare

当前 online prepare 路线真实发生：

```text
model.dlc
   ↓
libQnnModelDlc.so
   ↓
Compose
   ↓
QnnGraph_finalize()
   ↓
HTP executable state
   ↓
Execute
```

这也是为什么第一次用 DLC 启动会有很大的准备成本。

当前实测：

```text
Compose    ≈ 418 ms
Finalize   ≈ 9,670 ms
net-run INIT（包含 prepare）≈ 10,088 ms
```

所以一个 `.dlc` 文件“已经生成成功”不等于：

```text
已经针对 S25 Ultra 完成 HTP 编译/prepare
```

真正 HTP-specific 的 prepare 在后续 Finalize/context build 发生。

---

## 7. Context Binary `.bin`：把已 prepare 的 context 保存下来

项目命令：

```bash
qnn-context-binary-generator \
  --backend libQnnHtp.so \
  --model libQnnModelDlc.so \
  --dlc_path model.dlc \
  --binary_file rf_detr
```

概念链路：

```text
DLC
 ↓
Compose
 ↓
Finalize / HTP prepare
 ↓
getBinarySize / getBinary
 ↓
rf_detr.bin
```

当前实测 context binary：

```text
61,456,384 bytes
```

运行时自己的 C++ 不再需要：

```text
libQnnModelDlc.so
model.dlc
Compose
QnnGraph_finalize()
```

而是：

```text
rf_detr.bin
      ↓
contextCreateFromBinary()
      ↓
graphRetrieve()
      ↓
graphExecute()
```

---

## 8. 为什么 `.bin` 很像 TRT engine

非常有用的近似类比：

```text
ONNX / network          → TensorRT build → .engine
DLC / QNN graph         → HTP Finalize   → context .bin
```

所以：

```text
DLC                ~ build 前模型资产
Context Binary     ~ serialized prepared engine
```

但 Qualcomm AI Hub FAQ 明确强调：Context Binary 是 HTP-specific，并与目标 SoC 的 prepare 强相关；为了最佳性能应该针对正确 `soc_model` 构建。

因此不要因为它像 engine 就默认：

```text
任意 Snapdragon 都完全通用        ×
跨任意 QAIRT 版本都稳定           ×
.bin 是 ARM executable            ×
```

---

## 9. `.so`：同一个后缀里至少有三类完全不同的东西

### 9.1 ARM CPU 侧 QNN runtime/backend

当前项目：

```text
libQnnHtp.so
libQnnSystem.so
libQnnHtpPrepare.so
libQnnModelDlc.so
libQnnHtpV79Stub.so
```

这些由 Android ARM64 process 加载。

例如自己的 runner：

```cpp
dlopen(libQnnHtp.so)
dlopen(libQnnSystem.so)
```

### 9.2 Hexagon 目标侧 runtime

```text
libQnnHtpV79Skel.so
```

它不是给 ARM linker 当普通 Android `.so` 用的。

### 9.3 自定义 op package / plugin

QNN 生态也可以存在 custom op package，但当前 RF-DETR 路线没有一个：

```text
libMSDeformableAttention.so
```

因此后面 UniAD 遇到 MSDeformableAttention 时，第一反应不应该是：

```text
“我需要写一个 .so plugin”
```

Qualcomm BEVFormer/RF-DETR 给出的更现实路线是：**先把特殊 op 改写成 converter/backend 可接受的标准 tensor primitives。**

---

## 10. Executable：真正跑在 Android ARM CPU 上的程序

当前三类：

```text
qnn-net-run
qnn-context-binary-generator
qnn-context-runner
```

它们才是：

```bash
adb shell /data/local/tmp/.../qnn-net-run ...
adb shell /data/local/tmp/.../qnn-context-runner ...
```

能被操作系统 `exec()` 的 Android ARM64 ELF。

区别：

```text
qnn-net-run
    Qualcomm 写好的通用 runtime

qnn-context-binary-generator
    Qualcomm 写好的 build/serialize 工具

qnn-context-runner
    本项目自己 NDK 编译的 runtime
```

所以：

```text
context binary = 数据/模型执行资产
executable     = 驱动 runtime 的 CPU 程序
```

---

## 11. `.raw`：文件里只有 bytes，没有 tensor contract

RF-DETR 输入：

```text
image.raw
```

文件本身不知道：

```text
RGB 还是 BGR？
NCHW 还是 NHWC？
float32 还是 int32？
shape 是多少？
有没有 /255？
```

这些 contract 来自模型 recipe/metadata。

当前输入：

```text
RGB
NCHW
float32
[1,3,512,512]
range [0,1]
3,145,728 bytes
```

输出同理：

```text
classes.raw 1200 bytes
```

光看 1200 bytes 无法知道它是 `[1,300] int32` 还是 `[1,300] float32`。

---

## 12. Host 文件与 Target 文件不要混

当前开发机 Fedora x86_64：

```text
QAIRT_ROOT/
RF_DETR_DLC
Python preprocess
NDK compiler
```

目标手机 Android ARM64：

```text
/data/local/tmp/qnn_mobile/runtime/...
/data/local/tmp/qnn_mobile/models/rf_detr/model.dlc
/data/local/tmp/qnn_mobile/input/rf_detr/image.raw
/data/local/tmp/qnn_mobile/lifecycle/.../rf_detr.bin
```

相同名字的 `.so` 不代表可跨 host/target 架构使用。

例如：

```text
x86_64 host lib   ≠ Android aarch64 lib
Android aarch64   ≠ Hexagon V79 binary
```

这也是项目 `deploy-runtime.sh` 要做 ELF/DT_NEEDED 检查的原因。

---

## 13. 把 RF-DETR 和 UniAD 放到同一张产物地图

### RF-DETR 当前已经有

```text
Qualcomm AI Hub / recipe
        ↓
RF-DETR DLC             ← 这里进入本项目
        ↓
Compose / Finalize
        ↓
Context Binary
        ↓
C++ Execute
```

### UniAD 现在缺的主要是前半段

```text
OpenDriveLab UniAD source
        +
checkpoint.pth
        ↓
固定部署 contract
        ↓
替换/改写 mmcv/custom CUDA/dynamic logic
        ↓
export graph / ONNX
        ↓
QNN conversion
        ↓
UniAD DLC               ← 一旦到这里，开始复用现有 qnn_mobile 后半链路
        ↓
Compose / Finalize
        ↓
Context Binary
        ↓
C++ Execute
```

所以后面遇到“UniAD 为什么还跑不起来”，先判断问题发生在哪个产物边界：

```text
checkpoint load？
PyTorch eager？
export？
ONNX graph？
DLC conversion？
HTP Finalize？
Execute？
```

不要把所有问题都笼统叫“QNN 不支持”。

---

## 14. 一眼判断文件的口诀

```text
.pth   = 训练世界的权重/状态
.onnx  = 可交换的静态图中间层
.dlc   = Qualcomm 模型资产，仍要 backend prepare
.bin   = 已 prepare/序列化的 QNN context
.so    = 软件动态库，要看 CPU/Hexagon/插件职责
ELF executable = 真正由 Android CPU 启动的程序
.raw   = 没有 header 的 tensor payload
```

---

## 参考入口

- Qualcomm AI Hub FAQ: https://dev.aihub.qualcomm.com/docs/hub/faq.html
- 当前流程：`docs/lifecycle-guide.md`
- 当前实测：`report/lifecycle-results.md`
- 当前 runtime：`cpp/qnn_context_runner.cpp`
