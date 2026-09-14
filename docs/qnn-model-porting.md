# 从 PyTorch 开源模型到 QNN DLC：一条可执行的迁移路线

这篇文档回答的不是“QNN 支不支持 PyTorch”，而是：**拿到一个真实开源模型和 checkpoint 后，怎么把问题一步步缩小，最后得到一个能在当前 S25 Ultra 上 `Compose → Finalize → Execute` 的 DLC。**

当前项目已经把后半段跑通：

```text
DLC
→ libQnnModelDlc.so
→ Compose
→ QnnGraph_finalize()
→ Execute
→ Context Binary
→ 自写 C++ runner
```

UniAD 真正新增的是前半段：

```text
PyTorch source + checkpoint
→ 固定部署 contract
→ export-friendly PyTorch
→ export/ONNX
→ QNN conversion
→ DLC
```

所以不要把“RF-DETR DLC 已经能跑”误解成“已经会部署任意 PyTorch 模型”。

---

## 1. 两条入口先分清：现成 Qualcomm recipe vs 自己的模型

### 路线 A：Qualcomm AI Hub Models 已经有这个模型

例如 RF-DETR、BEVFormer。

这时最有价值的不是只下载 DLC，而是同时读：

```text
model.py
export.py
external_repos/*.diff / model_patch.py
input spec
release-assets.yaml
```

因为这些文件回答：

```text
上游模型哪里不能直接 export？
Qualcomm 改了哪些算子？
输入 shape/dtype/layout 固定成什么？
QNN_DLC 是否已经有 release/profile 证据？
```

RF-DETR 和 BEVFormer 就是我们后面处理 UniAD deformable attention 的参考来源。

### 路线 B：Qualcomm 仓库没有你的模型

UniAD 更接近这条。

这时要自己建立：

```text
original PyTorch
→ deployment wrapper / patch
→ exported model
→ QNN DLC
```

可以使用 Qualcomm AI Hub Workbench，也可以研究本地 QAIRT converter；当前 `qnn_mobile` 已验证的是 DLC 之后的本地 runtime，不应该假装已经验证了某一套 UniAD converter 命令。

---

## 2. 第一件事不是 export，而是固定部署 contract

在 PyTorch eager 模式先决定产品输入到底是什么。

对普通图像模型可能只有：

```text
image: [1,3,H,W]
```

UniAD 不是。至少要明确：

```text
camera image 数量
每路 image H/W
prev_bev 是否作为 graph input
can_bus / ego motion
lidar2img / calibration
track query / memory state
BEV H/W
num_query
feature levels
num_points
各 task head 要不要一次全部导出
```

如果这些东西运行时可以任意变化，export 和 QNN shape inference 会同时变难。

第一版部署应尽量固定：

```text
batch = 1
num_cams = 常量
input H/W = 常量
BEV H/W = 常量
num_query = 常量
num_levels = 常量
num_points = 常量
state tensor shape = 常量
```

这里的原则和 TensorRT static engine 一样：**先把动态业务问题变成静态 tensor contract，再谈 accelerator。**

---

## 3. 建一个 PyTorch 数值基线，后面所有改写都和它比

在任何 Qualcomm patch 之前先保存：

```text
固定 checkpoint hash
固定 sample input
固定 model.eval()
固定随机种子（如果模块涉及随机性）
固定所有输出 tensor
```

推荐每个风险模块都保留：

```text
input/*.npy
reference_output/*.npy
contract.json/yaml
```

比如 deformable attention：

```text
query
value
reference_points
spatial_shapes
level_start_index
mask
↓
reference output
```

后面改成 grid_sample、改 layout、Linear→Conv，都先做：

```text
Original PyTorch vs Patched PyTorch
```

这一步不过，根本不要进入 QNN。

---

## 4. 第二步才是静态审计：什么会阻止 export/QNN

不要只搜“unsupported op”。先搜 Python 和动态图行为：

```text
Tensor.item()
.tolist()
.numpy()
.cpu()
nonzero()
where 后产生变长 tensor
Python if 依赖 tensor value
Python for 次数依赖 tensor value
动态 max_len
动态 top-k K
动态 reshape/slice
custom CUDA/C++ extension
mmcv.ops
ext_loader.load_ext
自定义 torch.autograd.Function
```

为什么？

因为失败经常不是：

```text
QNN 不会 MatMul
```

而是：

```text
这个 MatMul 的输入长度来自 nonzero 运行时结果
```

后者才真正破坏静态编译。

---

## 5. 先按模块画风险地图，不要整模型盲 export

UniAD 第一轮建议：

| 模块 | 第一判断 | 原因 |
| --- | --- | --- |
| ResNet/FPN 普通部分 | 低~中 | Conv/BN/ReLU/FPN 相对常规 |
| DCNv2 | 高 | custom/deformable conv 路径需单独确认 |
| FFN/Linear/LayerNorm | 低~中 | 普通算子，但 layout/rank 会影响性能 |
| MultiheadAttention | 中 | 可拆为 projection + matmul + softmax |
| TemporalSelfAttention | 中~高 | deformable sampling + prev_bev，但 UniAD 是 single-level |
| MSDeformableAttention3D / SCA | 高 | 4-level sampling + camera mask/rebatch/scatter |
| Detection decoder deformable attention | 中~高 | UniAD 为 single-level，有 Qualcomm 近似实现 |
| Tracking state/query | 高 | 跨帧 state contract 要静态化 |
| Planning collision optimizer | 不进首版 DLC | `nonzero/CPU/Numpy/nonlinear solver` 更适合 CPU 后处理 |

这样一旦失败，知道是哪个边界坏，不是“UniAD 导出失败”。

---

## 6. 自定义 CUDA op 的正确处理顺序

看到：

```text
mmcv.ops.MultiScaleDeformableAttention
DCNv2 CUDA op
ScatterND custom Function
```

不要立刻写 QNN custom op。

优先级：

```text
1. 找数学等价标准 PyTorch primitives
2. 找 Qualcomm AI Hub Models 的同类已部署 patch
3. 固定 shape/layout，把动态逻辑移掉
4. 拆成 standalone module 验证
5. 如果仍无法表达，再研究 QNN custom op/package
```

Qualcomm BEVFormer 的 deformable attention 就是典型案例：把原来依赖 MMCV compiled extension 的核心计算，改写成 projection、reshape/permute、`grid_sample`、weighted sum 等普通 tensor graph。

---

## 7. 一个风险算子的 standalone 实验应该长什么样

例如先验证 `MSDeformableAttention3D`，不是整个 BEV encoder。

包装成：

```python
class DeployableMSDA(nn.Module):
    def forward(
        self,
        query,
        value,
        reference_points,
        spatial_shapes,
        mask,
    ):
        ...
```

固定 contract，例如：

```text
batch = 1
embed_dims = 256
num_heads = 8
num_levels = 4
num_points = 8
BEV H/W = 200/200
camera count = 6
feature H/W = 固定常量
```

然后按顺序验：

```text
Original eager
    ↓ compare
Patched eager
    ↓ export
Export runtime
    ↓ compile
QNN DLC
    ↓ device
HTP Finalize / Execute
```

每过一层再进入下一层。

---

## 8. PyTorch 改写时真正要保持的是数学，不是原 class 名

### 例 1：Linear → 1x1 Conv

对于规则空间 feature：

```text
[B,H,W,C]
```

每个位置执行同一个：

```text
Linear(Cin → Cout)
```

可以在正确重排权重/layout 后表达成：

```text
[B,C,H,W]
→ Conv2d(Cin,Cout,k=1)
```

Qualcomm BEVFormer 的 `OptimizedLinear` 就是这个思路。

### 例 2：MMCV MSDeformAttn extension → primitives

不是保留：

```text
MultiScaleDeformableAttnFunction.apply(...)
```

而是展开：

```text
sampling_offsets = projection(query)
attention_weights = projection(query) + softmax
sampling_locations = reference_points + normalized offsets
per-level grid_sample(value)
weighted sum
output projection
```

对于 QNN 来说，后者更像一个可分析的 tensor graph。

---

## 9. Export 层到底要验什么

一个模块 `torch.export` 或 ONNX 成功，只能说明 Python 图被捕获了。

还要检查：

```text
有没有自定义 domain op？
有没有 ATen fallback？
Shape/Gather 是否在驱动动态 reshape？
nonzero 输出是不是变长？
Expand/Tile 有没有把 tensor 爆炸放大？
某个常量 H/W 有没有变成 runtime tensor？
output name/shape/dtype 是否和 contract 一致？
```

RF-DETR 的上游 export 改造甚至专门把 `(H,W)` 以 Python int 传入 deformable attention core，避免从 tensor data 中提取 shape 导致 data-dependent tracing。

这就是“能 export”和“适合静态部署”的区别。

---

## 10. 生成 DLC 有两种现实工作流

### 10.1 Qualcomm AI Hub Workbench

当前官方文档支持直接把 exported PyTorch/ONNX 编译成 QNN DLC。

概念示例：

```python
import qai_hub as hub

compile_job = hub.Client().submit_compile_job(
    model=exported_model,
    device=hub.Device("目标设备"),
    input_specs={"image": (1, 3, H, W)},
    options="--target_runtime qnn_dlc",
)

dlc = compile_job.get_target_model()
```

对 UniAD 实际会有多个 input specs，而不是一个 image。

这条路线的价值：

```text
快速确认 converter 能不能接受图
可直接 profile 云端 Qualcomm device
能生成 DLC 与 QNN Context Binary
```

### 10.2 本地 QAIRT converter/compiler

适合需要完全本地、深入 converter 配置或最终产品工具链时研究。

但当前 `qnn_mobile` 还没有验证一套“UniAD PyTorch → 本地 QAIRT → DLC”的固定脚本，因此文档不应该凭空给一个看起来很完整但没跑过的命令。

原则是：

> **先把 source graph 和 input contract 搞对；Workbench 或本地 converter 只是下一层工具入口。**

---

## 11. 为什么第一版目标建议是 float QNN DLC，而不是直接量化 Context Binary

推荐阶段：

```text
Original PyTorch float
→ Patched PyTorch float
→ Export float
→ QNN float DLC
→ 手机 HTP Finalize/Execute
→ 数值对齐
→ Context Binary
→ 再研究 FP16 / W8A16 / W8A8
```

原因是把问题隔离：

```text
算子改写错误
动态 shape/export 错误
converter/QNN 错误
量化误差
```

如果第一次就混在一起，输出不对时很难定位。

Qualcomm AI Hub 官方量化流程同样把 optimized/exported ONNX 与 quantization 分成独立步骤。

---

## 12. DLC 一出来，就立即回到当前项目已经熟悉的链路

假设终于得到：

```text
uniad_encoder.dlc
```

不要马上继续拼更多模块。

先复用当前实验方法：

```text
1. deploy DLC + raw inputs
2. qnn-net-run + libQnnModelDlc.so + libQnnHtp.so
3. 确认 Compose
4. 确认 Finalize
5. Execute 多次
6. 拉回 native outputs
7. 和 Patched PyTorch 对齐
8. 生成 Context Binary
9. 再用 C++ runner 恢复
```

这就是 RF-DETR 项目真正的价值：

> **它把 DLC 之后的部署和验证方法已经做成实验底座。**

---

## 13. 转换失败时，不要把错误都归为“算子不支持”

### A. Export 失败

典型：

```text
data-dependent Python branch
Tensor.item()/numpy()
custom extension 无 symbolic/decomposition
```

处理：改 PyTorch 表达。

### B. Converter unsupported op

处理：

```text
标准 primitive decomposition
Qualcomm 同类模型 patch
必要时 custom op
```

### C. Shape inference 失败

处理：

```text
固定 H/W/K/num_levels
移除 data-dependent reshape
Python/static constant 替代 tensor-derived shape
```

### D. HTP Finalize 失败

说明：

```text
DLC 已生成
≠
该图一定能在目标 HTP 成功 prepare
```

此时保留 converter log + device Finalize log，把图缩成最小失败单元。

### E. Execute 成功但结果错

优先查：

```text
preprocess
layout
reference point convention
align_corners
padding
native dtype
weight mapping
```

不要第一反应认为 HTP 算错。

---

## 14. UniAD 推荐的实际迁移顺序

不是按源码目录顺序，而是按“已有 Qualcomm 参考 + 风险最小化”排序：

```text
0. 固定 stage2 部署 contract / sample / outputs

1. 普通 Conv/FPN/MLP/Norm 小块

2. TemporalSelfAttention standalone
   UniAD: num_levels=1
   Qualcomm BEVFormer: 有 TSA optimized reference

3. Detection decoder deformable attention standalone
   UniAD: num_levels=1
   Qualcomm: 有 CustomMSDeformableAttention_Decoder_Optimized

4. 4-level MSDeformableAttention3D core
   Qualcomm BEVFormer 单层实现不能直接照搬
   结合 RF-DETR multi-level core 参考扩展

5. SpatialCrossAttention
   camera mask / rebatch / scatter / fixed max length

6. 一层 BEVFormerEncoder

7. 完整 BEV encoder + prev_bev state

8. Backbone/FPN 中尚未验证的 DCNv2 风险

9. detection/tracking/map/motion heads 分块

10. planning neural head

11. collision/nonlinear optimizer 保持 CPU 后处理
```

这条顺序的核心不是“先简单后复杂”这么泛，而是：**先吃掉 Qualcomm 已经给过部署证据的结构，再处理真正缺口。**

---

## 15. 每个阶段必须产出什么，才算真正前进

一个模块卡片至少：

```text
源码 commit / checkpoint hash
Original class
Deploy class
input contract
output contract
Original vs Deploy PyTorch error
export result
export graph inspection
QNN compile result
DLC hash
HTP Finalize result
QNN vs PyTorch error
profiling
known limitation
```

状态只允许：

```text
EAGER_PASS
PATCHED_NUMERIC_PASS
EXPORT_PASS
QNN_COMPILE_PASS
HTP_FINALIZE_PASS
HTP_EXECUTE_PASS
NUMERIC_PASS
PROFILED
```

不要用一个模糊的：

```text
“这个算子 QNN 支持”
```

代替全部阶段。

---

## 16. 最后把整个迁移压成一条故事线

```text
PyTorch source/checkpoint
        │
        │ 固定部署输入/state
        ▼
Original eager baseline
        │
        │ 找 dynamic/custom op
        ▼
Deploy-friendly PyTorch
        │
        │ Original vs Deploy 数值对齐
        ▼
Export graph / ONNX
        │
        │ 检查静态 shape / primitives
        ▼
QNN DLC
        │
        │ 当前 qnn_mobile 接管
        ▼
Compose → Finalize → Execute
        │
        ▼
Context Binary
        │
        ▼
自写 Android C++ runtime
```

对 RF-DETR，我们从 `QNN DLC` 这格开始。

对 UniAD，我们现在正站在：

```text
Original eager baseline
→ Deploy-friendly PyTorch
```

之间。

这就是下一阶段工作的准确位置。

---

## 参考入口

- Qualcomm AI Hub compile examples: https://dev.aihub.qualcomm.com/docs/hub/compile_examples.html
- Qualcomm AI Hub quantization examples: https://dev.aihub.qualcomm.com/docs/hub/quantize_examples.html
- Qualcomm AI Hub Models: https://github.com/qualcomm/ai-hub-models
- 本项目 DLC 后半链路：`docs/lifecycle-guide.md`
- UniAD 算子映射：`docs/uniad-qnn-operator-reference.md`
