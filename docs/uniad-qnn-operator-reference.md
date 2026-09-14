# UniAD → QNN DLC：逐模块对照 Qualcomm 已有实现

这篇文档只做一件事：**把 UniAD stage2_e2e 的高风险模块逐个对到 Qualcomm `ai-hub-models` 里已经存在的源码实现，并明确“能直接参考什么、还缺什么、下一步应该写哪个 standalone 实验”。**

不是“QNN 支持表”，也不宣称 UniAD 已经可以直接导出 DLC。

当前最重要的结论已经从“BEVFormer 有点像 UniAD”推进到更具体的四条：

```text
1. UniAD TSA (num_levels=1)
   → Qualcomm BEVFormer TSA optimized 是最直接参考。

2. UniAD SCA core (num_levels=4)
   → Qualcomm BEVFormer 负责 SCA/3D-reference/layout 参考；
   → Qualcomm Mask2Former/RF-DETR 负责真正 multi-level sampling core 参考。

3. UniAD backbone DCNv2
   → Qualcomm CenterNet template 已经有 custom_deformconv2d decomposition，
      不再是“完全没有 Qualcomm 参考”。

4. Planning collision optimizer
   → 明确留 CPU，不把 nonzero/Numpy/nonlinear solver 强塞进第一版 DLC。
```

---

# 1. UniAD stage2 先固定这些结构事实

当前 stage2_e2e 需要关注：

```text
embed_dims = 256
BEV = 200 x 200
num_query = 900
feature levels = 4
```

BEV encoder：

```text
TemporalSelfAttention
    num_levels = 1
    num_heads = 8
    num_points = 4
    num_bev_queue = 2

SpatialCrossAttention
    ↓
MSDeformableAttention3D
    num_levels = 4
    num_heads = 8
    num_points = 8
```

Detection decoder：

```text
MultiheadAttention
+
CustomMSDeformableAttention
    num_levels = 1
```

其它：

```text
Panseg / segmentation:
    4-level deformable attention

Motion:
    MotionDeformableAttention
    num_levels = 1
    num_heads = 8
    num_points = 4
    num_steps = 12

Backbone:
    ResNet101
    stage 3/4 使用 DCNv2
```

因此“UniAD 的 deformable attention”不是一个问题，而至少是：

```text
TSA single-level
SCA 4-level
Detection decoder single-level
Segmentation 4-level
Motion single-level + temporal steps
```

---

# 2. 先给最终参考表

| UniAD 模块 | Qualcomm 源码参考 | 参考强度 | 还缺什么 |
| --- | --- | --- | --- |
| DCNv2 | `templates/centernet/model_patches.py::custom_deformconv2d` | **强** | 对齐 UniAD/mmcv DCNv2 参数与 batch/groups |
| GridSample | BEVFormer / Mask2Former / RF-DETR / CREStereo / SimpleBEV | **很强** | 固定 coordinate/mode/align_corners 语义 |
| TSA | BEVFormer `MSDeformableAttention_TSA_Optimized` | **很强** | UniAD 权重映射 + 200×200 性能/内存 |
| SCA 上层 | BEVFormer `SpatialCrossAttention` Qualcomm patch | **很强** | UniAD camera rebatch/nonzero 静态化 |
| SCA 3D MSDA | BEVFormer `MSDeformableAttention3D_SCA_Optimized` | **接口强 / core 不完整** | Qualcomm helper 只按 single-level 优化 |
| 4-level MSDA core | Mask2Former `multi_scale_deformable_attention` + RF-DETR core | **很强** | 与 UniAD 3D reference layout 拼接 |
| Decoder MSDA | BEVFormer `CustomMSDeformableAttention_Decoder_Optimized` | **很强** | 权重/shape 对齐 |
| MHA | BEVFormer `MultiheadAttention_Optimized` | **强** | 是否值得 split-head / layout 改写要 profile |
| Linear/FFN layout | BEVFormer `OptimizedLinear` | **强** | 只针对规则 spatial tensor 试，不全局替换 |
| ScatterND | BEVFormer `custom_utils.ScatterND`、BEVDet | **中~强** | 动态 index / duplicate semantics |
| NonZero/变长 query | BEVFormer static padding/mask；StateTransformer dynamic-control rewrite | **思想强** | UniAD 需要自己定 fixed max contract |
| BEV rotate | BEVFormer `custom_rotate` / GridSample | **强** | center/angle/layout 对齐 |
| Planning nonlinear optimizer | 无需 QNN 参考 | **明确图外** | CPU 后处理接口 |

---

# 3. Backbone DCNv2：现在有一个具体 Qualcomm 解法可抄

上一版只说：

```text
UniAD 有 DCNv2
Qualcomm BEVFormer 改成普通 ResNet50
所以 DCNv2 高风险
```

这不够。

Qualcomm `ai-hub-models` 里还有更直接的参考：

```text
src/qai_hub_models/models/templates/centernet/model_patches.py
```

里面明确提供：

```text
calculate_p0()
calculate_pk()
bilinear_sample()
custom_deformconv2d()
custom_dcn_forward()
```

## 3.1 Qualcomm 实际把 DCNv2 展成什么

原本：

```text
DCNv2 compiled/custom operator
```

改成：

```text
input x
  │
  ├── conv_offset_mask(x)
  │      ├── learned offset y
  │      ├── learned offset x
  │      └── sigmoid modulation mask
  │
  ├── regular conv base grid p0
  ├── kernel relative grid pk
  │
  └── p = p0 + pk + learned_offset
              │
              ▼
        bilinear_sample(x, p)
              │
              ▼
        * modulation mask
              │
              ▼
        reshape sampled patch
              │
              ▼
          F.conv2d
```

这就是 DCNv2 的 deploy-friendly decomposition。

## 3.2 对 UniAD 不能直接 copy 后就结束

必须逐项核对 UniAD 使用的 DCN：

```text
kernel_size
stride
padding
dilation
groups
deformable_groups
modulated=True/False
offset/mask channel ordering
batch size
weight layout
boundary behavior
```

Qualcomm CenterNet 实现里存在针对其模型 contract 的假设，所以第一件事不是直接替换 UniAD ResNet，而是：

```text
Experiment DCN-01

Original UniAD/mmcv DCNv2
        vs
Qualcomm-style custom_deformconv2d
```

固定一层真实 UniAD weight + feature input，先做 PyTorch 数值对齐。

## 3.3 证据等级

Qualcomm 当前 CenterNet Pose recipe 有 QNN Context Binary 路线，这说明这种 DCN decomposition 是 Qualcomm 实际用于 NPU/export 的代码路径。

但这不等于：

```text
UniAD 的所有 DCNv2 参数组合
→ QNN_DLC 原样 PASS
```

所以结论应写：

> **DCNv2 已找到 Qualcomm 官方 decomposition 参考；UniAD 仍需 standalone 验证。**

---

# 4. TemporalSelfAttention：最应该第一个真正导 DLC 的 UniAD attention

UniAD TSA：

```text
embed_dims = 256
num_heads = 8
num_levels = 1
num_points = 4
num_bev_queue = 2
```

Qualcomm BEVFormer patch：

```text
MSDeformableAttention_TSA_Optimized
```

参数结构几乎就是这个场景。

## 4.1 Qualcomm 改写后的核心路径

```text
current query + history/prev_bev
        │
        ├── sampling_offsets projection
        └── attention_weights projection + softmax

reference_points
        +
normalized offsets
        ↓
sampling_locations
        ↓
F.grid_sample(value)
        ↓
* attention_weights
        ↓
weighted sum
```

另外 Qualcomm 会控制：

```text
NCHW/NHWC layout
OptimizedLinear = Conv2d(k=1)
split-head / reshape pattern
```

## 4.2 我们要做的不是重新写 TSA，而是先验证“权重能否映射”

第一版 standalone contract 建议直接固定：

```text
query/current_bev     [1,200,200,256] 或 deployment layout
prev_bev              fixed shape
reference_points      fixed shape
spatial_shapes        [(200,200)] / static constants
num_levels            1
num_points            4
```

验收：

```text
Original UniAD TSA
vs
Qualcomm-style TSA
```

对齐：

```text
sampling_offsets.weight/bias
attention_weights.weight/bias
value layout
reference coordinate convention
head merge order
```

然后立即：

```text
export → QNN float DLC → S25 HTP
```

而不是先拼整个 encoder。

## 4.3 性能为什么必须单独测

UniAD BEV：

```text
200 × 200 = 40,000 queries
```

一个 `[1,200,200,256] float32` tensor payload 约 39 MiB。

所以：

```text
“Qualcomm 小 BEV 模型能表达 TSA”
```

不代表：

```text
“UniAD 40k queries 的 latency/memory 一定合理”
```

TSA 必须单独 profile。

---

# 5. SpatialCrossAttention：真正难点有两层，不要混在一起

UniAD SCA：

```text
Layer A: camera-visible query rebatch / mask / scatter
Layer B: MSDeformableAttention3D sampling core
```

如果一次全改，出了错完全不知道是：

```text
nonzero 动态 shape？
rebatch index？
reference point？
grid_sample？
attention weight？
scatter？
```

所以必须拆开。

---

# 6. SCA Layer A：camera rebatch / mask / ScatterND

原始 BEVFormer/UniAD 类逻辑可以抽象成：

```text
bev_mask per camera
        ↓
找 visible queries
        ↓
每个 camera query 数不同
        ↓
rebatch 成 camera-specific query list
        ↓
attention
        ↓
scatter/add 回完整 BEV slots
        ↓
按可见 camera 数 normalize
```

最危险的是：

```text
nonzero
variable length
max(dynamic lengths)
Python loop
scatter dynamic indices
```

## 6.1 Qualcomm BEVFormer patch 真正值得抄的是“静态化思想”

重点变量：

```text
padded_query_len(...)
sca_use_topk_query_pruning
sca_masking_before_gridsample
ScatterND
```

目标不是保留原来的 variable list，而是趋向：

```text
fixed max_len
+
padding
+
validity mask
+
fixed canvas
```

也就是：

```text
动态 shape
→ 静态 shape + data mask
```

## 6.2 UniAD 第一版建议明确 contract

例如先选择一个固定：

```text
MAX_VISIBLE_QUERIES_PER_CAMERA = K
num_cams = 6
```

每个 camera 永远输出：

```text
[K, C]
```

无效位置靠：

```text
valid mask
```

处理，而不是 tensor length 变化。

K 具体怎么定，要从真实 nuScenes sample 统计，不应该随便拍脑袋。

---

# 7. SCA Layer B：UniAD 是 4-level，Qualcomm BEVFormer helper 不是

UniAD：

```text
MSDeformableAttention3D
num_levels = 4
num_points = 8
```

Qualcomm BEVFormer：

```text
MSDeformableAttention3D_SCA_Optimized
```

非常有价值，因为它已经解决：

```text
3D/2D projected reference points
sampling offsets
attention weights
GridSample coordinate
HTP-friendly layout
```

但是其底层公开 helper：

```text
custom_multi_scale_deformable_attn_pytorch_single_grid()
```

实际按：

```text
value_spatial_shapes[0]
```

处理 single feature level。

所以这份代码**不能直接作为 UniAD 4-level core 的最终实现**。

---

# 8. 4-level core 应该直接看 Qualcomm Mask2Former

这轮补充后，这个参考比之前更明确。

文件：

```text
src/qai_hub_models/models/mask2former/model_patches.py
```

核心函数：

```text
multi_scale_deformable_attention()
```

Qualcomm 的结构：

```text
value
   ↓
split([H0*W0, H1*W1, H2*W2, H3*W3])
   ↓
level 0 reshape → grid_sample ─┐
level 1 reshape → grid_sample ─┤
level 2 reshape → grid_sample ─┤→ concat
level 3 reshape → grid_sample ─┘
                                │
attention_weights ──────────────┤
                                ↓
                            weighted sum
```

这是标准 multi-level deformable attention 最干净的 Qualcomm NPU-oriented 参考之一。

同时还可以看 RF-DETR：

```text
MSDeformAttn
ms_deform_attn_core_pytorch
```

RF-DETR 又提供了：

```text
multi-level + export shape handling
```

的另一份证据。

---

# 9. UniAD SCA 建议真正写的新 core

不要直接叫：

```text
copy_of_qualcomm_bevformer.py
```

更合理的是做自己的最小部署类：

```text
MSDeformableAttention3D4LevelQnn
```

职责明确：

```text
Input:
query
value_l0/l1/l2/l3 或 flattened value
reference_points
mask

Static config:
H0,W0
H1,W1
H2,W2
H3,W3
num_heads=8
num_points=8

Graph:
sampling_offsets
attention_weights
for four fixed levels:
    build grid_l
    grid_sample(value_l)
concat
weighted sum
```

注意这里“for four levels”在部署代码里应尽量静态展开，不让运行时 tensor 决定 loop 次数。

实现来源：

```text
BEVFormer SCA class
+
Mask2Former multi-level core
+
RF-DETR export shape handling
```

这才是 Qualcomm 代码真正对 UniAD 的组合利用方式。

---

# 10. Detection Decoder：第二个优先 standalone 的模块

UniAD decoder：

```text
MultiheadAttention
+
CustomMSDeformableAttention(num_levels=1)
```

Qualcomm BEVFormer：

```text
MultiheadAttention_Optimized
+
CustomMSDeformableAttention_Decoder_Optimized
```

这是目前最接近 source-level 替换的模块之一。

第一阶段要检查：

```text
query layout
reference_points layout
sampling_offsets weight shape
attention_weights weight shape
value_proj/output_proj weight shape
batch_first
residual location
```

如果 parameter shape 一致，优先写一个 weight mapping test：

```text
load same weights
Original decoder attention
vs
Optimized decoder attention
```

数值过了再导 QNN。

---

# 11. Panseg / segmentation：也有 4-level，不要复用 single-level helper

Segmentation head 里的 multi-scale attention 应直接进入：

```text
Mask2Former/RF-DETR multi-level reference
```

而不是：

```text
BEVFormer single-level helper
```

需要独立检查：

```text
reference_points 2D/4D
n_levels
n_points
value flatten order
level_start_index
output projection
```

如果和 Mask2Former 的 pixel decoder MSDA 足够接近，这可能比 SCA 更容易先跑通，因为没有 camera rebatch 那层动态逻辑。

---

# 12. MotionDeformableAttention：sampling core 可复用，motion semantics 不能抄

UniAD Motion：

```text
num_levels = 1
num_heads = 8
num_points = 4
num_steps = 12
```

可以复用的底层：

```text
single-level feature sample
reference + offset normalization
GridSample
attention weighted sum
```

不能直接复制的：

```text
num_steps 维度的 offset/weight layout
reference_trajs
bbox / trajectory semantic transform
future-step fusion
```

所以 motion 应在 TSA/SCA core 跑通后做，而不是第一批。

---

# 13. Linear / MHA：重点是 layout，不是“Linear 不支持”

Qualcomm BEVFormer：

```text
OptimizedLinear
```

实际：

```text
Linear on spatial locations
→ Conv2d(kernel=1)
```

价值在于：

```text
保持 NCHW/NHWC 规则 4D tensor
减少 sequence ↔ spatial layout 反复变换
更容易让 backend 使用成熟 Conv path
```

UniAD 建议先只在：

```text
TSA/SCA offset projection
attention weight projection
规则 BEV spatial FFN
```

试。

不要全局机械替换所有 Linear。

---

# 14. BEV rotate / warp：Qualcomm BEVFormer 可以直接当表达参考

原始业务：

```text
prev_bev
+
ego rotation / shift
→ aligned prev_bev
```

Deploy graph 可以：

```text
construct affine/grid
→ F.grid_sample(prev_bev, grid)
```

必须对齐：

```text
center
角度正负
radian/degree
nearest/bilinear
align_corners
coordinate convention
```

这一部分不需要自己从零发明。

---

# 15. ScatterND：Qualcomm 有代码，但 dynamic index 仍然是风险

参考：

```text
BEVFormer custom_utils.ScatterND
BEVDet model_patches.py
```

BEVDet 还明确处理：

```text
ONNX ScatterND/GatherND index 用 int64
```

所以：

```text
ScatterND 本身
```

并不是“没有任何 Qualcomm 先例”。

真正要审的是：

```text
indices.shape 是否静态
indices 内容是否 data-dependent
重复 index 是 assign 还是 add
canvas size 是否 compile-time fixed
```

SCA 的问题仍然主要在 variable visible query 数。

---

# 16. NonZero：不要问 QNN 有没有这个 op，先问它有没有改变图形状

危险逻辑：

```python
idx = mask.nonzero()
x = x[idx]
max_len = idx.shape[0]
y = torch.zeros(max_len, ...)
```

这是：

```text
data
→ control/shape
```

第一版应该改成：

```text
fixed capacity
+ mask
```

Qualcomm 其它模型（例如 StateTransformer）也会专门改掉“nonzero 结果决定后续执行哪些分支/专家”的 data-dependent control flow。

这个经验和 SCA/track queries 完全同类。

---

# 17. Tracking / Memory Bank：这是模型接口问题，不只是算子问题

UniAD 跨帧有：

```text
track query
memory bank
prev_bev
score/valid state
```

即使每个单独 operator 都能 export，整模型仍然可能因为 state 数量动态而无法静态部署。

第一版应该设计：

```text
fixed N track slots
feature tensor
reference tensor
valid mask
memory tensor
```

CPU/runtime 决定：

```text
哪些 slot 有效
跨帧怎么更新 state
```

而不是让 QNN graph I/O rank/length 每帧变化。

这部分要作为“deployment interface design”单独立项。

---

# 18. Planning：把神经网络和 nonlinear optimizer 分开

Planning neural graph 可以继续研究：

```text
Embedding
Linear
LayerNorm
Transformer decoder
Conv/MLP
cumsum
```

但 collision optimization 里：

```text
nonzero occupancy
动态筛选
.cpu()
.numpy()
CollisionNonlinearOptimizer
numpy → torch
```

第一版明确：

```text
QNN DLC
    ↓
raw planning trajectory + occupancy
    ↓
ARM CPU
    ↓
collision nonlinear optimizer
    ↓
final trajectory
```

这不是妥协，而是正确的软件边界。

---

# 19. 现在应该从 Qualcomm 仓库拉哪些文件

建议建立一个只读 reference 目录或记录 commit，不直接把整个仓库抄进项目。

## Attention / BEV

```text
src/qai_hub_models/models/bevformer/
  model.py
  external_repos/bevformertiny_minimal.diff
```

重点恢复：

```text
deformable_attention.py
MultiheadAttention.py
custom_utils.py
SpatialCrossAttention patch
TemporalSelfAttention patch
```

本项目已有抽取工具：

```bash
python scripts/extract-qualcomm-bevformer-patch-files.py ...
```

## 4-level MSDA

```text
src/qai_hub_models/models/mask2former/model_patches.py
src/qai_hub_models/models/rf_detr/model.py
```

外加 RF-DETR 上游：

```text
rfdetr/models/ops/modules/ms_deform_attn.py
rfdetr/models/ops/functions/ms_deform_attn_func.py
```

## DCNv2

```text
src/qai_hub_models/models/templates/centernet/model_patches.py
```

这是下一阶段应该直接读代码的文件，不再只记一个“DCN 高风险”标签。

## Scatter / BEV geometry

```text
src/qai_hub_models/models/bevdet/model_patches.py
src/qai_hub_models/models/simple_bev_cam/
```

---

# 20. 实际实验顺序现在改成这样

## Experiment 0：GridSample primitive

```text
small static feature
+ fixed grid
→ GridSample
→ QNN DLC / HTP
```

先确认目标 SDK/device 的最基本 sampling contract。

## Experiment 1：DCNv2 decomposition

```text
真实 UniAD DCNv2 layer + weight
Original mmcv
vs
Qualcomm-style custom_deformconv2d
```

## Experiment 2：TSA single-level

```text
Original UniAD TSA
vs
Qualcomm-style TSA optimized
→ QNN DLC
```

## Experiment 3：Decoder single-level MSDA

```text
Original decoder attention
vs
Qualcomm optimized decoder attention
```

## Experiment 4：4-level MSDA core

```text
Mask2Former/RF-DETR style per-level GridSample
```

先不加 camera rebatch。

## Experiment 5：SCA static rebatch/scatter

```text
fixed K per camera
+ mask
+ 4-level core
+ ScatterND
```

## Experiment 6：一层 BEV encoder

```text
TSA
→ Norm
→ SCA
→ Norm
→ FFN
```

## Experiment 7：6-layer BEV encoder + prev_bev

最后才扩大。

这比直接 export stage2_e2e 更容易产生真正可复用结论。

---

# 21. 每个实验必须输出一张算子证据卡

示例：

```text
Module:
UniAD DCNv2 stage3 blockX

Original:
mmcv ModulatedDeformConv2d

Deploy implementation:
CenterNet-style custom_deformconv2d

Static contract:
input      [1,C,H,W] fp32
kernel     3x3
offset     ...
mask       ...

Numeric:
Original vs Patched max_abs = ...

Export:
PASS / FAIL

QNN DLC:
PASS / FAIL

HTP Finalize:
PASS / FAIL

HTP Execute:
PASS / FAIL

Latency:
API wall = ...
accelerator = ...

Limitation:
...
```

最后形成的是：

```text
UniAD-on-QNN operator notebook
```

而不是一份没有实验证据的“支持列表”。

---

# 22. 当前最值得马上做的两件代码工作

如果接下来开始写代码，而不是继续看文档，我建议：

```text
Task A
从 Qualcomm CenterNet `custom_deformconv2d` 做一个 UniAD DCNv2 adapter
→ 用真实 UniAD DCN weight 做 eager 数值对齐

Task B
写 `MSDeformableAttention3D4LevelQnn`
→ SCA interface 参考 BEVFormer
→ multi-level core 参考 Mask2Former/RF-DETR
→ 暂时不加 camera dynamic rebatch
```

这两个都是明确的工程实验，不再是“研究一下算子支持”。

---

## 参考源码

Qualcomm AI Hub Models：

```text
src/qai_hub_models/models/bevformer/external_repos/bevformertiny_minimal.diff
src/qai_hub_models/models/mask2former/model_patches.py
src/qai_hub_models/models/templates/centernet/model_patches.py
src/qai_hub_models/models/bevdet/model_patches.py
src/qai_hub_models/models/simple_bev_cam/
src/qai_hub_models/models/rf_detr/model.py
```

本仓库：

```text
docs/qnn-operator-support.md
docs/qnn-model-porting.md
scripts/extract-qualcomm-bevformer-patch-files.py
```
