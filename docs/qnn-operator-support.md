# QNN 特殊算子迁移：不要问“支持吗”，要问“最后 lower 成什么 graph”

这篇文档专门服务 UniAD。目标不是列一张 QNN operator support 表，而是回答：**遇到 DCNv2、GridSample、MultiScaleDeformableAttention、ScatterND、NonZero、TopK、动态 shape 时，Qualcomm 自己在 `ai-hub-models` 里是怎么改写的，我们可以抄哪种“表达方式”，哪些只是参考而不是已经证明 UniAD 可直接部署。**

先给原则：

> **原模型里的 custom CUDA op 名字不是部署边界。部署边界是：你最终能不能把数学过程改写成 export/QNN/HTP 能接受的静态 tensor graph。**

---

## 1. “这个算子支持 QNN”至少分 6 层

以后不要一句话说“支持/不支持”。

```text
1. Original PyTorch eager 能跑
2. Patched PyTorch 数值等价
3. torch.export / ONNX 能表达
4. QNN conversion/compile 成功
5. 目标 HTP Finalize 成功
6. Execute + 数值 + 性能合理
```

Qualcomm 仓库里的代码可以提供第 2~6 层不同程度的证据，但要看具体模型 runtime。

例如：

```text
“某个 patch 里有 custom_deformconv2d”
```

只能证明 Qualcomm 使用过这种改写思路；如果该 model recipe 只发布 QNN Context Binary，那么它不是“UniAD DCNv2 已验证 QNN_DLC”的直接证据。

---

## 2. 先给 UniAD 算子参考总表

| UniAD 风险点 | Qualcomm 最值得看的参考 | 核心改写思想 | 对 UniAD 的结论 |
| --- | --- | --- | --- |
| `F.grid_sample` | BEVFormer、RF-DETR、Mask2Former、CREStereo、SimpleBEV | 直接使用标准 GridSample，固定 mode/align_corners/layout | **强参考** |
| Temporal MSDeformAttn | BEVFormer `MSDeformableAttention_TSA_Optimized` | offsets + weights + single-level grid_sample | UniAD TSA `num_levels=1`，优先实验 |
| Spatial MSDeformAttn3D | BEVFormer `MSDeformableAttention3D_SCA_Optimized` | camera reference + offsets + grid_sample | 接口很近，但 Qualcomm BEVFormer-Tiny core 是 single-level |
| 4-level MSDeformAttn | RF-DETR / Mask2Former multi-level core | split per level → grid_sample → concat/sum | **解决 UniAD 4-level SCA 的关键参考** |
| Decoder MSDeformAttn | BEVFormer `CustomMSDeformableAttention_Decoder_Optimized` | single-level deformable sampling + output proj | UniAD detection decoder 很接近 |
| DCNv2 | CenterNet template `custom_deformconv2d` | 生成采样坐标 → bilinear sample → regular conv | **新发现：很值得直接研究** |
| Scatter/ScatterND | BEVFormer、BEVDet | 固定 canvas + int64 index + scatter | 可参考，但动态 index 仍需验证 |
| `nonzero()` 动态长度 | StateTransformer 等 export patch | 去掉 data-dependent loop，改固定结构/全量计算/mask | 不要保留变长控制流 |
| fixed TopK | RF-DETR postprocess | 固定 K，输出 static shape | 通常比 threshold 后变长友好 |
| Linear on spatial grid | BEVFormer `OptimizedLinear` | `Linear → Conv2d(k=1)` + NCHW/NHWC 控制 | HTP/layout 优化参考 |

下面逐项拆。

---

# A. GridSample

## 3. 为什么 `grid_sample` 对 UniAD 是核心 primitive

很多看起来完全不同的操作最后都能归约到：

```text
feature map
+
sampling coordinates
        ↓
interpolation / sample
        ↓
sampled feature
```

对应：

```python
F.grid_sample(
    input,
    grid,
    mode="bilinear" or "nearest",
    padding_mode="zeros",
    align_corners=False,
)
```

在 UniAD 里它可以承载：

```text
Deformable Attention sampling
camera feature sampling
BEV warp / rotate
部分 deformable convolution sampling
```

Qualcomm 仓库里并不是只有 BEVFormer 在用它：

```text
BEVFormer
Mask2Former
RF-DETR
CREStereo
SimpleBEV
Detectron2 patches
custom_grid_sampler scorecard model
```

这比“某一个模型碰巧用了 grid_sample”更有参考价值。

---

## 4. GridSample 真正容易错的是语义，不是函数名

部署前固定：

```text
input layout     NCHW?
grid layout      [N,Hout,Wout,2]?
coordinates      [-1,1]?
order            (x,y) 还是 (y,x)?
mode             bilinear / nearest
padding_mode     zeros / border
align_corners    True / False
```

Deformable Attention 常见转换：

```text
reference point [0,1]
        ↓
reference * 2 - 1
        ↓
GridSample coordinate [-1,1]
```

一个 `align_corners` 不一致就足以产生明显数值偏差。

所以 standalone 测试一定要专门测：

```text
中心点
四个角
边界附近
完全越界
fractional coordinate
```

---

# B. Multi-Scale Deformable Attention

## 5. 原始 custom op 到底做什么

忽略 MMCV/CUDA class 名字，数学可以拆成：

```text
query
  │
  ├── Linear → sampling_offsets
  │
  └── Linear → attention_weights → softmax

value
  └── value_proj

reference_points
  + normalized offsets
        ↓
sampling_locations
        ↓
per-level interpolation
        ↓
weighted sum
        ↓
output_proj
```

所以真正的“特殊”主要集中在：

```text
按 reference point 动态采样 feature map
```

而不是 Linear/Softmax 本身。

---

## 6. Qualcomm BEVFormer：最接近 UniAD 接口，但有一个硬限制

Qualcomm patch 新增：

```text
MSDeformableAttention_TSA_Optimized
MSDeformableAttention3D_SCA_Optimized
CustomMSDeformableAttention_Decoder_Optimized
```

共同底层：

```text
custom_multi_scale_deformable_attn_pytorch_single_grid()
```

核心：

```text
value
→ reshape [B*heads, head_dim, H, W]
→ F.grid_sample
→ * attention_weights
→ sum(points)
```

**但公开 BEVFormer-Tiny QNN patch 的 helper 明确按 single level 特化：只使用 `value_spatial_shapes[0]`。**

所以它可以直接证明：

```text
single-level deformable attention
→ grid_sample graph
```

是一条 Qualcomm 已经认真优化过的路线。

它不能直接证明：

```text
UniAD num_levels=4 SCA
```

原样替换就能跑。

---

## 7. 4-level MSDeformableAttention 应该看 RF-DETR + Mask2Former

这个点比上一版文档更重要。

### RF-DETR

上游/Qualcomm recipe 使用 export-friendly MSDeformAttn，把不同 feature level 按固定 `(H,W)` split，然后每 level sampling，再合并。

### Mask2Former

Qualcomm `model_patches.py` 里甚至直接有：

```text
multi_scale_deformable_attention()
```

流程：

```text
value.split([H0*W0, H1*W1, ...])
        ↓
for each level:
    reshape → [B*heads,C,Hl,Wl]
    grid_sample
        ↓
concat sampled values
        ↓
* attention_weights
        ↓
sum
```

这就是 UniAD 4-level SCA core 最应该参考的结构。

所以后面不要只研究：

```text
BEVFormer single-level implementation
```

而应该组合：

```text
BEVFormer
    学 3D reference point / camera / SCA interface

RF-DETR + Mask2Former
    学 multi-level sample core
```

---

## 8. 一个 UniAD 4-level core 建议直接写成这种结构

概念代码：

```python
sampled_levels = []

for level, (h, w) in enumerate(static_hw):
    value_l = split_value[level]
    value_l = value_l.reshape(B * heads, head_dim, h, w)

    grid_l = sampling_grid_for_level[level]
    sampled_l = F.grid_sample(
        value_l,
        grid_l,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled_levels.append(sampled_l)

sampled = torch.cat(sampled_levels, dim=-1)
output = (sampled * attention_weights).sum(...)
```

第一版部署应该直接把：

```text
H0/W0
H1/W1
H2/W2
H3/W3
```

变成 static Python/config constants，而不是从运行时 `spatial_shapes` tensor 读 scalar 再驱动 `.view()`。

---

# C. DCNv2：这次新增一个非常值得看的 Qualcomm 参考

## 9. UniAD backbone 的 DCNv2 不应该继续标成“完全没参考”

Qualcomm `ai-hub-models` 的 CenterNet template 里已经有：

```text
custom_deformconv2d()
custom_dcn_forward()
```

它没有保留原始 compiled DCN op，而是把 DeformConv2d 展开。

流程非常清楚：

```text
input x
   │
   ├── conv_offset_mask
   │      ├── offset_y
   │      ├── offset_x
   │      └── sigmoid(mask)
   │
   ▼
regular convolution base grid p0
+
kernel relative offsets pk
+
learned offset
   ↓
sampling coordinates
   ↓
bilinear_sample
   ↓
* modulation mask
   ↓
reshape sampled patches
   ↓
普通 F.conv2d
```

这和 DCNv2 的数学本质高度一致。

---

## 10. CenterNet 这个 patch 对 UniAD 的价值是什么

不是：

```text
“UniAD DCNv2 直接 copy 就完事”
```

而是它提供了非常具体的 decomposition：

```text
DCNv2 custom CUDA op
        ↓
coordinate generation
+ bilinear sample
+ mask
+ normal Conv2d
```

这正好可以做 UniAD backbone 的 standalone 实验。

需要重点核对：

```text
kernel size
stride
padding
dilation
groups
deformable_groups
modulated mask
batch size
weight layout
padding edge behavior
```

Qualcomm 当前 `centernet_pose` recipe 公开目标主要是 QNN Context Binary / precompiled QNN ONNX，因此应把它当成“HTP/QNN-friendly DCN decomposition 证据”，而不是“原样 QNN_DLC 已验证”的过度结论。

---

# D. Scatter / ScatterND

## 11. Scatter 的问题通常不是 Scatter 本身，而是 index

BEV/SCA 常见：

```text
camera-visible queries
→ rebatch
→ attention
→ scatter back 到 BEV slots
```

风险：

```text
index 是否来自 nonzero？
index 长度是否数据相关？
重复 index 如何 reduce？
canvas shape 是否固定？
index dtype 是否 int64？
```

Qualcomm BEVFormer patch 有自定义 `ScatterND` 路径；BEVDet patch 也明确处理了 ONNX ScatterND/GatherND 所需的 int64 index。

所以 ScatterND 是**有参考实现**的。

但不要把它理解成：

```text
“任意动态 scatter 都安全”
```

如果 index 数量本身来自动态 `nonzero`，静态部署问题仍然存在。

---

## 12. UniAD SCA 的关键不应是“让 dynamic rebatch 原样导出”

更合理的是尝试把：

```text
每个 camera 可见 query 数量不同
```

变成：

```text
固定 max_len
+ validity mask
+ padding
```

即：

```text
动态 shape
→ 固定 shape + mask
```

Qualcomm BEVFormer 的 export 代码本身就在围绕 fixed padded query / mask / ScatterND 做修改。

这很可能是 UniAD SCA 上层真正的工作量，比 MSDA core 本身更麻烦。

---

# E. NonZero / 数据相关长度

## 13. `nonzero()` 本身不是绝对禁止；“nonzero 驱动后续 graph shape”才危险

区分：

### 情况 1：后处理 CPU

```python
idx = torch.nonzero(score > threshold)
```

然后只在 Python 画框。

可以直接留 CPU。

### 情况 2：网络内部

```python
idx = mask.nonzero()
x = x[idx]
max_len = idx.shape[0]
x = x.reshape(max_len, ...)
```

这会让后面 graph shape 依赖输入数据。

对 static HTP graph 极不友好。

Qualcomm 仓库里对类似问题的常见处理方向是：

```text
全量固定计算 + mask
固定上界 + pad
固定 K
消除 data-dependent Python loop
```

StateTransformer 的 export patch 就专门处理“nonzero 结果决定执行哪些 expert”的数据相关控制流问题。

所以 UniAD 中遇到 `nonzero()` 时，第一问题不是：

```text
QNN 有 NonZero op 吗？
```

而是：

```text
NonZero 的结果是不是改变后面 tensor 的 shape/control flow？
```

---

# F. TopK

## 14. 固定 K 和 threshold/filter 后变长是两回事

RF-DETR 的 postprocess 是很好的例子：

```text
num_select = num_queries
TopK(..., K=固定值)
```

输出：

```text
[B,K,...]
```

shape 是静态的。

这比：

```text
scores > threshold
→ nonzero
→ N 个结果
```

容易部署很多。

所以 UniAD 如果有动态筛选，优先考虑：

```text
fixed K + mask
```

而不是 variable N。

---

# G. Linear、Einsum、MatMul 与 layout

## 15. Qualcomm 很多 patch 并不是解决“unsupported op”，而是在减少 backend 不喜欢的图形状

典型：

### Linear → 1x1 Conv

BEVFormer `OptimizedLinear`：

```text
[B,H,W,C]
→ NCHW
→ Conv2d(k=1)
```

数学上等价于逐位置 Linear。

目的包括：

```text
保持规则 4D layout
减少 sequence ↔ image layout 来回转换
走成熟 Conv backend path
```

### Einsum → MatMul/reshape

Mask2Former 等 Qualcomm patch 常把：

```python
torch.einsum(...)
```

改成明确的：

```text
reshape
matmul
reshape
```

不是因为所有 Einsum 都不能编译，而是明确图结构通常更利于 export/lowering。

对 UniAD 也应优先使用可预测的 rank/layout。

---

# H. Dynamic Shape / tensor-derived shape

## 16. 这类代码在 eager 看起来最无害

例如：

```python
h = int(spatial_shapes[0, 0])
w = int(spatial_shapes[0, 1])
x = x.view(B, C, h, w)
```

如果 `h/w` 来自 tensor data，export 可能变成 data-dependent symbolic value。

RF-DETR 的 export 路径专门把：

```text
[(H0,W0), (H1,W1), ...]
```

作为 Python int list 往下传，用来 split/view。

Mask2Former Qualcomm patch 也使用：

```text
spatial_shapes_list: list[tuple[int,int]]
```

这已经形成很明确的迁移经验：

> **部署分辨率已固定时，不要坚持让 H/W 作为 runtime tensor 再控制图结构；把它提升成 compile-time/static config。**

---

# I. Rotate / BEV Warp

## 17. prev_bev rotate 可以继续用 GridSample 思路

BEVFormer temporal path 的 rotate/warp，本质可以写成：

```text
angle / center
→ affine/grid construction
→ F.grid_sample(prev_bev, grid)
```

检查：

```text
degree vs radian
正负方向
rotation center
NCHW/NHWC
mode
align_corners
out-of-bound padding
```

如果 current/prev BEV shape 固定，这条路线比保留任意 Python/OpenCV warp 更适合放进 graph。

---

# J. QNN custom op 什么时候才该上

## 18. 不是第一选择

只有在下面几步都失败后再考虑：

```text
标准 primitive decomposition
↓
Qualcomm 同类 patch
↓
静态化 shape/layout
↓
拆子图
↓
仍然无法在可接受性能/精度下表达
```

才值得研究：

```text
QNN custom op package
```

原因不是“custom op 不好”，而是它会引入新的工程面：

```text
op package build
backend implementation
target ABI
registration
部署额外 .so
跨 SDK/SoC 维护
性能实现
```

对于 UniAD 目前看到的关键算子，Qualcomm 已经给了很多 primitive decomposition 参考，暂时还没有理由第一步就自己写 HTP custom op。

---

# K. 现在真正值得拉到本地研究的 Qualcomm 文件

## 19. 第一组：UniAD BEV attention

```text
src/qai_hub_models/models/bevformer/external_repos/bevformertiny_minimal.diff
```

重点抽：

```text
deformable_attention.py
MultiheadAttention.py
custom_utils.py
spatial_cross_attention.py 的 patch
```

本仓库已有：

```bash
python scripts/extract-qualcomm-bevformer-patch-files.py ...
```

## 20. 第二组：真正 multi-level deformable attention

```text
src/qai_hub_models/models/mask2former/model_patches.py
```

重点：

```text
multi_scale_deformable_attention()
Patched...MultiscaleDeformableAttention
```

以及 RF-DETR 上游/recipe 的：

```text
MSDeformAttn
ms_deform_attn_core_pytorch
```

## 21. 第三组：UniAD backbone DCNv2

```text
src/qai_hub_models/models/templates/centernet/model_patches.py
```

重点：

```text
calculate_p0
calculate_pk
bilinear_sample
custom_deformconv2d
custom_dcn_forward
```

这个文件应该直接进入 UniAD DCNv2 实验的参考目录。

## 22. 第四组：Scatter/BEV geometry

```text
bevformer patch
bevdet/model_patches.py
simple_bev_cam patches
```

它们比普通 2D classification model 更接近多摄像头/BEV tensor 操作。

---

# L. 最小实验顺序现在可以更具体

```text
Experiment 1
F.grid_sample standalone

Experiment 2
CenterNet-style DCNv2 decomposition
原 DCNv2 vs custom_deformconv2d

Experiment 3
UniAD TSA single-level MSDA

Experiment 4
UniAD decoder single-level MSDA

Experiment 5
4-level MSDA core
参考 Mask2Former/RF-DETR

Experiment 6
SCA fixed camera rebatch + mask + scatter

Experiment 7
完整一层 BEVFormerEncoder
```

每一个都执行：

```text
Original eager
vs
Patched eager
vs
Export
vs
QNN/HTP
```

这比直接“export UniAD 看报错”快得多。

---

# M. 算子卡片模板

以后每遇到一个风险模块，用这个模板，不再写“支持/不支持”一句话：

```text
模块：DCNv2

原始实现：
mmcv / compiled deformable conv

数学分解：
offset/mask → bilinear sampling → Conv2d

Qualcomm 参考：
centernet template model_patches.py

输入 contract：
...

静态维度：
...

Original vs Patched：
max_abs = ...

Export：PASS/FAIL
QNN compile：PASS/FAIL
HTP Finalize：PASS/FAIL
HTP Execute：PASS/FAIL

性能：
API wall = ...
accelerator = ...

限制：
...
```

最终真正有价值的是这套经过 S25 Ultra 验证的算子经验库。

---

## 参考入口

- Qualcomm BEVFormer: `src/qai_hub_models/models/bevformer/`
- Qualcomm Mask2Former: `src/qai_hub_models/models/mask2former/model_patches.py`
- Qualcomm CenterNet DCN: `src/qai_hub_models/models/templates/centernet/model_patches.py`
- Qualcomm BEVDet: `src/qai_hub_models/models/bevdet/model_patches.py`
- Qualcomm SimpleBEV: `src/qai_hub_models/models/simple_bev_cam/`
- 本项目 UniAD 映射：`docs/uniad-qnn-operator-reference.md`
