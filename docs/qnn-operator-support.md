# QNN 特殊算子迁移：GridSample、DeformableAttention、Scatter、动态 shape

这篇文档解决：**遇到 QNN converter/HTP 不友好的特殊算子时，应该怎么判断、怎么改写、什么时候参考 Qualcomm 现成模型。**

## 1. 先不要把“特殊算子”直接等同于“必须写 QNN custom op”

一个 PyTorch 模型里出现：

```text
mmcv.ops.MultiScaleDeformableAttention
custom CUDA extension
torch.autograd.Function
ScatterND
grid_sample
dynamic nonzero/topk
```

只说明原模型用了一个特殊实现，不说明部署时必须保留同一个实现。

优先级应该是：

```text
1. 找数学等价的标准 PyTorch/ONNX primitives
2. 找 Qualcomm AI Hub Models 已经跑通过的改写方式
3. 固定 shape/layout，降低动态图复杂度
4. 拆成多个可编译子图
5. 最后才考虑 QNN custom op package
```

## 2. 如何判断“QNN 支持一个算子”

不要只看算子名字。

至少分四层：

```text
PyTorch 能不能表达
        ↓
export / ONNX 能不能表达
        ↓
QNN converter 能不能接收
        ↓
HTP backend 能不能成功 Finalize + Execute
```

例如某个 `grid_sample`：

```text
torch.export 成功
```

不等于：

```text
QNN DLC 一定成功
```

更不等于：

```text
在目标 HTP 上性能一定好
```

所以项目中将“支持”定义成：

> 在固定 input contract 下，最终能生成 DLC，并在目标手机 HTP 上 Finalize、Execute，且数值与参考实现对齐。

## 3. GridSample

### 为什么重要

Deformable Attention、BEV feature sampling、图像 warp/rotate 等都可能归约到：

```text
grid generation
+
F.grid_sample
```

### Qualcomm 已有参考

Qualcomm 的 BEVFormer patch 把 deformable attention 核心采样改写成：

```text
value
→ reshape to [B*heads, C, H, W]
→ F.grid_sample(...)
→ attention weight
→ weighted sum
```

这说明对其已经发布的 BEVFormer/QNN 路径，`grid_sample` 是一个现实可参考的表达方式。

### 实际迁移要检查

```text
mode = nearest / bilinear
align_corners
padding_mode
input rank
grid rank
坐标范围 [-1, 1]
静态 H/W
```

不同这些参数会改变数值，不能只说“都是 grid_sample”。

### 备选：gather-based sampling

如果目标工具链对 `grid_sample` 表达不友好，可以考虑把 bilinear sampling 展开为：

```text
floor/ceil
clip
linear index
gather 4 corners
compute weights
weighted sum
```

但 graph 会更大，应独立 profile 后再决定。

## 4. Multi-Scale Deformable Attention

### 原始形式

很多 DETR/BEV 模型使用 MMCV/CUDA extension：

```text
MultiScaleDeformableAttnFunction
```

这类 extension 不能期待 QNN converter 自动理解其 CUDA kernel。

### 推荐拆法

数学上拆成：

```text
query
 ├─ Linear → sampling_offsets
 └─ Linear → attention_weights

value
 └─ Linear / 1x1 Conv → projected value

reference_points + normalized offsets
      ↓
sampling locations
      ↓
per-level grid sample
      ↓
weighted sum
      ↓
output projection
```

这样转换问题从“一个未知 custom op”变成一组标准 tensor primitives。

### Qualcomm BEVFormer 的限制

其当前公开优化版底层 helper 明确按 **single feature level** 特化：只读取 `spatial_shapes[0]`。

因此：

```text
Qualcomm BEVFormer QNN success
```

能够证明：

```text
single-level deformable attention 的这套改写路线可参考
```

不能直接证明：

```text
任意 4-level MSDeformableAttention 原封不动可进 QNN
```

真正 multi-level 时需要显式处理每一 level，再 concat/stack/weighted sum。

## 5. Scatter / ScatterND

### 为什么危险

BEV/3D 模型经常：

```text
根据索引把 camera/query 结果重新散射到 BEV canvas
```

这可能涉及：

```text
ScatterND
index_put
advanced indexing
nonzero + scatter
```

这些模式的困难通常不是“加法本身”，而是：

```text
索引是否动态
输出尺寸是否静态
是否存在 duplicate index
reduction 是 assign / add / max
```

### Qualcomm 已有参考

Qualcomm BEVFormer patch 的 `custom_utils.py` 中提供了 ScatterND 风格的实现，用于改写原 BEVFormer 的 query/camera 聚合路径。

对 UniAD 的意义是：

> 如果碰到类似 BEV camera rebatch → scatter 回原 BEV query 的逻辑，先看 Qualcomm BEVFormer 的写法，而不是从零设计。

### 更好的策略

如果索引集合在固定输入尺寸下有可静态化部分，尽量把它们前移为常量；如果必须动态，单独做最小模型验证 converter 行为。

## 6. `nonzero()` 与变长 tensor

这是导出时非常常见的风险。

例如：

```python
idx = mask.nonzero()
x = x[idx]
```

`idx.shape[0]` 由数据决定。

随后如果：

```text
reshape
concat
for-loop
scatter output shape
```

依赖这个长度，静态编译会非常难。

### 改写思路

优先考虑：

```text
固定最大长度 + mask
Top-K 固定 K
padding
无效位置填 0 / 越界 grid
```

Qualcomm BEVFormer 就有类似思想：无效 camera sampling location 可以放到 `grid_sample` 合法区域之外，让其自然采样到 padding，而不是产生不定长执行结构。

## 7. TopK

TopK 本身不一定是问题，关键是：

```text
K 是否常量
TopK 输出是否继续产生动态控制流
```

更友好的：

```text
K = 300
固定输出 [B, 300]
```

更危险的：

```text
K = tensor.item()
根据 score threshold 得到不定长 K
```

目标是让 TopK 成为“静态 shape 的选择算子”，而不是动态 shape 生成器。

## 8. Rotate / Warp

BEV 时序融合经常对 `prev_bev` 做旋转/平移。

部署时可考虑：

```text
grid construction
→ grid_sample
```

Qualcomm BEVFormer patch 的 `custom_utils.py` 也提供了基于 `grid_sample` 的自定义旋转逻辑。

需要严格对齐：

```text
center
角度方向
degree/radian
align_corners
interpolation mode
coordinate convention
```

否则 export 虽成功，BEV 数值会系统性偏移。

## 9. Linear → 1x1 Conv

Qualcomm BEVFormer 定义 `OptimizedLinear`：

```text
Linear(Cin -> Cout)
```

在空间 tensor 上改写为：

```text
Conv2d(Cin, Cout, kernel=1)
```

其意义主要是：

```text
保持 4D NCHW/NHWC layout
减少 sequence/layout 来回变换
使用 backend 更成熟的 Conv 路径
```

它不是任何时候都该替换 `nn.Linear`，但如果 UniAD 某段：

```text
[B, H, W, C]
```

本来就代表规则 BEV/image grid，就非常值得测试。

## 10. Multihead Attention

标准 MHA 数学上可拆：

```text
Q/K/V projection
reshape heads
Q @ K^T
scale
softmax
attention @ V
output projection
```

这通常比依赖某个 fused PyTorch op 更容易迁移。

Qualcomm BEVFormer 还提供了 `MultiheadAttention_Optimized`，并支持将 head 拆开处理的路径。

对 UniAD：如果原模型使用 MMCV MHA wrapper，不要执着保留 wrapper；更重要的是保持权重映射和输出数学一致。

## 11. LayerNorm / normalization

普通 LayerNorm 通常不是第一风险项，但应注意：

```text
normalized_shape
输入 rank
axis
布局转换前后归一化维度是否仍然是 channel/embed dim
```

从：

```text
[B, N, C]
```

改成：

```text
[B, C, H, W]
```

以后不能机械地在同一个 axis 上做 LayerNorm。

## 12. Dynamic reshape / shape tensor

例如：

```python
h = spatial_shapes[0, 0]
w = spatial_shapes[0, 1]
x.view(b, c, h, w)
```

在 eager 模式很自然，但 exporter 可能把 `h/w` 变成运行时 tensor value。

如果部署输入分辨率固定，更稳的方法是：

```text
把 H/W 变成 Python/static config
```

RF-DETR 上游为了 export 也有类似处理：把 per-level `(H, W)` 以 Python int pairs 传入，避免在 export 时从 tensor 数据中提取 shape 值。

## 13. 数据相关 assert

类似：

```python
assert (spatial_shapes[:,0] * spatial_shapes[:,1]).sum() == len_input
```

在 eager 是很好的 sanity check，但 export 时可能变成 data-dependent guard。

处理方式：

```text
如果 shape 已经由静态 contract 保证：
    导出路径删除/替换该 assert
如果 shape 真可能变化：
    不能简单删，需要重新设计 contract
```

Qualcomm RF-DETR recipe 就对 MSDeformAttn export 的 data-dependent assert 做过专门 patch。

## 14. 算子参考的优先来源

迁 UniAD 时建议按这个顺序找参考：

```text
1. Qualcomm AI Hub Models 同类模型
2. 同一模型的 Qualcomm patch
3. 上游模型自己的 ONNX/export 路径
4. PyTorch/ONNX 标准 decomposition
5. QNN custom op
```

尤其关注 Qualcomm 已经发布 QNN DLC 的：

```text
BEVFormer
RF-DETR
BEVFusion 等 3D/BEV 模型
```

因为这些代码不只是“理论实现”，而是配套了实际 QNN runtime 目标的模型 recipe。

## 15. 给每个特殊算子建一张卡片

建议以后每碰到一个风险 op 都记录：

```text
模块：MSDeformableAttention3D
原始实现：mmcv CUDA extension
输入：...
输出：...
动态维度：...
Qualcomm 参考：BEVFormer
改写候选：grid_sample + weighted sum
PyTorch 数值：PASS/FAIL
ONNX：PASS/FAIL
QNN DLC：PASS/FAIL
HTP Finalize：PASS/FAIL
HTP Execute：PASS/FAIL
性能：...
```

这样最终形成的是自己的“QNN 算子经验库”，而不是零散报错记录。