# QNN 特殊算子迁移：GridSample、DeformableAttention、Scatter、动态 shape

这篇文档解决：**遇到 QNN converter/HTP 不友好的特殊算子时，怎么判断、怎么改写、什么时候优先参考 Qualcomm 现成模型。**

## 1. 特殊算子不等于必须写 QNN custom op

看到：

```text
mmcv.ops.MultiScaleDeformableAttention
custom CUDA extension
torch.autograd.Function
ScatterND
grid_sample
dynamic nonzero/topk
```

只说明原模型用了特殊实现，不说明部署时必须保留同一个实现。

优先顺序：

```text
1. 找数学等价标准 PyTorch/ONNX primitives
2. 找 Qualcomm AI Hub Models 已跑通的改写
3. 固定 shape/layout
4. 拆成可编译子图
5. 最后才考虑 QNN custom op package
```

## 2. “支持”必须分四层

```text
PyTorch 能表达
→ export / ONNX 能表达
→ QNN converter 能接收
→ HTP 能 Finalize + Execute
```

本项目把“支持”定义为：固定 input contract 下，最终能生成 DLC，在目标手机 HTP 上 Finalize、Execute，并与参考实现数值对齐。

## 3. GridSample

Deformable Attention、BEV feature sampling、warp/rotate 都可能归约到：

```text
grid generation + F.grid_sample
```

Qualcomm BEVFormer patch 把 deformable attention 核心采样改成：

```text
value
→ reshape [B*heads,C,H,W]
→ F.grid_sample
→ attention weight
→ weighted sum
```

这说明在其已发布 BEVFormer/QNN 路径里，`grid_sample` 是现实可参考的表达方式。

迁移时必须对齐：

```text
nearest/bilinear
align_corners
padding_mode
input/grid rank
坐标范围 [-1,1]
静态 H/W
```

如果目标工具链对 `grid_sample` 不理想，可研究 gather-based bilinear sampling：取四角、计算权重、加权求和；但图会变大，需要独立 profile。

## 4. Multi-Scale Deformable Attention

原始模型常调用 MMCV/CUDA extension。部署时建议拆成：

```text
query
 ├─ projection → sampling_offsets
 └─ projection → attention_weights

value → projection
reference_points + normalized offsets
→ sampling locations
→ per-level sample
→ weighted sum
→ output projection
```

这样“未知 custom op”变成一组普通 tensor primitives。

### Qualcomm BEVFormer 的重要限制

公开优化版的底层 sampling helper 按 **single feature level** 特化，只读取 `spatial_shapes[0]`。所以它能证明 single-level 改写路线有成功参考，不能证明任意 4-level MSDeformableAttention 原封不动可进 QNN。

真正 multi-level 时需要明确逐 level sampling，再组合结果。

## 5. Scatter / ScatterND

BEV/3D 模型经常要把 camera/query 结果散射回 BEV canvas。风险点不只是“scatter”名字，而是：

```text
index 是否动态
输出尺寸是否静态
duplicate index 怎么处理
reduction 是 assign/add/max
```

Qualcomm BEVFormer patch 的 `custom_utils.py` 提供了 ScatterND 风格实现，可作为 camera rebatch → scatter back 的参考。

如果固定输入下部分 index 可静态化，优先把它前移成常量；必须动态的则单独做最小模型验证。

## 6. `nonzero()` 与变长 tensor

例如：

```text
idx = mask.nonzero()
x = x[idx]
```

`idx.shape[0]` 由数据决定，后续如果参与 reshape/concat/loop/scatter，就很容易破坏静态编译。

改写优先考虑：

```text
固定最大长度 + mask
固定 K 的 TopK
padding
无效 grid 放到采样边界外
```

Qualcomm BEVFormer 对无效 camera sample 就采用了让 sampling location 越界、由 grid sampling 得到 padding 的思路。

## 7. TopK

TopK 的关键不是算子本身，而是 K 是否固定：

```text
K=300 → 输出静态 shape，通常更友好
K=tensor.item() / threshold 后变长 → 高风险
```

## 8. Rotate / Warp

BEV 时序融合常对 `prev_bev` 做旋转/平移，可考虑：

```text
grid construction → grid_sample
```

Qualcomm BEVFormer `custom_utils.py` 有基于 `grid_sample` 的 rotate 参考。必须对齐 center、角度方向、degree/radian、interpolation、align_corners 和坐标约定。

## 9. Linear → 1x1 Conv

Qualcomm BEVFormer 定义 `OptimizedLinear`，把空间位置上的 Linear 改成 1x1 Conv2d，用来维持规则 NCHW/NHWC layout、减少 sequence/layout 变换、走更成熟的卷积路径。

它不是所有 Linear 都该替换，但对 `[B,H,W,C]` / `[B,C,H,W]` 这类规则 BEV/image grid 很值得测试。

## 10. Multihead Attention

标准 MHA 可拆成：

```text
Q/K/V projection
reshape heads
Q @ K^T
scale + softmax
attention @ V
output projection
```

Qualcomm BEVFormer 还有 `MultiheadAttention_Optimized`，说明原 MMCV wrapper 并不是必须保留；真正要保持的是权重映射和数学结果。

## 11. Dynamic reshape / shape tensor

类似：

```text
h = spatial_shapes[0,0]
w = spatial_shapes[0,1]
x.view(b,c,h,w)
```

在 eager 很自然，但 export 可能把 H/W 变成运行时 tensor value。若部署分辨率固定，更稳的是把 H/W 作为静态配置/Python int。

RF-DETR 上游也采用了向 export 路径显式传 Python `(H,W)` 的做法，避免从 tensor 数据提取 shape。

## 12. 数据相关 assert

类似：

```text
assert sum(H_i*W_i) == len_input
```

在 eager 是好检查，在 export 里可能成为 data-dependent guard。若 shape 已由静态 contract 保证，可以为 export 路径移除/替换；如果 shape 真会变化，则不能简单删。

Qualcomm RF-DETR recipe 就专门 patch 过 MSDeformAttn 的 data-dependent assert。

## 13. 算子参考的优先来源

```text
1. Qualcomm AI Hub Models 同类模型
2. 同一模型的 Qualcomm patch
3. 上游模型自己的 export 路径
4. 标准 PyTorch/ONNX decomposition
5. QNN custom op
```

重点看已发布 QNN DLC 的 BEVFormer、RF-DETR、BEVFusion 等，因为这些代码不仅“理论能写”，而是配套了 QNN runtime 目标。

## 14. 给每个风险算子建卡片

```text
模块：MSDeformableAttention3D
原始实现：MMCV CUDA extension
输入/输出：...
动态维度：...
Qualcomm 参考：BEVFormer
候选改写：grid_sample + weighted sum
PyTorch 数值：PASS/FAIL
ONNX/export：PASS/FAIL
QNN DLC：PASS/FAIL
HTP Finalize：PASS/FAIL
HTP Execute：PASS/FAIL
性能：...
```

最终要形成自己的 QNN 算子经验库，而不是零散的 converter 报错记录。