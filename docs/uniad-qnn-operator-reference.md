# UniAD → QNN DLC 算子参考地图

这篇文档不是在宣称“UniAD 已经可以直接导出 QNN DLC”。它回答的是：**UniAD stage2_e2e 里哪些模块/算子已经有 Qualcomm AI Hub Models 的近似成功案例可以参考，哪些需要扩展，哪些应该先放在 QNN 图外。**

参考对象：

- OpenDriveLab UniAD `projects/configs/stage2_e2e/base_e2e.py`
- UniAD `spatial_cross_attention.py` / `temporal_self_attention.py`
- Qualcomm AI Hub Models 的 BEVFormer QNN 适配 patch
- Qualcomm AI Hub Models 的 RF-DETR export/QNN 路径

## 1. 先看 UniAD stage2 的关键静态参数

主配置中：

```text
embed_dims = 256
feature levels = 4
BEV = 200 x 200
num_query = 900
```

BEV encoder：

```text
TemporalSelfAttention:
    num_levels = 1

SpatialCrossAttention -> MSDeformableAttention3D:
    num_levels = 4
    num_points = 8
```

Detection decoder：

```text
MultiheadAttention
CustomMSDeformableAttention:
    num_levels = 1
```

Segmentation head：

```text
MultiScaleDeformableAttention:
    num_levels = 4
```

Motion head：

```text
MotionDeformableAttention:
    num_levels = 1
    num_heads = 8
    num_points = 4
    num_steps = 12
```

所以不要把“UniAD 的 deformable attention”当成一个算子；至少有 4 种不同使用场景。

## 2. 总体优先级

| UniAD 模块/算子 | 风险 | Qualcomm 可参考实现 | 结论 |
| --- | --- | --- | --- |
| ResNet 普通 Conv/BN/ReLU | 低 | 大量 AI Hub 模型 | 先保留 |
| DCNv2 backbone | 高 | Qualcomm BEVFormer 为部署改成普通 ResNet50 路径，没有直接沿用 UniAD DCNv2 | 单独处理，不能跳过 |
| FPN | 低~中 | Qualcomm BEVFormer | 优先原样 export 验证 |
| TemporalSelfAttention, level=1 | 中 | `MSDeformableAttention_TSA_Optimized` | **最接近直接参考** |
| SpatialCrossAttention 上层 rebatch/scatter | 高 | Qualcomm BEVFormer SCA patch | **强参考，但需适配 UniAD 200x200** |
| MSDeformableAttention3D, level=4 | 高 | Qualcomm `MSDeformableAttention3D_SCA_Optimized` 只验证 level=1 | 核心数学可复用，必须补 multi-level |
| Decoder MultiheadAttention | 中 | `MultiheadAttention_Optimized` | 可参考 1x1 Conv/layout 改写 |
| Decoder CustomMSDeformableAttention, level=1 | 中 | `CustomMSDeformableAttention_Decoder_Optimized` | **很接近直接参考** |
| Seg head MultiScaleDeformableAttention, level=4 | 高 | RF-DETR multi-level PyTorch core + QNN recipe | 用 RF-DETR 参考比 BEVFormer 更合适 |
| MotionDeformableAttention, level=1 | 中~高 | Qualcomm single-level deformable sampling core | 核心 sampling 可复用，外层 `num_steps` 需重写 |
| Standard MHA / FFN / LayerNorm | 中 | Qualcomm BEVFormer optimized MHA/Linear | 优先标准化/静态化 |
| BEV rotate/warp | 中 | Qualcomm `custom_rotate` + `grid_sample` | 可直接借鉴表达方式 |
| ScatterND | 高 | Qualcomm `custom_utils.ScatterND` | 可参考，但需验证索引 contract |
| 动态 `nonzero` / 变长 query | 很高 | Qualcomm SCA static padding/top-k/mask 思路 | 必须静态化 |
| Planning collision optimizer | 图外 | 无需硬塞 QNN | **保留 CPU 后处理** |

## 3. Backbone：先处理 DCNv2 这个现实问题

UniAD stage2 使用 ResNet101，并在后两 stage 打开 DCNv2：

```text
stage_with_dcn = (False, False, True, True)
```

Qualcomm BEVFormer QNN 适配选择的是更简单的 ResNet50/普通卷积配置，而不是证明原版 BEVFormer/UniAD 的 DCNv2 可以原样进入 QNN。

因此第一轮完整 DLC 实验不能假设 backbone 已解决。建议两个方向并行：

```text
A. 给 DCNv2 做最小 export/QNN 测试
B. 准备无 DCN 的部署 backbone 版本，做权重/精度迁移实验
```

如果目标是先打通 UniAD 主干链路，B 往往更可控。

## 4. TemporalSelfAttention：第一优先级实验

UniAD 配置中 TSA 是：

```text
num_levels = 1
num_bev_queue = 2
num_points = 4
embed_dims = 256
num_heads = 8
```

这与 Qualcomm BEVFormer patch 中的：

```text
MSDeformableAttention_TSA_Optimized
```

非常接近。

Qualcomm 的改写思想：

```text
query/current+history BEV
→ sampling_offsets / attention_weights
→ reference_points + normalized offset
→ grid_sample
→ weighted sum
```

并把规则 BEV tensor 组织成 NCHW/NHWC，空间 Linear 可替换为 1x1 Conv。

### 这里要注意

Qualcomm BEVFormer-Tiny 实验 BEV 更小，UniAD 是 200x200；“算子能表达”不等于 40k BEV queries 的性能/内存一定可接受。

所以 TSA 应先做独立 DLC benchmark：

```text
[1,200,200,256] query
+ prev_bev
+ reference_points
→ TSA
```

同时记录 QNN graph memory 和 Execute 时间。

## 5. SpatialCrossAttention 上层：比 deformable core 更危险

UniAD 原始 SCA 在每个 camera 上：

```text
bev_mask
→ sum/nonzero
→ 得到每个 camera 的有效 query index
→ max(variable lengths)
→ 创建 max_len buffer
→ Python loop rebatch
→ deformable attention
→ 再按 index scatter/add 回 slots
→ camera count normalize
```

这段的风险是：**动态 query 数 + `nonzero` + Python loop + scatter**，不只是 deformable attention 本身。

Qualcomm BEVFormer patch 正好值得参考这里。它引入了：

```text
padded_query_len(...)
sca_use_topk_query_pruning
sca_masking_before_gridsample
ScatterND
固定/受控的 query canvas
```

目标是把“不定长 camera query list”改成更静态的 tensor graph。

因此 UniAD SCA 推荐先分成两个子问题：

```text
A. camera rebatch/scatter 静态化
B. MSDeformableAttention3D core
```

不要一次同时改。

## 6. MSDeformableAttention3D：Qualcomm 有参考，但不是直接替换

UniAD 原版 SCA core：

```text
num_levels = 4
num_points = 8
```

它生成：

```text
sampling_offsets:
[B,Q,heads,levels,points,2]

attention_weights:
[B,Q,heads,levels,points]
```

然后原代码调用 MMCV/CUDA `MultiScaleDeformableAttnFunction`。

Qualcomm 的：

```text
MSDeformableAttention3D_SCA_Optimized
```

把 CUDA extension 改成了 `grid_sample + weighted sum`，但其公开 BEVFormer-Tiny QNN 配置只有：

```text
num_levels = 1
```

底层 helper 也只读取 `spatial_shapes[0]`。

### 所以 UniAD 要补的真正代码

不是重新发明 attention，而是把 Qualcomm single-level core 扩成：

```text
for each fixed level:
    slice value_l
    reshape [B*head,C,H_l,W_l]
    build grid_l
    grid_sample

concat/stack all levels
× attention_weights
sum(level * point)
```

这里 level 数固定为 4，完全可以做静态展开，不需要运行时 Python loop。

### 第二个参考：RF-DETR

Qualcomm RF-DETR 使用的 `ms_deform_attn_core_pytorch` 本身有多 level sampling 逻辑，而且 RF-DETR recipe 已支持 QNN DLC/Context Binary。

因此对 UniAD 4-level SCA，最值得做的是：

```text
BEVFormer Qualcomm：参考 3D reference point/layout/SCA 接口
+
RF-DETR Qualcomm：参考 multi-level sampling decomposition
```

把两者组合，而不是只抄一个文件。

## 7. Detection Decoder：比 SCA 更接近可直接搬

UniAD detection decoder：

```text
MultiheadAttention
+
CustomMSDeformableAttention(num_levels=1)
```

Qualcomm BEVFormer patch 已有对应：

```text
MultiheadAttention_Optimized
CustomMSDeformableAttention_Decoder_Optimized
```

而且 Qualcomm 部署配置同样把 decoder deformable attention 设为 single-level。

所以这块优先级很高：如果权重/shape contract 对得上，可以先尝试源码层替换，再做 PyTorch 数值对齐。

## 8. Segmentation Head：不要拿 single-level BEVFormer core 硬套

UniAD Pansegformer 的 encoder/decoder 使用：

```text
MultiScaleDeformableAttention(num_levels=4)
```

这一块更适合参考 RF-DETR 的 multi-level deformable attention export core。

需要额外核对：

```text
reference_points shape
num_points
value layout
attention weight layout
output projection
```

思想可复用，不代表 class API 可直接替换。

## 9. MotionDeformableAttention

UniAD MotionHead 配置：

```text
num_levels = 1
num_heads = 8
num_points = 4
num_steps = 12
```

源码同样依赖 `MultiScaleDeformableAttnFunction_fp32`，但它额外把未来 trajectory steps 编进 sampling offsets / attention weights，并有 `reference_trajs`、bbox 等运动语义。

因此可复用：

```text
single-level value projection
sampling location normalization
grid_sample
weighted sum
```

不可直接照搬：

```text
sampling_offsets/weights 的 num_steps 维
reference trajectory 构造
多 step output fusion
```

建议在 SCA/TSA core 跑通后第三个做它。

## 10. Planning Head：明确划 QNN / CPU 边界

Planning 主干本身包含：

```text
Embedding
Linear/LayerNorm/ReLU
TransformerDecoder
Conv adapter
cumsum
```

这些可以继续研究 QNN export。

但 `use_col_optim=True` 的测试后处理明确包含：

```text
torch.nonzero(occupancy)
动态筛选
.cpu().detach().numpy()
CollisionNonlinearOptimizer
外部非线性求解
numpy → torch
```

这一段不要尝试塞进 DLC。

建议产品边界：

```text
QNN DLC 输出 raw planning trajectory + occupancy
        ↓
CPU
collision nonlinear optimization
        ↓
最终轨迹
```

先把 QNN graph 和 CPU 后处理边界画清楚，比强行“全模型一个 DLC”更现实。

## 11. Query Interaction / Memory Bank：整模型静态化的大风险

UniAD tracking 不是单帧纯前馈模型。它有 track query、query interaction、memory bank、score threshold 等状态逻辑。

对 QNN 来说建议优先设计：

```text
固定容量 query slots
+
valid mask
+
host 侧状态管理
```

而不是让 graph 输出/输入不定长 query list。

因此第一版 DLC 最好明确哪些 state tensor 是 graph I/O：

```text
prev_bev
track query features
track reference points
valid mask
memory state
```

这属于模型接口设计，不只是 op support。

## 12. `OptimizedLinear` 可以在哪些地方借鉴

Qualcomm BEVFormer 把规则空间 tensor 上的 Linear 映射成 1x1 Conv2d。

UniAD 里最值得考虑的场景：

```text
BEV [B,H,W,C]
image feature [B,C,H,W]
attention offset/weight projection
FFN 中保持空间 layout 的 projection
```

不建议一上来全局替换所有 `nn.Linear`；先针对 BEV/SCA/TSA 做。

## 13. Rotate / geometry 可以直接参考 BEVFormer

UniAD 的 BEV encoder 继承了 BEVFormer 的：

```text
rotate_prev_bev
use_shift
can_bus
reference point projection
lidar2img
bev_mask
```

Qualcomm BEVFormer 已经为 export 做了大量改写，包括 `grid_sample` rotate、固定输入 `can_bus/lidar2img`、SCA mask/query 处理。

这一部分是目前最直接的上游参考，优先从 Qualcomm BEVFormer patch 对照 UniAD，而不是从零改 UniAD。

## 14. 第一阶段不要导整个 UniAD，建议这个顺序

### Experiment 1：TSA

```text
TemporalSelfAttention(level=1)
```

目标：验证 Qualcomm TSA optimized 路线能否直接映射 UniAD 权重。

### Experiment 2：Decoder MSDA

```text
CustomMSDeformableAttention(level=1)
```

目标：验证 decoder optimized 路线。

### Experiment 3：4-level deformable core

```text
只做 MSDeformableAttention3D core
levels=4
```

组合 BEVFormer + RF-DETR 的 Qualcomm 参考。

### Experiment 4：SpatialCrossAttention

加上：

```text
camera rebatch
mask
scatter/camera aggregation
```

### Experiment 5：BEV encoder

```text
TSA + SCA + FFN x 1 layer
→ x 6 layers
```

### Experiment 6：Backbone/neck

单独处理 DCNv2，再与 BEV encoder 拼接。

### Experiment 7：Detection decoder

### Experiment 8：Motion / Occ / Planning neural part

碰到 CPU solver、动态 tracking state 时主动划 graph boundary。

## 15. Qualcomm 源码参考入口

`qualcomm/ai-hub-models`：

```text
src/qai_hub_models/models/bevformer/model.py
src/qai_hub_models/models/bevformer/external_repos/bevformertiny_minimal.diff
src/qai_hub_models/models/rf_detr/model.py
```

BEVFormer patch 应重点抽出：

```text
projects/mmdet3d_plugin/bevformer/modules/deformable_attention.py
projects/mmdet3d_plugin/bevformer/modules/MultiheadAttention.py
projects/mmdet3d_plugin/custom_utils.py
projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py 的 patch
projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py 的 patch
```

仓库里的 `scripts/extract-qualcomm-bevformer-patch-files.py` 用于从 Qualcomm 官方 patch 重建其中“新增文件”，避免手工复制第三方 800+ 行源码。

## 16. 当前最值得先验证的三个结论

1. **UniAD TSA 是 single-level，和 Qualcomm TSA optimized 最接近。** 这是最适合第一个做 standalone DLC 的 attention。
2. **UniAD SCA 是 4-level，而 Qualcomm BEVFormer QNN 版本是 1-level。** 不能直接替换，但可以用 Qualcomm BEVFormer 的 SCA/layout + RF-DETR 的 multi-level sampling 拼出部署版。
3. **完整 UniAD 不应该先追求“所有逻辑都进一个 DLC”。** Collision optimizer、动态 track state 等天然适合留在 CPU；先定义稳定的 QNN graph I/O 边界。