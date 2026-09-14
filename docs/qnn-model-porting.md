# 从 PyTorch 开源模型到 QNN DLC

这篇文档解决：**拿到一个新的 PyTorch 开源模型后，怎么判断它能不能进 QNN，怎么一步步变成 DLC。**

## 1. 当前项目已经解决的是后半段

`qnn_mobile` 已经验证：

```text
DLC → Compose → Finalize → Context Binary → Execute
```

UniAD 新增的难点在前半段：

```text
PyTorch source + checkpoint
→ 可静态导出的 PyTorch graph
→ ONNX / Qualcomm 接受的中间图
→ QNN converter/compiler 能接受的算子和 shape
→ DLC
```

所以“会运行 DLC”和“会把模型做成 DLC”是两件事。

## 2. 不要第一步就导整模型

先按模块切开：

```text
preprocess
backbone
neck
BEV encoder
  ├── Temporal Self Attention
  └── Spatial Cross Attention
Decoder
task heads
postprocess
```

给每块标记：

```text
A 普通静态算子
B 有 Qualcomm 成功参考
C 特殊算子，需要改写
D 动态控制流/动态 shape，高风险
E 可以放模型外/CPU
```

## 3. 第一轮静态审计

重点找：

```text
Tensor.item()/tolist()
nonzero()
数据相关 if/for
动态 reshape/slice
custom CUDA/C++ extension
mmcv.ops
动态 TopK / 动态 query 数
动态 camera / level / H/W
```

部署优先固定：

```text
batch = 1
camera 数 = 常量
输入 H/W = 常量
BEV H/W = 常量
num_query = 常量
num_levels = 常量
num_points = 常量
```

## 4. 自定义 CUDA op 的处理顺序

看到 `torch.autograd.Function`、`ext_loader.load_ext`、`mmcv.ops` 时先标红，但不要直接跳到 QNN custom op。

优先顺序：

```text
1. 用标准 PyTorch/ONNX primitives 重写
2. 参考 Qualcomm AI Hub Models 同类 patch
3. 固定 shape/layout 降低动态图复杂度
4. 拆成可编译子图
5. 最后才研究 QNN custom op package
```

Qualcomm BEVFormer 的 deformable attention 就是典型：原 MMCV CUDA extension 被改成更容易导出的普通 tensor 运算和 `grid_sample` 路径。

## 5. 建立最小可导出单元

每个风险模块包成独立 `nn.Module`，固定输入 contract：

```text
name
shape
dtype
layout
value range
哪些维度允许动态
```

例如先单测：

```text
query + value + reference_points + spatial_shapes + mask
→ MSDeformableAttention3D
→ output
```

而不是整个 UniAD。

## 6. 每个单元保留三套数值基线

```text
A 原始 PyTorch
B 改写后 PyTorch
C 导出图 / QNN
```

正确顺序：

```text
A vs B 先对齐
B vs export 再对齐
export vs QNN 最后对齐
```

记录至少：最大绝对误差、平均绝对误差、必要时 cosine/PSNR，以及任务关键输出。

## 7. PyTorch 改写原则

### 保留数学含义，换表达形式

例如空间位置上的：

```text
Linear(Cin→Cout)
```

可以在权重映射正确时改成：

```text
Conv2d(Cin,Cout,kernel=1)
```

### 优先静态 tensor primitives

偏好：

```text
reshape/permute/concat/static split
matmul/conv/softmax/add/mul/grid_sample
```

警惕：

```text
nonzero 产生变长索引
数据相关循环
数据相关 reshape
Tensor value 转 Python 控制流
```

### 非核心后处理可移出图

可视化、字符串类别名、复杂 Python 结构组装等可以放 CPU；但 geometry、attention、BEV 时序融合等主干计算不能随意移走。

## 8. 导出阶段记录什么

每次 export 都记录：

```text
checkpoint hash
代码 commit
input spec
PyTorch/opset/工具版本
做过哪些 patch
output name/shape/dtype
```

检查图里是否出现：

```text
custom domain op
ATen fallback
意外动态维度
巨大 Expand/Tile
Shape/Gather 驱动的数据相关逻辑
```

成功生成 ONNX 不等于适合 QNN。

## 9. QNN 转换失败分类

### Operator unsupported

找等价标准算子、Qualcomm patch；custom op 放最后。

### Shape inference / dynamic shape

固定输入，去掉数据相关 reshape，把固定 H/W 等提前成静态常量。

### Layout / rank / backend constraint

尝试 NCHW/NHWC、降低 rank、将 sequence 重新组织成规则 H×W feature、减少 permute。

### dtype / quantization

先 float 跑通，再处理 FP16 / W8A16 / W8A8。

## 10. 为什么先 float 再量化

推荐：

```text
PyTorch float
→ export float
→ QNN float DLC
→ 数值对齐
→ HTP 跑通
→ 再量化
```

否则算子改写、动态 shape、量化精度会混在一起。

## 11. DLC 出来后回到当前项目的熟悉链路

```text
UniAD DLC
→ qnn-net-run + libQnnModelDlc.so
→ Compose / Finalize
→ Execute
→ Context Binary
→ C++ runner
→ profiling / output compare
```

所以当前 RF-DETR 工作就是后续 UniAD 的执行底座。

## 12. 推荐 UniAD 迁移顺序

```text
1. CNN backbone/neck
2. MLP/Linear/Norm/FFN
3. Multihead Attention
4. Deformable Attention 最小单元
5. Spatial Cross Attention
6. Temporal Self Attention / prev_bev
7. camera geometry projection
8. detection decoder
9. tracking/map/motion/planning heads
10. 整图拼接
```

## 13. 最小验收标准

```text
[ ] 固定 input contract
[ ] 原始 PyTorch 可跑
[ ] 改写 PyTorch 可跑
[ ] 原版 vs 改写数值通过
[ ] export 成功
[ ] export runtime vs PyTorch 通过
[ ] QNN DLC 生成成功
[ ] 手机 HTP Finalize 成功
[ ] HTP Execute 成功
[ ] QNN vs PyTorch 数值通过
[ ] profiling 可读
```

少一项，只能说某一步成功，不能说“该算子已支持 QNN”。