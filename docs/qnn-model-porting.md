# 从 PyTorch 开源模型到 QNN DLC：模型迁移方法

这篇文档解决：**拿到一个新的 PyTorch 开源模型后，怎么判断它能不能进 QNN，怎么一步步变成 DLC，而不是直接拿整模型碰运气。**

## 1. 先明确当前项目已经解决了什么

当前 `qnn_mobile` 已经验证的是 DLC 之后的链路：

```text
DLC
→ Compose
→ Finalize
→ Context Binary
→ qnn-net-run Execute
→ 自己的 C++ runner Execute
```

对于 UniAD 这类新模型，真正新增的难点在前半段：

```text
PyTorch source + checkpoint
        ↓
可静态导出的 PyTorch graph
        ↓
ONNX / Qualcomm 接受的中间图
        ↓
QNN converter/compiler 能接受的算子与 shape
        ↓
DLC
```

所以“会跑 DLC”和“会把模型做成 DLC”是两件事。

## 2. 不要第一步就导整模型

大模型迁移最差的方式：

```text
整模型
→ export
→ 报 200 个错误
→ 随便改
→ 再报 180 个错误
```

更可控的方式是先画模块边界：

```text
Input preprocessing
Backbone
Neck
BEV encoder
  ├── Temporal Self Attention
  └── Spatial Cross Attention
Decoder
Heads
Postprocess
```

然后给每块标：

```text
A. 普通静态算子
B. 有 Qualcomm 成功参考
C. 特殊算子，需要改写
D. 动态控制流/动态 shape，高风险
E. 可以放在 CPU/模型外
```

## 3. 第一次静态审计看什么

### 3.1 Python 控制流

重点找：

```python
if tensor.any():
for n in tensor_shape_from_runtime:
.nonzero()
.tolist()
.item()
```

如果这些结果参与：

```text
分支
循环次数
reshape size
slice bound
```

export 很容易出现 data-dependent graph 问题。

### 3.2 动态 shape

重点找：

```text
动态 batch
动态 camera 数
动态 BEV H/W
动态 feature level 数
动态 query 数
动态 top-k
根据 mask 产生变长 tensor
```

手机部署优先固定：

```text
batch = 1
camera count = 常量
input H/W = 常量
BEV H/W = 常量
num_query = 常量
num_levels = 常量
num_points = 常量
```

静态 shape 不是“低级 workaround”，而是移动端编译型部署最重要的工程约束之一。

### 3.3 自定义 CUDA/MMCV op

看到：

```text
torch.autograd.Function
mmcv.ops
ext_loader.load_ext
CUDA extension
C++ extension
```

先标红。

它们在 PyTorch GPU 上能跑，不代表 converter 知道这个 op 的数学含义。

优先问：

> 能否用标准 PyTorch/ONNX primitives 重写？

而不是直接问：

> 怎么给 QNN 写 custom op？

Qualcomm BEVFormer 的 deformable attention 就是典型案例：原来的 MMCV deformable-attention extension 被改成更容易导出的普通 tensor 运算和 `grid_sample` 路径。

## 4. 建立“最小可导出单元”

一个需要迁移的模块最好包成：

```python
class ExportUnit(nn.Module):
    def forward(self, input_0, input_1, ...):
        return output_0, output_1, ...
```

然后固定所有输入 contract：

```text
name
shape
dtype
layout
value range
哪些维度允许动态
```

例如 deformable attention 不要一开始塞整个 BEVFormer，而是先验证：

```text
query
value
reference_points
spatial_shapes
mask
        ↓
MSDeformableAttention3D
        ↓
output
```

## 5. 每个单元必须有三套数值基线

不要只验证“能 export”。

至少保留：

```text
A. 原始 PyTorch 模块输出
B. 改写后 PyTorch 模块输出
C. 导出图 / QNN 输出
```

比较：

```text
max abs error
mean abs error
cosine similarity / PSNR（按场景）
关键任务输出
```

正确顺序是：

```text
原实现 vs 改写实现
        先相等

改写实现 vs 导出运行
        再相等

导出运行 vs QNN
        最后相等
```

否则一旦 QNN 输出错，不知道错误来自模型改写、export 还是 backend。

## 6. PyTorch 改写时的原则

### 原则 A：保留数学含义，换表达形式

例如：

```text
nn.Linear over spatial positions
```

可以在特定 layout 下改成：

```text
1x1 Conv2d
```

只要权重映射正确，数学上仍是每个空间位置独立的线性变换。

### 原则 B：优先静态 tensor 变换

更偏好：

```text
reshape
permute
transpose
concat
split（静态长度）
matmul
conv
softmax
add/mul
```

警惕：

```text
nonzero 生成变长索引
数据相关循环
数据相关 reshape
Python list ← Tensor values
```

### 原则 C：把纯后处理移出图

例如：

```text
字符串类别名
可视化
复杂 Python NMS 包装
轨迹结构组装
```

如果不影响加速器主干执行，可以放 CPU 侧。

但不能随意把主干中的 attention、geometry projection 等核心算子移出去，否则整体性能和接口会完全改变。

## 7. ONNX/export 阶段应该记录什么

每次 export 产物都记录：

```text
源 checkpoint hash
代码 commit
input spec
opset / torch version
是否做过 patch
输出 tensor name / shape / dtype
```

然后检查 graph：

```text
有没有自定义 domain op
有没有 ATen fallback
有没有意外动态维度
有没有巨大 Expand/Tile
有没有 Shape/Gather 参与数据相关控制
```

一个图“成功生成 .onnx”不表示它适合 QNN。

## 8. QNN 转换失败时怎么分类

不要只保存最后一行 error。

把问题归入四类：

### A. Operator unsupported

例如 converter 不认识某个 op。

处理：

```text
寻找等价标准算子改写
参考 Qualcomm model patch
必要时再研究 custom op package
```

### B. Shape inference / dynamic shape

处理：

```text
固定输入 shape
去掉数据相关 reshape
将常量 shape 提前变成 Python/static constant
拆模块验证
```

### C. Layout / rank / backend constraint

处理：

```text
改 NCHW/NHWC
降低 tensor rank
把 sequence 重新解释成 HxW feature
减少不必要 permute
```

### D. 数值支持 / dtype / quantization

处理：

```text
先 float 跑通
再处理 FP16 / W8A16 / W8A8
检查敏感算子
```

## 9. 为什么先 float，再量化

第一次迁模型时同时处理：

```text
算子改写
动态 shape
量化
精度
```

会把问题混在一起。

建议：

```text
PyTorch float
→ export float
→ QNN float DLC
→ 数值对齐
→ HTP 跑通
→ 再量化
```

如果 float 都没对齐，先不要讨论 INT8 精度损失。

## 10. DLC 生成后才回到当前 qnn_mobile 熟悉的链路

一旦 UniAD 某个切片或完整模型拿到了 DLC：

```text
UniAD DLC
   ↓
qnn-net-run + libQnnModelDlc.so
   ↓
Compose
   ↓
Finalize
   ↓
Execute
```

然后可以复用当前仓库已经验证过的方法：

```text
生成 context binary
恢复 context
C++ runner
profiling
输出比较
```

所以当前项目不是旁支，而是 UniAD DLC 生成后的执行底座。

## 11. 推荐的 UniAD 迁移顺序

不要按源码文件顺序，而按风险排序：

```text
1. 普通 CNN backbone/neck
2. 简单 MLP / Linear / Norm / FFN
3. Multihead Attention
4. Deformable Attention 单元
5. BEV Spatial Cross Attention
6. Temporal Self Attention / prev_bev
7. geometry / camera projection
8. detection decoder
9. tracking / map / motion / planning heads
10. 整图拼接
```

原因：先知道最难算子能否被改写，再决定整模型架构边界。

## 12. 最小验收标准

一个模块只有同时满足下面条件才算“迁移完成”：

```text
[ ] 固定输入 contract
[ ] PyTorch 原版可跑
[ ] 改写版 PyTorch 可跑
[ ] 原版 vs 改写版数值通过
[ ] export 成功
[ ] 导出运行 vs PyTorch 通过
[ ] QNN DLC 生成成功
[ ] 目标手机 Finalize 成功
[ ] HTP Execute 成功
[ ] QNN vs PyTorch 数值通过
[ ] profiling 可读
```

少一个，都只能说“某一步成功”，不能说“这个算子已支持 QNN”。