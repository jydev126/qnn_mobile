# QNN Profiling：INIT、Finalize、Execute、Accelerator Time 怎么看

这篇文档解决：**QNN 日志里不同时间到底代表什么，怎么避免把进程总耗时、Finalize 时间和 HTP Execute 时间混为一谈。**

## 1. 先建立四层时间口径

看任何 QNN 性能数据前，先问它属于哪一层：

```text
A. 应用/进程 wall time
B. QNN runtime/API wall time
C. graph prepare/finalize 时间
D. accelerator execute 时间
```

这四个数可能差很多。

例如：

```text
adb shell 启动进程
  + ELF/dlopen
  + backend/device init
  + 读取 DLC/context
  + graph prepare/restore
  + 读 input.raw
  + Execute
  + 写 output.raw
```

全部加起来才接近“命令从开始到结束”的 wall time。

所以：

```text
进程耗时 != HTP 推理耗时
```

## 2. INIT 应该怎么理解

Qualcomm 工具里的 `INIT` 是工具定义的一段初始化阶段。

在 DLC 现场 prepare 路径中，它可能覆盖：

```text
模型加载
Compose
部分 backend 初始化
Finalize/prepare 相关工作
其他工具初始化
```

因此不能看到：

```text
INIT = 100 ms
Finalize = 80 ms
```

就机械地认为：

```text
总准备时间 = 180 ms
```

两者可能存在包含关系或统计口径差异。

正确做法是：

> 按工具/版本的 profiling event 语义读，不要把所有看起来像阶段时间的数直接相加。

## 3. Compose 是什么时间

Compose 是模型 loader 创建 QNN graph 的过程：

```text
创建 tensor
创建 node
连接 graph
设置参数
```

它不是一个统一的：

```text
QnnGraph_compose()
```

所以 profiling 中不一定存在一个名字就叫 Compose 的单独事件。

有时 Compose 成本会被包含在模型加载或 INIT 范围内。

这也是为什么要区分：

```text
概念阶段
vs
profiling event 名字
```

## 4. Finalize 是什么时间

`QnnGraph_finalize()` 是明确 API 边界。

对 HTP backend，它通常意味着：

```text
验证 graph
后端优化
目标相关 lowering/prepare
memory planning
生成可执行状态
```

因此 Finalize 往往是 DLC 首次加载路径里的重成本阶段。

当前项目把它从每次启动路径移出的办法就是：

```text
DLC
→ Compose
→ Finalize
→ serialize Context Binary
```

以后运行时恢复 context，而不是重新 Finalize。

## 5. `Accelerator finalize time` 和 `QNN time`

如果 profiling 同时出现类似：

```text
Finalize / QNN time
Accelerator finalize time
```

不要默认二者是两段串行、可以相加的互斥时间。

更合理的理解是：

```text
QNN event
  └── 里面可能包含 accelerator/backend 子事件
```

是否包含、怎样嵌套，要以 profiler event 层级为准。

因此性能汇报优先写：

```text
Finalize wall/QNN event: X
其中 accelerator event: Y
```

而不是：

```text
Finalize 总计 = X + Y
```

## 6. Execute 到底是哪一个 Execute

模型跑 6 次时要区分：

```text
Execute #0
Execute #1
Execute #2
...
```

第一次可能包含更多一次性成本：

```text
lazy initialization
cache warm-up
resource activation
first-use setup
```

所以至少看：

```text
first execute
steady-state median/mean
min/max
```

不要只报一次。

当前项目默认多次 execute，就是为了把：

```text
启动/准备成本
```

和：

```text
稳态单次推理成本
```

拆开。

## 7. QNN Execute 时间和 Accelerator 时间

一次 `QnnGraph_execute()` 的 CPU wall time可能包括：

```text
提交工作
runtime orchestration
memory synchronization
等待 HTP 完成
返回结果
```

accelerator event 更接近：

```text
目标计算资源真正执行 graph 的时间
```

二者相关但不必相等。

因此建议同时保留：

```text
API wall time
accelerator time
```

不要只留其中一个。

## 8. `qnn-net-run` 和自己的 C++ runner 为什么不能直接用总时间比较

两个程序做的外围工作不同。

`qnn-net-run` 可能额外包含：

```text
命令行解析
input_list 解析
目录创建
通用 tensor handling
profiling setup
raw 文件写入
日志
```

自己的 C++ runner 可以非常薄：

```text
restore
buffer
execute
write
```

所以：

```text
qnn-net-run process wall time
vs
qnn-context-runner process wall time
```

不能直接说明：

> 自己写 C++ 后 HTP 快了多少。

真正有意义的是比较同一层级的：

```text
graph restore
first execute
steady execute
accelerator execute
```

## 9. Context Binary 应该观察哪几个时间

DLC 路径：

```text
DLC load
Compose
Finalize
Execute x N
```

Context 路径：

```text
read binary
context restore/deserialization
graph retrieve
Execute x N
```

所以 Context Binary 的收益主要应该看：

```text
DLC init/finalize
vs
context restore
```

而不是期待：

```text
Execute x N
```

一定明显变快。

因为二者最终可能执行相同或等价的 prepared graph。

## 10. Host offline prepare 和 device prepare 的比较

如果以后把 context binary 在 PC/host 生成，还要额外区分：

```text
生成 binary 的机器/target config
运行 binary 的 SoC
SDK 版本
HTP 架构
```

host 生成更快，不等于运行时 graph 更快；它主要改变“准备工作发生在哪里”。

## 11. Profiling 的推荐输出格式

每次实验最好同时记录：

```text
device
SoC
HTP arch
QAIRT/QNN version
model hash
context hash
input hash
command
backend
profiling level
```

性能表建议：

| metric | value |
| --- | ---: |
| process wall | ... ms |
| context/DLC init | ... ms |
| finalize | ... ms |
| first execute API wall | ... ms |
| steady execute median API wall | ... ms |
| accelerator execute median | ... ms |
| output write | ... ms |

## 12. 不要用平均值隐藏 first-run

例如：

```text
[18, 8, 8, 8, 8, 8] ms
```

如果只报：

```text
平均 9.7 ms
```

信息损失很大。

更好的：

```text
first: 18 ms
steady median: 8 ms
```

对于手机产品启动延迟和持续帧率，这是两个完全不同的问题。

## 13. 输出一致性要和性能一起看

性能实验必须同时确认：

```text
DLC output
Context output
C++ runner output
```

仍然一致。

否则：

```text
某条路径更快
```

可能只是输入 dtype、输出 dtype、执行次数、graph 或 context 已经不一致。

所以当前项目将：

```text
command.txt
sha256.txt
execution metadata
profiling
Result_N
```

一起保存是正确方向。

## 14. UniAD 之后应该怎么 profile

UniAD 不要只测“整模型 FPS”。

一开始应按风险模块 profile：

```text
backbone
neck
Temporal Self Attention
Spatial Cross Attention
MSDeformableAttention3D
Decoder
heads
```

对每个改写版本同时记录：

```text
PyTorch CPU/GPU 只做数值基线
QNN converter 是否成功
HTP Finalize 时间
HTP Execute 时间
峰值内存
输出误差
```

尤其是特殊算子改写，目标不是：

> 能编译就结束。

而是：

> 能编译 + 数值对 + HTP 性能合理。

## 15. 最后记住一句

任何性能数字都要带“口径”说。

不要说：

```text
QNN 推理 10 ms
```

而说：

```text
HTP accelerator execute median 10 ms
```

或者：

```text
QnnGraph_execute API wall median 12 ms
```

或者：

```text
context restore + first execute 40 ms
```

这样后续和 TensorRT、CPU、GPU、不同 runner 比较才不会混乱。