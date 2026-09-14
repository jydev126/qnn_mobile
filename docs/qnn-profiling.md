# QNN Profiling：INIT、Finalize、Execute、Accelerator Time 怎么看

这篇文档解决：**QNN 日志里不同时间代表什么，怎么避免把进程总耗时、Finalize 和 HTP Execute 混为一谈。**

## 1. 先建立四层时间口径

```text
A. 应用/进程 wall time
B. QNN runtime/API wall time
C. graph prepare/finalize time
D. accelerator execute time
```

一次命令可能包含：

```text
进程启动 + dlopen + backend/device init + 模型/context 读取
+ graph prepare/restore + input I/O + Execute + output I/O
```

所以：

```text
进程耗时 != HTP 推理耗时
```

## 2. INIT

Qualcomm 工具的 `INIT` 是工具定义的一段初始化阶段。DLC 现场 prepare 路径中它可能覆盖模型加载、Compose、部分 backend 初始化、Finalize/prepare 相关工作等。

因此不要把 `INIT`、`Finalize` 等所有 event 机械相加；它们可能有包含/嵌套关系。要按具体 profiler event 层级和工具版本解释。

## 3. Compose

Compose 是 loader 创建 tensor/node/graph 并连接它们的过程，不是一个统一的 `QnnGraph_compose()` API。因此 profiling 不一定存在一个名字就叫 Compose 的独立 event，它可能包含在模型加载或 INIT 中。

## 4. Finalize

`QnnGraph_finalize()` 是明确 API 边界。对 HTP backend，它通常涉及：

```text
graph validation
backend optimization/lowering
目标相关 prepare
memory planning
生成可执行状态
```

Context Binary 的主要意义就是把这段准备从每次应用启动路径中移走。

## 5. `Accelerator finalize time` 与 `QNN time`

如果 profiler 同时给出 QNN finalize event 和 accelerator 子 event，不要默认二者是互斥串行时间并直接相加。更安全的汇报是：

```text
Finalize QNN event: X
其中 accelerator event: Y
```

而不是 `X + Y`。

## 6. 第一次 Execute 与稳态 Execute

第一次可能包含 lazy initialization、cache/resource first-use 成本。因此多次执行时至少记录：

```text
first execute
steady-state median/mean
min/max
```

例如 `[18,8,8,8,8,8] ms`，比“平均 9.7 ms”更有信息的是：

```text
first = 18 ms
steady median = 8 ms
```

## 7. QNN Execute API wall 与 Accelerator time

一次 `QnnGraph_execute()` CPU wall time可能包含工作提交、runtime orchestration、memory synchronization、等待目标完成等；accelerator event 更接近目标计算资源执行 graph 的时间。

所以建议同时保留：

```text
QnnGraph_execute API wall
accelerator execute
```

## 8. 为什么不能用 qnn-net-run 和自己 C++ 的进程总时间判断 HTP 谁更快

`qnn-net-run` 还做参数解析、input_list、通用 tensor handling、profiling、目录和 raw 文件 I/O；自己的 runner 可以更薄。

因此：

```text
两个 executable 的 process wall
```

主要说明应用包装成本，不直接说明 HTP graph 变快。

应该比较同层级：

```text
context restore
first execute
steady execute
accelerator execute
```

## 9. Context Binary 应该比较什么

DLC 路径：

```text
DLC load → Compose → Finalize → Execute x N
```

Context 路径：

```text
read binary → restore/deserialization → graph retrieve → Execute x N
```

因此 context 的主要收益看：

```text
DLC init/finalize vs context restore
```

不要预设稳态 Execute 一定明显变快。

## 10. 推荐记录字段

每次实验至少记录：

```text
device / SoC / HTP arch
QAIRT/QNN version
model/context/input hash
command
backend
profiling level
```

性能表：

| metric | value |
| --- | ---: |
| process wall | ... ms |
| context/DLC init | ... ms |
| finalize | ... ms |
| first execute API wall | ... ms |
| steady execute median API wall | ... ms |
| accelerator execute median | ... ms |
| output I/O | ... ms |

## 11. 性能必须和输出一致性一起看

DLC、Context、C++ runner 的 output 必须仍然一致。否则某条路径“更快”可能只是 input/output dtype、执行次数或 context 已经不同。

因此当前项目保留 `command.txt`、hash、execution metadata、profiling 和 Result_N 是正确方向。

## 12. UniAD 以后怎么 profile

一开始不要只测整模型 FPS。按风险模块看：

```text
backbone / neck
Temporal Self Attention
Spatial Cross Attention
MSDeformableAttention3D
Decoder
各 task head
```

每个改写版本同时记录：

```text
converter 是否成功
HTP Finalize
HTP Execute
峰值内存
输出误差
```

特殊算子的目标是：**能编译 + 数值对 + HTP 性能合理**。

## 13. 汇报性能时一定带口径

不要只说：

```text
QNN 推理 10 ms
```

而要说：

```text
HTP accelerator execute median 10 ms
```

或：

```text
QnnGraph_execute API wall median 12 ms
```

或：

```text
context restore + first execute 40 ms
```

这样后续和 TensorRT、CPU/GPU、不同 runner 的比较才有意义。