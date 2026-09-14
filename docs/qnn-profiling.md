# QNN Profiling：拿当前 RF-DETR 的真实 10 秒 Finalize 和 160 ms Execute 讲清时间口径

这篇文档不再抽象解释 `INIT / Finalize / Execute`。直接用当前 Samsung Galaxy S25 Ultra + RF-DETR 的三条实测路径说明：**你看到的每一个 ms 到底包了什么，哪些数字可以横向比较，哪些绝对不能相加。**

当前固定实验环境：

```text
Device     Samsung Galaxy S25 Ultra / SM-S9380
SoC        Snapdragon 8 Elite / HTP V79
QAIRT      v2.45.0.260326154327
Backend    libQnnHtp.so
Runs       每条路径 6 次，相同 input/context
```

三条路径的 54 组输出 tensor 已逐元素一致，最大绝对差 0。所以下面的性能比较至少是在同一结果前提下进行。

---

## 1. 先看项目里真正测到的数字

单位 ms：

| 阶段 | DLC + qnn-net-run | Context + qnn-net-run | Context + C++ |
| --- | ---: | ---: | ---: |
| Compose | 418.095 | 0 | 未调用 |
| Finalize | 9,670.098 | 无此事件 | 未调用 |
| net-run INIT / C++ `contextCreateFromBinary` | 10,088.341 | 57.937 | 60.419 |
| Execute 0 | 156.013 | 156.818 | 153.594 |
| Execute 1–5 平均 | 162.344 | 167.215 | 162.969 |
| Execute 1–5 范围 | 154.844–172.729 | 156.660–174.584 | 154.157–168.317 |

只看这张表就能得到三个重要结论：

```text
1. 当前 DLC online prepare 的启动成本主要被 Finalize 吃掉，约 9.7 秒。
2. 预生成 Context Binary 后，恢复 context 是约 60 ms 量级，不再做 Finalize。
3. 三条路径 Execute 都在约 150~175 ms；当前实验不支持“Context/C++ 让稳态推理更快”。
```

这比一句“context binary 更快”准确很多。

---

## 2. 第一条原则：不同层级的时间不能混成一个“推理时间”

至少分四层：

```text
A. 整个 process / command wall time

B. 某个 CPU 侧 QNN API wall time
   例如 contextCreateFromBinary / graphExecute

C. graph build/prepare/finalize event

D. backend/accelerator profiler 中的 target execution event
```

它们可能有包含关系：

```text
process wall
┌───────────────────────────────────────────┐
│ file I/O                                  │
│ dlopen                                    │
│ backend/device init                       │
│    ┌──────── QNN API wall ─────────────┐  │
│    │ submit / RPC / wait / sync        │  │
│    │   ┌── accelerator work ────────┐  │  │
│    │   └────────────────────────────┘  │  │
│    └───────────────────────────────────┘  │
│ output I/O                                │
└───────────────────────────────────────────┘
```

所以永远不要：

```text
process wall = accelerator time         ×
QnnGraph_execute wall = pure HMX time   ×
```

---

## 3. 当前 C++ runner 到底测了哪些边界

`cpp/qnn_context_runner.cpp` 自己把主流程拆成：

```text
load_libraries
backend_device_create
read_context_file
read_metadata
context_create_from_binary
graph_retrieve
read_inputs
execute_0 ... execute_N
write_outputs_0 ... write_outputs_N
free_resources
```

真实代码就是：

```cpp
timings.measure("context_create_from_binary", [&] {
    contextCreateFromBinary(...);
});

for (...) {
    timings.measure("execute_i", [&] {
        graphExecute(...);
    });
}
```

因此 `timings.csv` 的一个好处是：**它明确告诉你计时器包住的是哪一个 CPU API，而不是一个模糊 INIT。**

---

## 4. `execute_i` 为什么仍然不是“HTP 内部 kernel 时间”

源码计时注释已经写明：

```text
execute 的区间不包含文件读写。
它是 CPU 侧同步 API wall time（含 RPC 等），并不等于 HTP 内部 kernel 时间。
```

`graphExecute()` 可以概念理解：

```text
ARM CPU
start timer
  │
  ├── QNN runtime orchestration
  ├── input/output synchronization / memory handling
  ├── RPC / driver submit
  │          ↓
  │       HTP execution
  │          ↓
  ├── wait/sync/return
  ▼
stop timer
```

所以当前：

```text
execute_3 = 160 ms
```

最严谨的汇报是：

> `QnnGraph_execute` synchronous API wall time = 160 ms。

不是：

> HTP kernel = 160 ms。

要说 accelerator time，需要 backend profiling 里的目标执行 event。

---

## 5. DLC 路径里的 `INIT` 为什么不能再加 Compose + Finalize

当前实测：

```text
Compose       418.095 ms
Finalize    9,670.098 ms
INIT       10,088.341 ms
```

注意：

```text
418.095 + 9,670.098
≈ 10,088.193 ms
```

它几乎已经等于 INIT。

这正好说明：

> **当前工具的 INIT 不是和 Compose/Finalize 平级、互斥的第三段时间；它基本包含了它们。**

所以错误汇报：

```text
模型准备 = INIT + Compose + Finalize
        ≈ 20 秒                         ×
```

正确理解：

```text
INIT ≈ 整体初始化/prepare 区间
其中 profiler 又拆出了 Compose / Finalize 等子事件
```

看 profiler 一定先确认 event hierarchy，不要见一行就求和。

---

## 6. Finalize 在这个模型上为什么是最重要的启动成本

DLC online prepare：

```text
Compose     ~0.42 s
Finalize    ~9.67 s
Execute     ~0.16 s / 次
```

也就是说一次只推 1 帧时：

```text
prepare 成本 >> 单次 inference
```

这就是 context binary 的工程价值。

Finalize 对 HTP 可以涉及目标相关的：

```text
graph validation
lowering / optimization
Hexagon compiler / target prepare
memory planning
execution setup
```

当前应用不需要知道内部每个 pass，但必须认识到它不是普通的“API 初始化几十毫秒”。

---

## 7. Context Binary 真正省掉了什么

### Online DLC

```text
DLC read
→ Compose
→ Finalize                 ~10 秒级
→ Execute × N
```

### Context restore

```text
.bin read
→ contextCreateFromBinary  ~60 ms
→ graphRetrieve
→ Execute × N
```

所以更准确的收益表达：

> **Context Binary 把目标相关的 Compose/Finalize 从产品运行阶段搬到了 build/prepare 阶段。**

不是：

> Context Binary 把每次 HTP inference 从 160 ms 变成 60 ms。

60 ms 是 restore，不是 Execute。

---

## 8. Context build 自己也不便宜，而且不同进程 Finalize 会波动

生成 context 是另一个进程，本次实测：

```text
Compose           456.462 ms
Finalize       15,462.258 ms
getBinarySize       0.338 ms
getBinary           78.391 ms
```

注意另一次 DLC execute 进程的 Finalize 是：

```text
9,670.098 ms
```

同一模型 Finalize 已有明显波动。

所以不要拿一次：

```text
Finalize = 9.67 s
```

写成芯片固定 benchmark。

它可能受：

```text
温度
频率
进程状态
系统负载
backend/compiler path
profiling overhead
```

影响。

当前实验的作用是看数量级和阶段关系，不是给 SM8750 定一个官方 Finalize 延迟。

---

## 9. C++ context restore 的 60 ms 也不是完整启动时间

本次 C++ 还单独测到：

```text
load libraries                3.855 ms
backend/device create        93.599 ms
read context file            47.470 ms
contextCreateFromBinary      60.419 ms
read input                    2.821 ms
```

所以如果有人问：

> “Context 模型启动只要 60 ms 吗？”

答案是：

**不能这么说。**

60.419 ms 只对应：

```text
contextCreateFromBinary API
```

完整产品 cold start 还需要：

```text
process start
dlopen
backend/device
.bin I/O
metadata
graph retrieve
input setup
第一次 execute
```

要比较 cold start，就必须定义共同边界。

---

## 10. `qnn-net-run INIT` 和 C++ `contextCreateFromBinary` 也不能强行 1:1

表里：

```text
context + net-run INIT = 57.937 ms
C++ contextCreate      = 60.419 ms
```

它们数值很近，但文档仍然不把它们定义成同一个 API 边界。

原因：

```text
qnn-net-run INIT
```

是 Qualcomm 工具自己定义的 profiling 区间；

```text
contextCreateFromBinary
```

是我们 C++ timer 精确包住的一个 API。

可以说它们“本次数量级接近”，不要说“两个指标完全等价”。

---

## 11. 第 0 次 Execute 也不能自动叫 warm-up

本次：

```text
DLC      execute0 156.013 ms
Context  execute0 156.818 ms
C++      execute0 153.594 ms
```

后 5 次：

```text
约 154~175 ms
```

这次第 0 次甚至没有明显更慢。

所以当前源码注释特意写：

```text
第 0 次单独看，后续也不能武断称稳态
```

正确实验语言是：

```text
first execute
subsequent executes
```

等样本量、温度/频率控制、预热策略足够以后，再使用：

```text
steady state
```

这个词。

---

## 12. 当前 6 次执行能说明什么，不能说明什么

可以说明：

```text
DLC/context/C++ Execute 在相同量级
输出完全一致
context 没有改变模型数学结果
C++ runtime 没有明显多出一个数量级开销
```

不能说明：

```text
平均 162.969 ms 就是芯片最终性能
已经排除 thermal throttling
已经找到最佳 power config
已经得到 HTP-only compute time
UniAD 会有相同 overhead 比例
```

当前没有：

```text
严格频率锁定
温控实验
大量重复样本
backend deep profiling 全量分析
```

---

## 13. 为什么 qnn-net-run 和自己的 C++ 不应该比整个 process wall

`qnn-net-run` 是通用工具，它还做：

```text
CLI parse
input_list parse
generic tensor handling
profiling setup
输出目录/文件
更多日志
```

自己的 C++ 是薄 runtime。

如果：

```text
C++ process = 500 ms
qnn-net-run process = 700 ms
```

不能推出：

```text
C++ 让 HTP model 快了 200 ms
```

正确比较相同层级：

```text
context restore API
QNN Execute API wall
accelerator profile
output tensor
```

---

## 14. Profiling 文件在当前项目分别回答什么

每次实验目录里：

```text
command.txt
    到底执行了什么命令

environment.txt / sha256.txt
    哪个 runtime/model/input/context

run.log
    完整 stdout/stderr

profile.txt / profile.csv
    Qualcomm profiler 解出的 QNN/backend events

qnn-profiling-data.log
    原始 profiling 数据

timings.csv
    自写 C++ 的 CPU API wall timer

tensors.tsv
    自写 C++ 看到的 graph I/O contract

Result_0 ... Result_N
    实际输出
```

性能结论必须能追溯到其中一个具体文件和具体边界。

---

## 15. 三种常见的错误性能句子

### 错误 1

```text
RF-DETR NPU 推理 160 ms
```

问题：不知道这是 API wall 还是 accelerator。

改成：

```text
RF-DETR `QnnGraph_execute` synchronous API wall 约 154~175 ms（本次 6-run 实验）。
```

### 错误 2

```text
Context Binary 把推理从 10 秒降到 60 ms
```

问题：把 prepare/restore/execute 混了。

改成：

```text
Context Binary 把约 10 秒的 DLC online Compose/Finalize 移到预生成阶段；运行时 context restore 约 60 ms，单次 Execute 仍约 160 ms 量级。
```

### 错误 3

```text
Finalize = INIT + 9.7 秒
```

问题：INIT 已包含 Finalize。

改成：

```text
本次 DLC INIT 约 10.09 s，其中 profiler 拆出 Compose ~0.42 s、Finalize ~9.67 s。
```

---

## 16. 做 UniAD 算子 standalone 时怎么 profile

不要一开始只看整车模型 FPS。

每个风险模块记录：

```text
Original PyTorch latency（参考）
Patched PyTorch latency
QNN compile/finalize 是否成功
Context size
context restore API wall
execute first
execute subsequent median/min/max
accelerator event（如果 profiler 能读）
logical I/O bytes
peak memory（如果工具可得）
numeric error
```

例如：

```text
MSDeformableAttention3D-4level
--------------------------------
input contract       ...
DLC                  PASS
Finalize             850 ms
context size         12 MiB
context restore       8 ms
execute API median    2.4 ms
accelerator median    1.8 ms
max abs error         2e-5
```

只有这种卡片以后才能比较：

```text
BEVFormer-style single-level
vs
Mask2Former-style multi-level
vs
自己的 UniAD rewrite
```

---

## 17. UniAD 最值得单独测的性能热点

按当前算子风险和 tensor 规模：

```text
DCNv2 rewrite
4-level MSDeformableAttention3D
SpatialCrossAttention camera rebatch/scatter
TemporalSelfAttention + prev_bev
BEV 200×200×256 大 tensor 的 layout conversion
tracking/memory state copy
```

尤其要警惕一种情况：

```text
某改写让 QNN 成功编译
但引入大量 Expand / Tile / Transpose / Scatter
```

功能 PASS 不代表部署方案好。

所以算子迁移最终验收一定是：

```text
compile
+ numeric
+ latency
+ memory
```

四项一起看。

---

## 18. 性能结果推荐统一成这张表

| Metric | 含义 | 当前 RF-DETR 示例 |
| --- | --- | ---: |
| Compose | model → QNN graph creation | 418.095 ms |
| Finalize | HTP target prepare | 9,670.098 ms |
| Context restore API | `contextCreateFromBinary` CPU wall | 60.419 ms |
| Execute first API | `graphExecute` CPU wall | 153.594 ms |
| Execute subsequent avg API | runs 1–5 | 162.969 ms |
| Execute range | runs 1–5 | 154.157–168.317 ms |
| Accelerator execute | profiler target event | 单独记录，不用 API wall 冒充 |
| Output I/O | raw 写盘 | 单独 timer |

表头里直接写 `API wall` / `accelerator`，不要都叫 latency。

---

## 19. 最后只记三个性能结论

```text
Finalize ≠ Execute
API wall ≠ accelerator-only time
Context restore ≠ inference
```

当前 RF-DETR 最有价值的实测不是“160 ms”这个孤立数字，而是我们已经把：

```text
模型准备
context 恢复
API execute
文件 I/O
输出一致性
```

分成了可复现的独立证据。

UniAD 后面所有算子实验都应该沿用这一套口径。

---

## 参考入口

- 当前实测：`report/lifecycle-results.md`
- 当前 C++ timer：`cpp/qnn_context_runner.cpp`
- 生命周期教程：`docs/lifecycle-guide.md`
- Qualcomm AI Hub profiling examples: https://dev.aihub.qualcomm.com/docs/hub/profile_examples.html
