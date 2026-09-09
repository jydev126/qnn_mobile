# RF-DETR：DLC、Context Binary 与自写 C++ 的设备验证

2026-09-09。三条路径均在同一台 Samsung Galaxy S25 Ultra 上成功执行。DLC online prepare、context + qnn-net-run、context + 自写 C++ 各 6 次，boxes / logits / classes 共 54 组张量检查全部逐元素一致，最大绝对差为 0。自写程序也通过正常退出及三种失败路径检查。

这是同一模型资产、同一输入的 runtime 一致性验证；不是 PyTorch 精度对齐、mAP 测试或 UniAD 成功部署。

## 环境与产物

| 项目 | 实测 |
| --- | --- |
| Host / Python | Fedora Linux x86_64 / 项目 venv Python 3.11.16 |
| QAIRT build | v2.45.0.260326154327 |
| 手机 | SM-S9380，board sun，Android 16 / API 36，SELinux Enforcing |
| backend / DSP | libQnnHtp.so，V79 Stub/Skel |
| 编译器 | 本机 NDK 28.2.13676358 / Clang 19.0.1，aarch64-linux-android28 |
| 输入 | RGB NCHW float32 [1,3,512,512]，3,145,728 bytes |
| context binary | 61,456,384 bytes，在目标手机预生成 |
| executable | artifacts/cpp/qnn-context-runner，Android ARM64 PIE，静态链接 C++ 标准库，动态加载 QNN |

配置使用 `RF_DETR_DLC`、`QAIRT_ROOT`、`ANDROID_NDK_ROOT`，实际机器路径在被忽略的 `config/local.env`。未下载依赖。源码、脚本、讲解可以提交；SDK、DLC、binary、executable、raw、profiling 与图片均在忽略目录。

## 时间揭示了什么

单位 ms，以下固定引用本次结果目录，不是实时更新的性能承诺。

| 阶段 | DLC + net-run | 恢复 context + net-run | 恢复 context + C++ |
| --- | ---: | ---: | ---: |
| Compose | 418.095 | 0 | 未调用 |
| Finalize | 9,670.098 | 未记录此事件 | 未调用 |
| net-run INIT / C++ contextCreateFromBinary¹ | 10,088.341 | 57.937 | 60.419 |
| 第 0 次 Execute | 156.013 | 156.818 | 153.594 |
| 第 1–5 次 Execute 平均 | 162.344 | 167.215 | 162.969 |
| 第 1–5 次范围 | 154.844–172.729 | 156.660–174.584 | 154.157–168.317 |

¹ 三列此行的测量边界不同：DLC 的 INIT 包含其建图/Finalize 工作，不能再与前两行相加；恢复 net-run 的 INIT 接近其 load-binary API 区间，不能解读为整个进程启动。C++ 单独测量恢复 API，另外还测到加载库 3.855、backend/device create 93.599、读 binary 文件 47.470、读输入 2.821 ms。并非所有启动成本都在 context restore 一行中。

生成 context 是另一次预处理进程：Compose 456.462 ms、Finalize 15,462.258 ms、getBinarySize 0.338 ms、getBinary 78.391 ms。getBinary 不等于包含所有写盘开销的“保存文件总时间”。不同进程的 Finalize 有明显波动，不把单次数字当稳定基准。

本次证据支持：预生成把建图和 Finalize 移到 build 阶段，恢复执行时仍有初始化与加载成本；三条路径的 Execute 都在约 150–175 ms 量级。它不支持“context 或 C++ 让稳态推理更快”的结论。工具使用 verbose/basic profiling，C++ 使用 warn/无 backend profiling；频率、温度、后台负载未受控，且只有少量同图样本。

## 原始证据入口

| 实验 | 固定目录与证据 |
| --- | --- |
| DLC | [命令](../output/lifecycle/dlc.dfykEg/command.txt)、[日志](../output/lifecycle/dlc.dfykEg/run.log)、[profile](../output/lifecycle/dlc.dfykEg/profile.txt)、[CSV](../output/lifecycle/dlc.dfykEg/profile.csv)、[metadata](../output/lifecycle/dlc.dfykEg/execution_metadata.yaml) |
| 生成 context | [命令](../output/lifecycle/build-context.x2kTtq/command.txt)、[profile](../output/lifecycle/build-context.x2kTtq/profile.txt)、[binary](../output/lifecycle/build-context.x2kTtq/rf_detr.bin)、[哈希](../output/lifecycle/build-context.x2kTtq/sha256.txt) |
| net-run 恢复 | [命令](../output/lifecycle/context.tBqaMX/command.txt)、[日志](../output/lifecycle/context.tBqaMX/run.log)、[profile](../output/lifecycle/context.tBqaMX/profile.txt)、[metadata](../output/lifecycle/context.tBqaMX/execution_metadata.yaml) |
| C++ 恢复 | [命令](../output/lifecycle/cpp.Qs4jXZ/command.txt)、[完整日志](../output/lifecycle/cpp.Qs4jXZ/run.log)、[API 计时](../output/lifecycle/cpp.Qs4jXZ/timings.csv)、[tensor contract](../output/lifecycle/cpp.Qs4jXZ/tensors.tsv)、[整个进程退出码](../output/lifecycle/cpp.Qs4jXZ/exit-code.txt) |
| 验收 | [最新比较 JSON](../output/lifecycle/comparison.json)、[设备失败路径检查](../output/lifecycle/check.xhLNq8/checks.txt) |

日志和 binary 为本地忽略产物，单独 clone 仓库不会自动带上这些链接目标；按教程重跑后会得到新的随机目录。`comparison.json` 会随下一次比较更新，本报告的固定目录与表格保留本次快照。

## 做了哪些工程改动

- [lifecycle.sh](../scripts/lifecycle.sh)：四种模式对应 DLC 执行、context 生成、context 恢复、自写 C++；验证部署文件哈希、保存命令/环境、抓 profiling、拉取并核对输出，成功后更新指针。
- [qnn_context_runner.cpp](../cpp/qnn_context_runner.cpp)：单文件 runtime，主流程六段；metadata 驱动图名和 tensor buffer，同步重复执行，按 native dtype 保存，正常/异常路径均释放 QNN 资源。
- [build-cpp.sh](../scripts/build-cpp.sh)：直接调用本地 NDK，检查 ELF 架构和依赖，记录编译器与源码/二进制哈希；没有 CMake/JNI/APK。
- [compare-lifecycle.py](../scripts/compare-lifecycle.py)：核对模型/输入/runtime/context 身份、输出 metadata、每轮 raw shape/dtype/数值，并提取逐次 Execute。类别要求精确一致；float 的允许阈值是 rtol=atol=1e-4，本次实际为完全相同。
- [教程](../docs/lifecycle-guide.md)和 [C++ 导读](../cpp/README.md)：说明 DLC 生命周期、完整复现、API/句柄/buffer 的职责、实测日志读法，以及拿到新 DLC 后该改的模型专用部分。

## 发现并处理的实际问题

1. native 类别输出：QAIRT 2.45 的 net-run 写成 `classes_native.raw`，类型 int32；旧检测路径写 float。新脚本显式处理文件名与 dtype，不用字节数相同来假设类型相同。
2. profiling log 是 symlink：adb 目录拉取会提示跳过，脚本另外读取原始目标并核对 SHA256。CSV 和文本都由同版本设备端 viewer 生成，避开 host viewer 缺少 libc++.so.1 的问题。
3. C++ 卸载动态库后的退出崩溃：首次实现完成 execute/free 后 SIGSEGV，已保留 [失败日志](../output/lifecycle/cpp.SaitKX/run.log)。修正为 QNN 资源显式释放、runtime 使用 RTLD_NODELETE 保留映射至进程结束。随后两个独立进程均退出 0，错误输入路径也正常退出 1。未定位 SDK 内部具体符号，不把推断当作已证明的内部根因。

## 验证与范围

Bash 全部脚本语法、配置示例语法、git diff --check、make doctor、设备属性检查、Python 编译检查、Android `-Wall -Wextra -Werror` 交叉编译均通过。两条工具执行和最终 C++ 各 6 次输出一致；C++ 错名、错长度、损坏 context 共 3 个真实设备错误检查通过。另有 3 个 host 验收测试：原始证据通过，临时副本篡改类别被拒绝，输入身份不一致被拒绝；原始结果未改动。

尚未实现/验证：host offline prepare、动态 shape、其他 tensor dtype、多图选择、共享内存、异步执行、性能调优、匹配权重的 PyTorch/mAP、UniAD 子模块。本阶段完成后这些都作为独立后续任务，不自动扩展。
