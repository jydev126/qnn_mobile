# 当前阶段：掌握 DLC 在高通上的完整执行流程

2026-09-09 用户批准：RF-DETR → shell 工具验证 → context binary → 最小 C++ executable，并配套能逐段理解的讲解。此计划替代此前“先等 PyTorch checkpoint”的执行顺序；精度对齐仍未完成，但不阻断生命周期学习。

交付状态：第 1–4 步已实现并完成设备验证，证据见 report/lifecycle-results.md；第 5 步为后续可选课题，未自动展开。

固定环境：Fedora x86_64、Python 3.11 venv、QAIRT 2.45.0.260326、S25 Ultra / SM8750 / HTP V79，模型路径 RF_DETR_DLC。仅 Android 命令行 executable，在 /data/local/tmp 执行；不创建 APK/CMake/JNI/Android Studio 工程。

1. 保留原有 RF-DETR 预处理和检测流程，新增 DLC online prepare 生命周期实验：记录命令、环境、Compose/Finalize/首次与后续 Execute profiling。原始输出统一为 native dtype，明确与旧 float 文件不同。
2. 用设备端 qnn-context-binary-generator 读取 DLC、Compose/Finalize 并序列化 context；qnn-net-run --retrieve_context 执行相同输入。保留模型、输入、binary、设备与 SDK 的身份信息。比较两条路径输出和分阶段耗时，不预设稳态性能提升。
3. 用本机已有 Android NDK 编译最小 C++ executable：加载 QNN/HTP 和 System API、读取 metadata、恢复 context、取得图、准备 raw tensor buffer、同步重复 execute、保存 native 输出、释放资源。按 metadata 检查支持范围；对未支持的动态 shape/数据类型明确报错。与 qnn-net-run 对比输出。
4. 交付从零复现命令、源码阅读顺序、生命周期/API/TensorRT 对照、buffer 所有权、日志解释、实测证据和“拿到新 DLC 如何落成项目”的说明。Bash 语法、make doctor、交叉编译、真实设备运行和结果对比构成验收。
5. 后续独立课题：host offline prepare、匹配权重的 PyTorch 精度对齐、性能配置/共享内存、实际 UniAD 子模块支持验证。完成本阶段后不自动展开这些课题。

依赖只检查本地路径，不下载 SDK、模型、NDK 等大型资源；SDK、二进制、输入输出放忽略目录，机器配置放 config/local.env。汇报区分已验证、尚未验证和受阻事项。
