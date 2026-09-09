# RF-DETR QNN 生命周期学习路线评估

2026-09-09：本页最初为路线评估，现已由用户批准并实施。正式计划在 [memory/PLAN.md](../memory/PLAN.md)，完成情况见 [实测报告](lifecycle-results.md)，使用教程见 [lifecycle-guide](../docs/lifecycle-guide.md)。以下保留设计边界。

1. 保留 RF-DETR DLC、输入和已有 shell/qnn-net-run 路径，观察 Compose、Finalize、首次 Execute、后续 Execute。分别记录初始化、建图、finalize、执行和文件 I/O；verbose 日志不保证提供每个阶段的完整计时，需要 profiling 或后续 API 计时补足。
2. 使用 qnn-context-binary-generator，通过 libQnnModelDlc.so 读取 RF_DETR_DLC，以 HTP backend 完成建图与 finalize 后序列化。优先考虑手机预生成以保持目标环境一致，再研究 host offline prepare。预生成不等于必须在 PC 上生成；host 路径需检查 Fedora 兼容性、目标 SoC/V79 配置和库依赖。设备用 qnn-net-run --retrieve_context 恢复执行，使用相同输入对比原始输出、首次执行和稳态执行。
3. 最小 Android aarch64 C++ executable 先仅支持 context 恢复：加载 API provider、初始化 backend/device、通过 System API 读取图和张量信息、contextCreateFromBinary、graphRetrieve、准备 tensor buffer、graphExecute、释放资源。用 shell 调用交叉编译器即可，不需要 APK/JNI/CMake；已发现并使用本地 Android NDK 28.2，未下载。

语义边界：Compose 是通过模型加载器创建图、张量和节点的流程，不是一个通用 QnnGraph_compose API；Finalize 是关键 API 边界，HTP prepare 工作主要关联该过程，不应假设日志中全部准备成本都落在一个事件。Context binary 是序列化的已准备 context，可包含多个图，不是 executable，仍依赖兼容 QNN/HTP runtime。恢复仍有反序列化、设备资源初始化和首次执行成本；不保证稳态推理提速。记录 SDK、目标 SoC/HTP、配置和模型身份，不能假设任意设备/版本通用。

验收：DLC 与 context 输出一致性；qnn-net-run 与 C++ 输出一致性；同一输入与执行配置下分阶段计时。注意现有 qnn-net-run 默认 raw 输出为 float，C++ 应以 tensor metadata 的逻辑 dtype 处理，尤其 classes int32，比较前统一表示。工具链一致性不等价于 PyTorch 精度验证，也不证明 UniAD 算子可部署。

依据：本机 QAIRT 2.45 docs/QAIRT-Docs/QNN/general/tools.html、examples/QNN/SampleApp/SampleApp/src/QnnSampleApp.cpp；[Qualcomm AI Hub FAQ](https://dev.aihub.qualcomm.com/docs/hub/faq.html)；[Qualcomm DLC2BIN 示例](https://github.com/qualcomm/qai-appbuilder/blob/main/tools/convert/dlc2bin/README.md)。
