# 任务记录

## 2026-09-09：实现 QNN 生命周期工程与 C++ executable

- 用户正式批准脚本 → context binary → C++ executable 与配套讲解。更新 PLAN/AGENTS，原 PyTorch 权重等待不再阻断当前阶段，精度对齐仍未完成。
- 使用已有 Python 3.11 venv、QAIRT 2.45、本机 NDK 28.2；手机 adb 本轮正常。更新本地忽略配置的 NDK 路径，未下载依赖。
- 新增 lifecycle 四模式、直接 NDK 编译脚本、单文件 C++ runtime、native 输出比较、设备失败检查、host 验收负例测试；扩展 Makefile 与 runtime/doctor。旧单图检测流程保留。
- 实测 dlc.dfykEg、build-context.x2kTtq、context.tBqaMX、cpp.Qs4jXZ：context 61,456,384 bytes，三条执行路径各 6 次、54 组 tensor 检查逐元素完全一致。C++ 成功进程两次，退出 0；3 项错误输入检查退出 1。
- 处理 native classes 文件后缀、profiling symlink 拉取、动态库卸载后的退出 SIGSEGV（使用 RTLD_NODELETE 保留映射；QNN 资源仍全部 free，SDK 内部根因未定位）。失败日志保留。
- 新增 docs/lifecycle-guide.md、cpp/README.md、report/lifecycle-results.md，更新主入口和汇报。报告解释计时边界，不将 context 启动优势等同于稳态推理提速。
- Bash 语法、diff、doctor、设备检查、Python 编译、NDK -Werror 构建、真实设备输出对比、3 个设备负例与3个 host 测试通过。未实现 host offline prepare/动态 shape/量化 I/O/UniAD，未做 PyTorch 对齐。

## 2026-09-09：QNN 生命周期路线评估

- 核对现有 run.sh、本机 QAIRT 2.45 工具文档与 SampleApp，评估用户提出的脚本 → context binary → C++ executable 路线，记录于 report/qnn-lifecycle-roadmap.md。
- 明确预生成可在手机进行、恢复仍有初始化成本、C++ 首版仅恢复 context，以及原生 tensor dtype 与 qnn-net-run float 文件的区别。未修改运行脚本，未执行新推理或下载依赖；PLAN 尚未改写为实施计划。
- 检查：Bash 语法与 git diff --check 通过；make doctor 因 Python 3.14.7（要求 3.11）和沙箱 adb/USB 权限失败，所检查 SDK/DLC 文件存在。

## 2026-09-09

### Git 忽略规则

- 新建并完善根目录 .gitignore：覆盖本地 Python 环境和缓存、config/local.env、用户提供的 external/、生成的 artifacts/ 与 output/，以及 DLC、ONNX、TFLite、PyTorch/SafeTensors 权重和 raw 张量文件。
- 验证：Bash 语法、git diff --check 和各类忽略规则检查通过。make doctor 确认 SDK/DLC 文件存在，但当前 shell 为 Python 3.14.7 且沙箱无法启动 adb server，因而按预期报告 2 项失败。

建立显式 memory 与用户汇报 report。已记录用户最新目标和按阶段执行计划。

- 更新 AGENTS.md：任务开始读取 memory，结束更新 memory 与 report；取消过时的“仅初始化”阶段限制，保留下载/路径等约束。
- 实际拉取 run.lTymhL，四文件 SHA256 一致。新增解码及原图/检测并排图，确认 raw 默认存储 float32（包括 classes），分数不重复 sigmoid。
- 检测图已查看，0.5 阈值下 32 框。未宣称精度一致或完整 UniAD 成功。
- 在 PyTorch 权重决策点暂停，详见 STATE.md 与 report/README.md。
- 验证：全部 Bash 语法、Python 编译、git diff --check、make doctor 通过；输出图片确认被 git 忽略。

### 组会故事线与实验树

- 根据用户补充背景，新增 report/group-meeting-story.md：两分钟讲稿、五页汇报结构、已有证据与边界、正确性→性能→实际 UniAD MSDeformAttn 主干实验树、可选分支及常见问答。report/README.md 已增加入口。
- 检查执行日志与 metadata，确认原有一次成功记录；本轮没有启动新推理、profiling、下载或模型路线变更，保留权重决策暂停点。
- Bash 语法检查、报告相对链接与代码围栏检查通过。make doctor 未通过：当前工具环境 Python 3.14.7（要求 3.11 venv），沙箱 adb 无法访问 USB/启动监听；SDK 和 DLC 文件检查通过。未将此次环境检查报告为成功。
