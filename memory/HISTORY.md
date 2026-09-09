# 任务记录

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
