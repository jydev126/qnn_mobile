# 当前状态

更新日期：2026-09-09。

本轮完成：补充根目录 .gitignore，忽略项目 Python 环境与缓存、机器专用 config/local.env，以及 external/、artifacts/、output/ 和常见模型/原始张量文件，避免 SDK、模型、输入和结果进入 Git。Bash 语法、git diff --check 和忽略规则检查通过；make doctor 仍因当前 shell 的 Python 3.14.7（要求 3.11）及沙箱没有 USB/adb server 监听权限而失败，QAIRT runtime 和 DLC 文件检查通过。

已验证：QAIRT v2.45.0.260326154327；手机 SM-S9380 / sun / qcom / Android 16 API 36 / SELinux Enforcing。
make doctor、inspect-device、prepare、deploy-runtime、deploy、run 已实现。RF-DETR small 输入 image float32 [1,3,512,512]，RGB bilinear resize /255，无外部 mean/std，3145728 bytes。
HTP DLC online prepare 已成功，metadata: inferences_completed=1。最近成功设备目录由 output/rf_detr/qnn-net-run.log 的 [OK] 行记录。输入输出数据在 artifacts/ 和 output/，机器路径在 config/local.env。
实际输出：boxes float32 [1,300,4]、logits float32 [1,300]、classes int32 [1,300]。qnn-net-run 未指定 native output，raw 默认 float 输出，解码时必须与 metadata 中逻辑 dtype 区分。
当前进行：结果拉取与可视化，随后检查 PyTorch 权重。

本轮完成：make pull（按成功日志定位 run、拉取并校验 SHA256）、make decode（默认 float 输出正确读取 classes，按 recipe 分数和 xyxy 解码）。结果 output/rf_detr/run.lTymhL/，comparison.png 已视觉检查；阈值 0.5 显示 32 框，完整 300 条预测在 detections.json。report/README.md 包含图片、指标、限制与决策项。

暂停点：计划第 2 步 PyTorch 同图对比。torch/rfdetr 已安装；项目及 ~/.cache/huggingface、~/.cache/torch、常用缓存未找到真实 checkpoint；DLC metadata 无 checkpoint 哈希。等用户选择：提供本地权重与来源/版本，或明确跳过对比先做 profiling。未自动下载，未做 profiling，未参数化通用模型运行，未开始 UniAD。

本轮汇报规划：已根据用户补充的 UniAD 环境受阻背景和现有日志，新增 report/group-meeting-story.md，包含组会讲稿、五页结构、事实边界、正确性→性能→实际 UniAD MSDeformAttn 实验树及分支交付标准。未开展新推理或 profiling；权重决策暂停点保持不变。
本轮检查限制：Bash 语法与报告链接检查通过；make doctor 因当前 Python 3.14.7 和沙箱 adb/USB 权限失败，SDK/DLC 文件检查通过。此为当前工具环境检查，不覆盖此前真实设备成功记录。
