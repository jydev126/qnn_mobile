# 目标与顺序

用户目的：工作汇报与个人实验。工作方向是 UniAD 部署到高通平台，目前用商用 S25 Ultra，无开发板。汇报必须区分已验证事实和未完成目标。

AI Hub API Token 无法获取，因此当前采用现成 RF-DETR small DLC + QAIRT 2.45 HTP online prepare。不要求先导出 qnn_context_binary；不将 DLC 成功描述为 context binary 或完整 UniAD 成功。

1. 拉取手机结果、按 recipe 解码、画检测框。
2. 匹配权重和同图预处理的 PyTorch 对比；权重缺失/不匹配时停下让用户决策，不自动下载。
3. profiling：分开 graph prepare/finalize 与 execute，重复测量波动，不以 adb 总耗时作为推理延迟。
4. 更新 report 的检测图片、性能表、证据与可复现命令。
5. 提取实际 UniAD MSDeformAttn standalone，固定 shape，建立基准并检查本地转换/HTP 支持。
6. 随实验需要参数化脚本，避免提前扩展工程。

按顺序执行，遇到需要用户决策的实际分歧停下。禁止自动下载 SDK、模型权重等大型资源；不修改 RF-DETR 来掩盖运行错误。
