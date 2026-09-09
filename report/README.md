# 高通手机部署实验汇报

组会材料：[故事线、两分钟讲稿、五页汇报结构与后续实验树](group-meeting-story.md)。

当前阶段：[DLC / context / C++ 实测报告](lifecycle-results.md) · [操作与原理教程](../docs/lifecycle-guide.md) · [C++ 源码导读](../cpp/README.md)。本阶段替代此前等待 PyTorch 权重的推进顺序，精度对齐仍单独标记为未完成。

## 目标

用 Samsung Galaxy S25 Ultra 验证高通 HTP 部署流程，为后续 UniAD MSDeformAttn 单算子/模块实验积累可复现证据。目前实验模型是 RF-DETR small，不是完整 UniAD。

## 已完成

QAIRT 2.45 → RF-DETR DLC → libQnnModelDlc → libQnnHtp → 手机 HTP 在线图准备及一次推理成功。输入 RGB float32 NCHW [1,3,512,512]。日志及 metadata 证实 Compose、Finalize、Execute 完成，inferences_completed=1。

新增：本地手机预生成 context binary、net-run 恢复执行、自写 C++ executable 恢复执行均通过。三条路径各 6 次的原始输出逐元素相同。数据、计时口径及限制见当前阶段实测报告。

## 限制

无 AI Hub Token，当前不走云端 context binary 导出。尚未证明 PyTorch 数值一致性、检测精度、稳定性能或 UniAD 可部署。RF-DETR 成功不能证明 UniAD 不同实现和 shape 的 MSDeformAttn 兼容。

## 复现入口

先配置 config/local.env 并手动准备依赖，然后依次执行 make doctor、make prepare MODEL=rf_detr、make deploy-runtime、make deploy MODEL=rf_detr、make run MODEL=rf_detr。日志在 ../output/rf_detr/qnn-net-run.log。

最新检测展示、证据和待决策事项随任务更新。数据/图片保留在被忽略的 output/，本目录只保存可提交的文字汇报与链接。

## 2026-09-09：第一张检测结果

![原图与 HTP 检测结果](../output/rf_detr/run.lTymhL/comparison.png)

[全分辨率检测图](../output/rf_detr/run.lTymhL/detections.png) · [完整 300 条预测](../output/rf_detr/run.lTymhL/detections.json) · [执行 metadata](../output/rf_detr/run.lTymhL/execution_metadata.yaml) · [运行日志](../output/rf_detr/qnn-net-run.log)

| 检查项 | 结果 |
| --- | --- |
| 成功推理次数 | 1 |
| boxes / logits / classes 文件大小 | 4800 / 1200 / 1200 bytes |
| 拉取完整性 | 四个原始结果文件与手机 SHA256 一致 |
| 分数范围 | 0.06909–0.93945 |
| 展示阈值 | 0.5，显示 32 个框 |
| 解码 | boxes 为 512 坐标系 xyxy，按原图宽高分别缩放；分数已 sigmoid，不重复处理 |
| 类别读取 | 本次默认 float32 输出，读取后验证整数再转类别 ID；不能按逻辑 int32 直接读取 raw |

图像可见合理的公交车、汽车、行人和摩托车检测；这是单图可视化检查，不是 mAP 或数值一致性验证。显示采用 recipe DETR app 的阈值筛选，不额外做 NMS；JSON 保留全部 300 条预测。类别名称来自本地 Qualcomm DETR LABEL_MAP，个别英文名在该表中为简写。

复现新增步骤：`make pull MODEL=rf_detr` → `make decode MODEL=rf_detr`。阈值可用 `SCORE_THRESHOLD=0.7 make decode MODEL=rf_detr` 调整。

## 暂缓的 PyTorch 对齐

torch/rfdetr 已安装，但项目及常用缓存未找到 RF-DETR small 权重；现有 DLC metadata 不含 checkpoint 版本或哈希，不能保证任意下载的 small 权重与 DLC 完全一致。

用户于 2026-09-09 明确改为先完成生命周期、context 和 C++ 项目。未来提供匹配权重后可恢复框架对齐；当前 runtime 三路径一致性不替代 PyTorch/mAP 验证，不自动下载权重。
