# RF-DETR small

模型来源：Qualcomm AI Hub Models。唯一路线：RF-DETR QAIRT DLC → QAIRT 2.45.0.260326 → Samsung Galaxy S25 Ultra（SM8750）→ `qnn-net-run` → `libQnnModelDlc.so` → `libQnnHtp.so` → HTP V79。

本目录只保存可提交的说明及未来的模型专用脚本，不保存模型、图片或推理结果。通过 `RF_DETR_DLC` 指定用户提供的真实 `.dlc` 文件，通过 `RF_DETR_INPUT_IMAGE` 指定测试图片；文件放在被忽略的 `artifacts/` 中。SDK 由用户手动提供，使用 `QAIRT_ROOT` 指定路径。

## 输入准备

运行 `make prepare MODEL=rf_detr`，使用项目 Python 3.11 `.venv` 中的 numpy 和 Pillow。

已阅读本地 Qualcomm recipe（相对于 `external/ai-hub-models/src/qai_hub_models/`）：

- `models/rf_detr/model.py`：`DEFAULT_VARIANT="small"`、`VARIANT_RESOLUTION["small"]=512`；`RF_DETR.get_input_spec` 明确输入名 `image`、shape `(1,3,512,512)`、dtype `float32`、RGB、range `[0,1]`。
- `models/templates/detr/app.py`：`DETRApp.predict` 直接 resize 到目标尺寸，使用 PIL `Resampling.BILINEAR`，无 padding/crop。
- `utils/image_processing.py`：`preprocess_PIL_image` 转 NCHW、float32 并除以 255；`normalize_image_torchvision` 是 mean/std 操作，由 `RF_DETR.forward` 在模型内部调用，不能在 raw 输入上重复执行。

`prepare.py` 使用 PIL RGB → resize 512×512 bilinear → numpy float32 /255 → 连续 NCHW `[1,3,512,512]`。raw 为小端 float32、无文件头，强制检查 3145728 bytes、有限值及范围。

输出在被忽略的 `artifacts/rf_detr/input/`：

- `image.raw`
- `input_list.txt`，内容为 `image:=image.raw` 加换行。

输入 tensor 名称依据上述 recipe，未用 DLC metadata 验证实际 DLC。列表中的相对路径要求未来执行 `qnn-net-run` 时，工作目录为这两个文件所在目录；部署时应保持二者同目录。

`prepare` 只准备输入，不读取或修改 DLC。`deploy-runtime` 已实现。`make deploy MODEL=rf_detr` 部署 DLC 和输入，并另外生成含设备绝对路径的列表，详见根目录 README；不修改本地 DLC。`make run MODEL=rf_detr` 已实现 HTP DLC online prepare，完整日志位于 `output/rf_detr/qnn-net-run.log`；`make pull MODEL=rf_detr` 拉取及校验结果；`make decode MODEL=rf_detr` 生成可视化与完整预测 JSON。
