# qnn_mobile

Host 为 Fedora Linux x86_64，使用 Python 3.11 venv；Target 为 Samsung Galaxy S25 Ultra（Snapdragon 8 Elite for Galaxy / SM8750，HTP V79）。第一个模型为 Qualcomm AI Hub Models 的 **RF-DETR small**。

唯一路线：

```text
RF-DETR QAIRT DLC
-> QAIRT 2.45.0.260326
-> Samsung Galaxy S25 Ultra
-> qnn-net-run
-> libQnnModelDlc.so
-> libQnnHtp.so
-> HTP V79
```

通过 adb 将 runtime 放入 `/data/local/tmp/qnn_mobile/runtime`，由 `qnn-net-run` 执行。当前实现 doctor、设备检查、最小 runtime 部署与 `--help` 验证；支持 RF-DETR small 图片输入准备，支持模型及输入部署，支持 HTP DLC online prepare 推理，不包含 APK、CMake、JNI 或 Android Studio 工程。

## 目录

```text
config/                 配置示例；local.env 为本地私有配置
scripts/                Bash 检查与命令入口
models/rf_detr/          模型说明，后续添加模型专用处理逻辑
external/               用户手动提供的 QAIRT SDK 等（忽略）
artifacts/              模型、测试图片、预处理输入等（忽略）
output/                 手机拉回的输出（忽略）
.venv/                  Python 3.11 虚拟环境（忽略）
```

数据目录按需创建。SDK、模型、输入和输出不能提交 git，必须放入忽略的数据目录。

## 配置与检查

主机需已有 Python 3.11、git、make 和 adb。外部依赖由用户手动提供，仓库不下载资源，也不使用 sudo。

```bash
# 若尚无 venv，先创建；已有 Python 3.11 venv 可直接激活。
python3.11 -m venv .venv
source .venv/bin/activate
cp config/s25u.env.example config/local.env
```

编辑 `config/local.env`，填写 QAIRT 2.45.0.260326 SDK 根目录 `QAIRT_ROOT`、模型文件路径 `RF_DETR_DLC` 和测试图片路径 `RF_DETR_INPUT_IMAGE`。推荐使用绝对路径，不能提交该配置。也可通过环境变量传入配置；存在 `config/local.env` 时，其中赋值优先。该文件会作为 Bash 执行，只使用自己信任的配置。

手机连接 USB，开启 USB 调试并接受授权后运行 `make doctor`。doctor 检查：

- 当前 PATH 中的 Python 是否为 3.11，git 和 adb 是否存在。
- `QAIRT_ROOT` 非空，并包含以下真实文件（路径相对于 SDK 根目录）：

```text
bin/aarch64-android/qnn-net-run
lib/aarch64-android/libQnnHtp.so
lib/aarch64-android/libQnnModelDlc.so
lib/aarch64-android/libQnnHtpV79Stub.so
lib/aarch64-android/libQnnHtpPrepare.so
lib/aarch64-android/libQnnSystem.so
lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so
```

- `RF_DETR_DLC` 指向真实存在且扩展名为 `.dlc` 的文件。
- `adb devices -l` 是否发现状态为 `device` 的手机。设置 `DEVICE_SERIAL` 后只匹配该设备；留空时任一可用设备即可通过。

缺失项会如实报告，doctor 返回非零状态。文件存在不代表 SDK 版本、DLC 内容、手机型号或 HTP 推理兼容性已验证；图片和输入输出规格留待后续阶段检查。adb 检查可能启动主机 adb server。

## 命令入口

```bash
make doctor
make inspect-device
make prepare MODEL=rf_detr
make deploy-runtime
make deploy MODEL=rf_detr
make run MODEL=rf_detr
make pull MODEL=rf_detr
```

`inspect-device` 输出型号、board platform、hardware、Android 版本、API level 与 SELinux 状态。`inspect-device` 和 `deploy-runtime` 在 `DEVICE_SERIAL` 留空时要求恰好一台在线设备，并固定使用该序列号。

`deploy-runtime` 需要主机 `readelf`，从 `QAIRT_ROOT` 检查上述 7 个文件的 ELF 架构与 DT_NEEDED，递归补充真实依赖的 Android ARM64 SDK 库，并检查设备端系统/vendor 依赖文件是否存在。只 push 选中的文件，不复制整个 SDK，不读取模型。设备目录固定为 `/data/local/tmp/qnn_mobile/runtime`。随后添加 `qnn-net-run` 执行权限，以该目录设置 `LD_LIBRARY_PATH` 和 `ADSP_LIBRARY_PATH`，实际执行 `qnn-net-run --help`；任一步失败即返回非零状态。

`--help` 成功仅验证程序启动，不代表 backend、DLC 或 DSP 加载成功。V79 Skel 的 DSP 依赖单独报告，不用 Android 库替代；系统/vendor 文件存在也不代表其对所有动态加载场景可见。

`make pull MODEL=rf_detr` 拉取最近成功运行并校验 SHA256；`make decode MODEL=rf_detr` 解码并生成检测图。

修改后的检查：

```bash
for script in scripts/*.sh; do bash -n "$script"; done
bash -n config/s25u.env.example
git diff --check
make doctor
```

## RF-DETR 输入准备

`make prepare MODEL=rf_detr` 使用项目 `.venv`（Python 3.11，需用户手动提供 numpy 和 Pillow）读取 `RF_DETR_INPUT_IMAGE`，生成 `artifacts/rf_detr/input/image.raw` 和同目录 `input_list.txt`，打印 shape、dtype、min、max 和文件字节数。预处理及 recipe 依据见 [模型说明](models/rf_detr/README.md)。不加载模型、不下载依赖、不访问手机。

## RF-DETR 部署

`make deploy MODEL=rf_detr` 从 `RF_DETR_DLC` 和 prepare 产物部署到手机：

- `/data/local/tmp/qnn_mobile/models/rf_detr/model.dlc`
- `/data/local/tmp/qnn_mobile/input/rf_detr/image.raw`
- `/data/local/tmp/qnn_mobile/input/rf_detr/input_list.txt`

同时创建 `/data/local/tmp/qnn_mobile/output/rf_detr`，保留已有输出。设备列表内容固定为 `image:=/data/local/tmp/qnn_mobile/input/rf_detr/image.raw`，不包含 host 路径，不依赖运行工作目录。其本地副本为被忽略的 `artifacts/rf_detr/input/input_list.device.txt`。部署前检查 DLC 非空和 raw 字节数，部署后比对三个文件的 SHA256；任何失败返回非零。不执行推理。

## RF-DETR 推理

部署 runtime、模型和输入后执行 `make run MODEL=rf_detr`。每次先读取设备 `qnn-net-run --help` 并检查所需参数。根据设备帮助及 QAIRT 2.45 本地 `QNN/general/tutorial5.html`，使用 `--backend libQnnHtp.so`、`--model libQnnModelDlc.so`、`--dlc_path model.dlc`（实际传入设备绝对路径），进行 DLC online prepare，不使用离线 context。两个库搜索环境变量均设为 `/data/local/tmp/qnn_mobile/runtime`。

完整 stdout/stderr（含帮助、路径及退出状态）保存到 `output/rf_detr/qnn-net-run.log`，上一次日志保留为 `qnn-net-run.previous.*.log`。设备结果使用 `/data/local/tmp/qnn_mobile/output/rf_detr/run.XXXXXX` 独立目录，实际路径记入日志，避免混入旧结果。失败返回非零，不修改模型。

本机实测 QAIRT `v2.45.0.260326154327` 完成 Compose、Finalize、Execute，退出 0，产生 `Result_0/boxes.raw`、`classes.raw`、`logits.raw` 及 `execution_metadata.yaml`。这确认推理执行完成，尚未验证检测精度。

若后续失败，按 adb → 动态库 → LD_LIBRARY_PATH → ADSP_LIBRARY_PATH → V79 Stub/Skel → HTP device create → libQnnModelDlc → DLC load → input tensor name → input raw 的顺序排查，保留原始日志及 DLC。

## 项目记忆与汇报

每次任务读取 [memory](memory/README.md)，按 [计划](memory/PLAN.md) 推进并更新状态。用户查看 [实验汇报](report/README.md)，包括检测图片、证据、限制和待决策事项。
