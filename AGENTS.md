# qnn_mobile 开发约束

- Host：Fedora Linux x86_64；Python：3.11 venv；手机通过 adb 访问。
- Target：Samsung Galaxy S25 Ultra，Snapdragon 8 Elite for Galaxy / SM8750，HTP V79。
- 唯一路线：Qualcomm AI Hub Models RF-DETR small QAIRT DLC → QAIRT 2.45.0.260326 → Samsung Galaxy S25 Ultra → `qnn-net-run` → `libQnnModelDlc.so` → `libQnnHtp.so` → HTP V79；模型路径使用 `RF_DETR_DLC`。
- 初期仅通过 `qnn-net-run` 在手机 `/data/local/tmp` 下执行，不做 APK、CMake、JNI 或 Android Studio 工程。
- 外部依赖由用户手动下载。不得主动下载 QAIRT SDK、模型大文件或其他大型第三方资源；只定义路径、检查文件并报告缺失项。
- SDK、模型文件、输入及输出数据不得提交 git，放入被忽略的 `external/`、`artifacts/`、`output/`；机器专用配置放在 `config/local.env`。
- Bash 脚本必须开启 `set -euo pipefail`，路径须正确引用，不得硬编码用户 HOME，不得使用 sudo。
- 保持结构简单，按 memory/PLAN.md 分阶段实现；用户已授权输入准备、手机部署、推理、解码及后续实验，到需要用户决策时停下。
- 每次 Codex 任务开始必须读取 memory/README.md、memory/PLAN.md、memory/STATE.md；任务结束必须更新 memory/STATE.md 和 memory/HISTORY.md，有用户相关成果时同步更新 report/。记忆中的旧结论不得覆盖用户新指令。
- 修改后运行 Bash 语法检查、`make doctor` 以及相关检查。环境资源缺失应如实报告，不得伪装成功。
