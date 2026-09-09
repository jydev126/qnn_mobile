# 当前状态

更新日期：2026-09-09。

## 当前阶段已完成

用户批准的 RF-DETR 生命周期路线已经实现：shell/qnn-net-run 观察 DLC Compose/Finalize/Execute → 设备预生成 context → net-run 恢复 → 自写 C++ executable 恢复执行。正式计划在 PLAN.md；教程 docs/lifecycle-guide.md、源码导读 cpp/README.md、实测 report/lifecycle-results.md。AGENTS 已同步授权边界：Android 命令行 executable，无 APK/CMake/JNI/Android Studio。

本机可用 Python 3.11.16 venv、QAIRT v2.45.0.260326154327、Android NDK 28.2.13676358；本轮 adb 可访问手机 SM-S9380 / sun / qcom / Android 16 API 36 / SELinux Enforcing。之前 Python/沙箱 adb 检查失败已不代表当前环境，运行时需 source .venv/bin/activate。NDK 路径写在被忽略的 config/local.env，未下载依赖。

## 工程与可复现证据

新增 make lifecycle-dlc / context-build / context-run / cpp-build / cpp-run / lifecycle-compare / cpp-check。先 doctor、inspect-device、prepare、deploy-runtime、deploy。运行次数默认6，可用 NUM_INFERENCES 调整。所有生命周期日志/输入输出/binary/executable 为忽略产物，仓库只存源码、脚本、测试和讲解。

固定成功记录（output/lifecycle/ 和设备 /data/local/tmp/qnn_mobile/lifecycle/）：
- dlc.dfykEg：6次，Compose 418.095 ms，Finalize 9670.098 ms。
- build-context.x2kTtq：rf_detr.bin 为 61,456,384 bytes；设备生成，不是 host offline prepare。
- context.tBqaMX：6次，恢复路径 INIT 57.937 ms，无再次 Finalize。
- cpp.Qs4jXZ：6次，contextCreateFromBinary 60.419 ms，整个进程退出0；更早 cpp.22aA7I 也退出0。
- comparison.json：三条路径各6次、54组输出逐元素完全一致，max_abs=0；后续平均 Execute 分别162.344 / 167.215 / 162.969 ms。时间口径与频率未控制，不宣称 C++ 加速或稳定性能。
- check.xhLNq8：错张量名、错输入字节数、损坏context均在Execute前失败并退出1。
- latest-*.txt 记录每阶段最近成功，失败不更新；报告固定目录保留本次快照。

C++ 仅支持单graph、静态dense float32/int32 tensor、native raw、同步Execute。图/tensor描述从binary metadata读取；使用systemContextGetMetaData。QNN资源显式free，System metadata保持到执行结束。初次cpp.SaitKX完成执行后在库卸载/退出阶段SIGSEGV，日志保留；现用RTLD_NODELETE保持runtime映射到进程退出，内部具体符号根因未定位。

新工具实验开启native输出：classes为int32，net-run文件名classes_native.raw，C++为classes.raw；boxes/logits为float32。profiling主log是symlink，脚本显式取目标并核对hash；viewer在设备运行。

验证通过：全部Bash语法、配置示例语法、git diff --check、make doctor、inspect-device、Python编译、NDK -Wall/-Wextra/-Werror构建、真实设备三路径输出比较、3项设备失败检查、3项host比较测试（原始证据通过、篡改类别拒绝、输入身份不一致拒绝）。

## 保留的已有成果与后续边界

旧 make run/pull/decode 检测流程保留。RF-DETR image float32 [1,3,512,512]、RGB bilinear resize /255，不外加mean/std，3,145,728 bytes。输出boxes float32 [1,300,4]、logits float32 [1,300]、classes int32 [1,300]；旧net-run raw默认float（包括classes）。output/rf_detr/run.lTymhL/comparison.png已查看，阈值0.5为32框，300条预测保留。新native多次结果不能直接交给旧单次decode。

PyTorch匹配checkpoint仍缺失，DLC metadata无checkpoint哈希；这限制框架精度对齐，但用户新计划已解除等待权重的推进阻塞。未下载权重，未验证mAP。report/group-meeting-story.md保留此前组会规划，当前事实以lifecycle-results.md为准。

本阶段交付完成。下一步由用户选择学习/扩展主题：host offline prepare、其他dtype/动态shape/共享内存/异步、匹配权重精度对齐，或实际UniAD子模块。未自动开展UniAD或将RF-DETR成功描述为UniAD成功。
