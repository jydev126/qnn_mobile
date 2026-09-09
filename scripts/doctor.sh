#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_ROOT/config/local.env" ]]; then
    # 本地配置为受信任的 Bash 文件，优先于同名环境变量。
    source "$PROJECT_ROOT/config/local.env"
fi

failures=0
ok() { printf '[OK] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; failures=$((failures + 1)); }

python_cmd=""
if command -v python >/dev/null 2>&1; then
    python_cmd=python
elif command -v python3 >/dev/null 2>&1; then
    python_cmd=python3
fi
if [[ -z "$python_cmd" ]]; then
    fail '未找到 python / python3；请准备 Python 3.11 venv 并激活。'
elif python_version="$("$python_cmd" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"; then
    if [[ "$python_version" == 3.11.* ]]; then
        ok "Python $python_version ($(command -v "$python_cmd"))"
    else
        fail "当前 Python 为 $python_version；项目要求 Python 3.11，请激活对应 venv。"
    fi
else
    fail 'Python 无法运行。'
fi

if command -v git >/dev/null 2>&1; then
    ok "git: $(command -v git)"
else
    fail '未找到 git。'
fi

# 固定检查目标 Android / HTP V79 的 SDK 文件，不使用 host 库替代。
check_file() {
    if [[ -f "$1" ]]; then
        ok "文件存在: $1"
    else
        fail "缺少文件: $1"
    fi
}

if [[ -n "${QAIRT_ROOT:-}" ]]; then
    check_file "$QAIRT_ROOT/bin/aarch64-android/qnn-net-run"
    check_file "$QAIRT_ROOT/bin/aarch64-android/qnn-context-binary-generator"
    check_file "$QAIRT_ROOT/bin/aarch64-android/qnn-profile-viewer"
    check_file "$QAIRT_ROOT/include/QNN/QnnInterface.h"
    check_file "$QAIRT_ROOT/include/QNN/System/QnnSystemInterface.h"
    for library in libQnnHtp.so libQnnModelDlc.so libQnnHtpV79Stub.so libQnnHtpPrepare.so libQnnSystem.so; do
        check_file "$QAIRT_ROOT/lib/aarch64-android/$library"
    done
    check_file "$QAIRT_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so"
else
    fail 'QAIRT_ROOT 未设置；请在 config/local.env 或环境变量中填写 QAIRT 2.45.0.260326 SDK 根目录，无法检查 runtime 文件。'
fi

if [[ -n "${ANDROID_NDK_ROOT:-}" ]]; then
    check_file "$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${ANDROID_API:-28}-clang++"
else
    printf '[INFO] ANDROID_NDK_ROOT 未设置；DLC/context 工具阶段可继续，cpp-build 前需手动准备。\n'
fi

if [[ -z "${RF_DETR_DLC:-}" ]]; then
    fail 'RF_DETR_DLC 未设置；请填写 RF-DETR QAIRT .dlc 文件路径。'
elif [[ "$RF_DETR_DLC" != *.dlc ]]; then
    fail "RF_DETR_DLC 必须指向 .dlc 文件: $RF_DETR_DLC"
else
    check_file "$RF_DETR_DLC"
fi

if command -v adb >/dev/null 2>&1; then
    ok "adb: $(command -v adb)"
    if devices="$(adb devices -l 2>&1)"; then
        printf '%s\n' "$devices"
        if awk -v serial="${DEVICE_SERIAL:-}" '
            $2 == "device" && (serial == "" || $1 == serial) { found = 1 }
            END { exit !found }
        ' <<< "$devices"; then
            ok 'adb 发现可用设备（状态为 device）。'
        else
            fail '未发现匹配的可用设备；请连接手机、开启 USB 调试并授权，检查 DEVICE_SERIAL。'
        fi
    else
        printf '%s\n' "$devices"
        fail 'adb devices 执行失败；请检查 adb server 与主机 USB 访问权限。'
    fi
else
    fail '未找到 adb，无法检查手机连接。'
fi

if (( failures > 0 )); then
    printf '\ndoctor: %d 项未通过。\n' "$failures"
    exit 1
fi
printf '\ndoctor: host、SDK 所需文件与 DLC 路径检查通过；未验证 SDK 版本、DLC 内容、手机型号或 HTP 推理兼容性。\n'
