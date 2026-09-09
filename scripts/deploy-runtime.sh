#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/device-common.sh"
command -v readelf >/dev/null || die '未找到 readelf。'
export LC_ALL=C
[[ -n "${QAIRT_ROOT:-}" ]] || die 'QAIRT_ROOT 未设置。'
android_lib="$QAIRT_ROOT/lib/aarch64-android"
files=("$QAIRT_ROOT/bin/aarch64-android/qnn-net-run"
       "$QAIRT_ROOT/bin/aarch64-android/qnn-context-binary-generator"
       "$QAIRT_ROOT/bin/aarch64-android/qnn-profile-viewer")
for name in libQnnHtp.so libQnnModelDlc.so libQnnHtpV79Stub.so libQnnHtpPrepare.so libQnnSystem.so; do
    files+=("$android_lib/$name")
done
skel="$QAIRT_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so"
declare -A seen=()
for file in "${files[@]}"; do seen["${file##*/}"]=1; done

# 递归解析实际 ARM64 DT_NEEDED；只补充被引用的 SDK 库。
# 系统 / vendor 库留在设备原位置，不能用 host 或 DSP 库替代。
for ((i=0; i<${#files[@]}; i++)); do
    file="${files[i]}"
    [[ -f "$file" ]] || die "缺少文件: $file"
    header="$(readelf -h "$file")"
    [[ "$header" == *AArch64* ]] || die "不是 Android ARM64 ELF: $file"
    dynamic="$(readelf -d "$file")"
    printf '\nDT_NEEDED %s\n' "$file"
    needed="$(sed -n 's/.*(NEEDED).*\[\([^]]*\)\].*/\1/p' <<< "$dynamic")"
    printf '%s\n' "${needed:-(none)}"
    while IFS= read -r dep; do
        [[ -n "$dep" ]] || continue
        [[ "$dep" =~ ^[a-zA-Z0-9_.+-]+$ ]] || die "不安全的库名称: $dep"
        [[ -z "${seen[$dep]:-}" ]] || continue
        seen["$dep"]=1
        if [[ -f "$android_lib/$dep" ]]; then
            files+=("$android_lib/$dep")
        elif [[ "$dep" == libQnn* ]]; then
            die "SDK 缺少依赖: $android_lib/$dep"
        else
            "${ADB[@]}" shell -n "test -f /system/lib64/$dep || test -f /vendor/lib64/$dep || test -f /apex/com.android.runtime/lib64/bionic/$dep" \
                || die "设备未找到 ARM64 依赖: $dep"
            printf '  device system/vendor: %s (运行时可见性由后续加载验证)\n' "$dep"
        fi
    done <<< "$needed"
done
[[ -f "$skel" ]] || die "缺少文件: $skel"
header="$(readelf -h "$skel")"
[[ "$header" == *'QUALCOMM DSP6'* ]] || die "不是 Hexagon ELF: $skel"
dynamic="$(readelf -d "$skel")"
printf '\nDT_NEEDED %s (DSP)\n' "$skel"
sed -n 's/.*(NEEDED).*\[\([^]]*\)\].*/\1/p' <<< "$dynamic"
printf 'DSP 依赖由 DSP 环境提供；--help 不验证 DSP 加载或 HTP 推理。\n'

"${ADB[@]}" shell "mkdir -p '$RUNTIME_DIR'"
"${ADB[@]}" push "${files[@]}" "$skel" "$RUNTIME_DIR/"
"${ADB[@]}" shell "chmod +x '$RUNTIME_DIR/qnn-net-run' '$RUNTIME_DIR/qnn-context-binary-generator'"
"${ADB[@]}" shell "chmod +x '$RUNTIME_DIR/qnn-profile-viewer'"
"${ADB[@]}" shell "LD_LIBRARY_PATH='$RUNTIME_DIR' ADSP_LIBRARY_PATH='$RUNTIME_DIR' '$RUNTIME_DIR/qnn-net-run' --help"
"${ADB[@]}" shell "LD_LIBRARY_PATH='$RUNTIME_DIR' ADSP_LIBRARY_PATH='$RUNTIME_DIR' '$RUNTIME_DIR/qnn-context-binary-generator' --help"
printf '\nruntime 部署完成；qnn-net-run --help 退出成功。\n'
