#!/usr/bin/env bash
set -euo pipefail
# 真实设备的失败路径检查：必须返回 1，不能段错误，也不能产生推理成功假象。
source "$(dirname -- "${BASH_SOURCE[0]}")/device-common.sh"
root="$PROJECT_ROOT/output/lifecycle"
[[ -f "$root/latest-cpp.txt" && -f "$root/latest-build-context.txt" ]] || die '请先完成 cpp-run。'
cpp="$(< "$root/latest-cpp.txt")"
build="$(< "$root/latest-build-context.txt")"
[[ "$cpp" =~ ^cpp\.[a-zA-Z0-9]+$ && "$build" =~ ^build-context\.[a-zA-Z0-9]+$ ]] || die '实验记录路径异常。'
base=/data/local/tmp/qnn_mobile
exe="$base/lifecycle/$cpp/qnn-context-runner"
binary="$base/lifecycle/$build/rf_detr.bin"
remote="$("${ADB[@]}" shell "mktemp -d '$base/lifecycle/check.XXXXXX'" | tr -d '\r')"
[[ "$remote" =~ ^/data/local/tmp/qnn_mobile/lifecycle/check\.[a-zA-Z0-9]+$ ]] || die '检查目录异常。'
result="$root/${remote##*/}"
mkdir -p "$result"
env_cmd="LD_LIBRARY_PATH='$RUNTIME_DIR' ADSP_LIBRARY_PATH='$RUNTIME_DIR'"
common="$env_cmd '$exe' --backend '$RUNTIME_DIR/libQnnHtp.so' --system '$RUNTIME_DIR/libQnnSystem.so' --runs 1"
"${ADB[@]}" shell "$env_cmd '$exe' --help" > "$result/help.txt"

expect_failure() {
    local name="$1" expected="$2" command="$3" status=0
    "${ADB[@]}" shell "$command" > "$result/$name.log" 2>&1 || status=$?
    [[ "$status" == 1 ]] || die "$name 应退出 1，实际 $status；见 $result"
    rg -F "$expected" "$result/$name.log" >/dev/null || die "$name 失败原因不符预期。"
    if rg '\[OK\]|\[TIME\] execute_' "$result/$name.log" >/dev/null; then
        die "$name 不应执行图或报告成功。"
    fi
    printf '[OK] %s: exit=1, %s\n' "$name" "$expected" | tee -a "$result/checks.txt"
}
expect_failure wrong-name 'Missing input: image' \
    "$common --context '$binary' --input 'wrong=$base/input/rf_detr/image.raw' --output-dir '$remote/wrong-name'"
expect_failure wrong-size 'Wrong input byte count for image' \
    "$common --context '$binary' --input 'image=$base/lifecycle/$cpp/Result_0/classes.raw' --output-dir '$remote/wrong-size'"
expect_failure malformed-context 'systemContextGetMetaData failed' \
    "$common --context '$base/input/rf_detr/image.raw' --input 'image=$base/input/rf_detr/image.raw' --output-dir '$remote/malformed-context'"
printf '[OK] C++ 失败路径与正常退出检查: %s\n' "$result"
