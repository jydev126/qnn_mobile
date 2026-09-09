#!/usr/bin/env bash
set -euo pipefail
# 一次只做一个阶段；每次输出独立目录。旧 make run/pull/decode 不受影响。
mode="${1:-}"
case "$mode" in dlc|build-context|context|cpp) ;; *) printf '用法: %s dlc|build-context|context|cpp\n' "$0" >&2; exit 2 ;; esac
source "$(dirname -- "${BASH_SOURCE[0]}")/device-common.sh"
[[ "${MODEL:-rf_detr}" == rf_detr ]] || die '此脚本的输入部署与验收仅支持 rf_detr。'
count="${NUM_INFERENCES:-6}"
[[ "$count" =~ ^[1-9][0-9]*$ ]] && (( count <= 1000 )) || die 'NUM_INFERENCES 必须为 1..1000。'
base=/data/local/tmp/qnn_mobile
remote_root="$base/lifecycle"
local_root="$PROJECT_ROOT/output/lifecycle"
mkdir -p "$local_root"
"${ADB[@]}" shell "mkdir -p '$remote_root'"
remote="$("${ADB[@]}" shell "mktemp -d '$remote_root/$mode.XXXXXX'" | tr -d '\r')"
[[ "$remote" =~ ^/data/local/tmp/qnn_mobile/lifecycle/[a-z-]+\.[a-zA-Z0-9]+$ ]] || die '设备实验路径异常。'
result="$local_root/${remote##*/}"
mkdir -p "$result"

# 所有设备路径由本脚本产生并验证；命令写入记录，便于手动重放和学习。
env_cmd="LD_LIBRARY_PATH='$RUNTIME_DIR' ADSP_LIBRARY_PATH='$RUNTIME_DIR'"
check_hash() {
    local host_hash device_hash
    [[ -s "$1" ]] || die "缺少本地文件: $1"
    host_hash="$(sha256sum < "$1")"
    device_hash="$("${ADB[@]}" shell "sha256sum '$2'")"
    [[ "${host_hash%% *}" == "${device_hash%% *}" ]] || die "内容不同，请重新部署: $2"
    printf '%s  %s\n' "${host_hash%% *}" "$2" >> "$result/sha256.txt"
}
[[ -n "${QAIRT_ROOT:-}" && -n "${RF_DETR_DLC:-}" ]] || die '缺少 QAIRT_ROOT / RF_DETR_DLC。'
check_hash "$RF_DETR_DLC" "$base/models/rf_detr/model.dlc"
check_hash "$PROJECT_ROOT/artifacts/rf_detr/input/image.raw" "$base/input/rf_detr/image.raw"
check_hash "$PROJECT_ROOT/artifacts/rf_detr/input/input_list.device.txt" "$base/input/rf_detr/input_list.txt"
for name in libQnnHtp.so libQnnSystem.so libQnnModelDlc.so libQnnHtpPrepare.so libQnnHtpV79Stub.so; do
    check_hash "$QAIRT_ROOT/lib/aarch64-android/$name" "$RUNTIME_DIR/$name"
done
check_hash "$QAIRT_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "$RUNTIME_DIR/libQnnHtpV79Skel.so"
for name in qnn-net-run qnn-context-binary-generator qnn-profile-viewer; do
    check_hash "$QAIRT_ROOT/bin/aarch64-android/$name" "$RUNTIME_DIR/$name"
done
{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    printf 'mode=%s\nnum_inferences=%s\nqairt_root=%s\nremote=%s\noutput_dtype=native\n' "$mode" "$count" "$QAIRT_ROOT" "$remote"
    "${ADB[@]}" shell 'getprop ro.product.model; getprop ro.board.platform; getprop ro.build.fingerprint'
    "${ADB[@]}" shell "$env_cmd '$RUNTIME_DIR/qnn-net-run' --version"
} > "$result/environment.txt" 2>&1

context=""
if [[ "$mode" == context || "$mode" == cpp ]]; then
    [[ -f "$local_root/latest-build-context.txt" ]] || die '先运行 make context-build。'
    context_result="$(< "$local_root/latest-build-context.txt")"
    [[ "$context_result" =~ ^build-context\.[a-zA-Z0-9]+$ ]] || die 'context 记录异常。'
    context="$remote_root/$context_result/rf_detr.bin"
    check_hash "$local_root/$context_result/rf_detr.bin" "$context"
    # 防止更换 DLC / SDK 后误用旧 binary；输入变更可以运行，但 compare 会检查。
    for name in model.dlc libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so libQnnHtpV79Skel.so; do
        old="$(awk -v name="/$name" 'index($2,name)==length($2)-length(name)+1 {print $1}' "$local_root/$context_result/sha256.txt")"
        new="$(awk -v name="/$name" 'index($2,name)==length($2)-length(name)+1 {print $1}' "$result/sha256.txt")"
        [[ -n "$old" && "$old" == "$new" ]] || die 'DLC 或 runtime 已变更，请重新 context-build。'
    done
    printf '%s\n' "$context_result" > "$result/context-source.txt"
fi

case "$mode" in
    dlc)
        command="$env_cmd '$RUNTIME_DIR/qnn-net-run' --backend '$RUNTIME_DIR/libQnnHtp.so' --model '$RUNTIME_DIR/libQnnModelDlc.so' --dlc_path '$base/models/rf_detr/model.dlc'" ;;
    build-context)
        command="$env_cmd '$RUNTIME_DIR/qnn-context-binary-generator' --backend '$RUNTIME_DIR/libQnnHtp.so' --model '$RUNTIME_DIR/libQnnModelDlc.so' --dlc_path '$base/models/rf_detr/model.dlc' --binary_file rf_detr --output_dir '$remote' --profiling_level basic --log_level verbose" ;;
    context)
        command="$env_cmd '$RUNTIME_DIR/qnn-net-run' --backend '$RUNTIME_DIR/libQnnHtp.so' --retrieve_context '$context'" ;;
    cpp)
        exe="$PROJECT_ROOT/artifacts/cpp/qnn-context-runner"
        [[ -s "$exe" ]] || die '先运行 make cpp-build。'
        # executable 与本次结果共处独立目录，不覆盖正在运行的程序。
        "${ADB[@]}" push "$exe" "$remote/qnn-context-runner"
        check_hash "$exe" "$remote/qnn-context-runner"
        "${ADB[@]}" shell "chmod +x '$remote/qnn-context-runner'"
        command="$env_cmd '$remote/qnn-context-runner' --backend '$RUNTIME_DIR/libQnnHtp.so' --system '$RUNTIME_DIR/libQnnSystem.so' --context '$context' --input 'image=$base/input/rf_detr/image.raw' --output-dir '$remote' --runs '$count'" ;;
esac
if [[ "$mode" == dlc || "$mode" == context ]]; then
    command+=" --input_list '$base/input/rf_detr/input_list.txt' --output_dir '$remote' --num_inferences '$count' --keep_num_outputs '$count' --use_native_input_files --use_native_output_files --profiling_level basic --log_level verbose"
fi
printf '%s\n' "$command" | tee "$result/command.txt"
# pipeline 配合 pipefail 保留 adb 的真实错误。失败不更新 latest 指针。
if "${ADB[@]}" shell "$command" 2>&1 | tee "$result/run.log"; then
    printf '0\n' > "$result/exit-code.txt"
else
    status=$?
    printf '%s\n' "$status" > "$result/exit-code.txt"
    die "阶段失败，日志保留在 $result/run.log"
fi
if [[ "$mode" != cpp ]]; then
    "${ADB[@]}" shell "$env_cmd '$RUNTIME_DIR/qnn-profile-viewer' --input_log '$remote/qnn-profiling-data.log' --output '$remote/profile.csv'" > "$result/profile.txt" 2> "$result/profile-stderr.txt"
fi
"${ADB[@]}" pull "$remote/." "$result/"
if [[ "$mode" != cpp ]]; then
    # QAIRT 2.45 的主 log 是 symlink；adb pull 目录会跳过它，显式读取其目标。
    "${ADB[@]}" exec-out "cat '$remote/qnn-profiling-data.log'" > "$result/qnn-profiling-data.log"
    check_hash "$result/qnn-profiling-data.log" "$remote/qnn-profiling-data.log"
fi
if [[ "$mode" == build-context ]]; then
    check_hash "$result/rf_detr.bin" "$remote/rf_detr.bin"
else
    for ((i=0; i<count; i++)); do
        for name in boxes logits classes; do
            file="$name.raw"
            # net-run 为非 float 的 native 输出添加后缀；C++ 直接采用 tensor 名。
            if [[ "$name" == classes && "$mode" != cpp ]]; then file=classes_native.raw; fi
            check_hash "$result/Result_$i/$file" "$remote/Result_$i/$file"
        done
    done
fi
printf '%s\n' "${remote##*/}" > "$local_root/latest-$mode.txt"
printf '[OK] %s 完成: %s\n' "$mode" "$result"
