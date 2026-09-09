#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$PROJECT_ROOT/output/rf_detr"
log="$PROJECT_ROOT/output/rf_detr/qnn-net-run.log"
# Preserve the previous attempt, including failures, before opening the new log.
if [[ -f "$log" ]]; then
    cp "$log" "$(mktemp "$PROJECT_ROOT/output/rf_detr/qnn-net-run.previous.XXXXXX.log")"
fi
run_model() {
    [[ "${MODEL:-rf_detr}" == rf_detr ]] || { printf '不支持的 MODEL: %s\n' "$MODEL" >&2; return 1; }
    source "$PROJECT_ROOT/scripts/device-common.sh"
    local base=/data/local/tmp/qnn_mobile
    local env_cmd="LD_LIBRARY_PATH='$RUNTIME_DIR' ADSP_LIBRARY_PATH='$RUNTIME_DIR'"
    local help_text
    if help_text="$("${ADB[@]}" shell "$env_cmd '$RUNTIME_DIR/qnn-net-run' --help" 2>&1)"; then
        :
    else
        printf '%s\n' "$help_text"
        die '设备 --help 启动失败，请先检查 adb、动态库和库路径。'
    fi
    printf '%s\n' "$help_text"
    for option in --backend --model --dlc_path --input_list --output_dir --log_level; do
        [[ "$help_text" == *"$option"* ]] || die "设备 --help 缺少参数: $option"
    done
    # A fresh directory prevents old results from being mistaken for this run.
    local result_dir
    "${ADB[@]}" shell "mkdir -p '$base/output/rf_detr'"
    result_dir="$("${ADB[@]}" shell "mktemp -d '$base/output/rf_detr/run.XXXXXX'")"
    result_dir="${result_dir//$'\r'/}"
    [[ "$result_dir" =~ ^/data/local/tmp/qnn_mobile/output/rf_detr/run\.[a-zA-Z0-9]+$ ]] || die '设备输出目录异常。'
    printf 'LD_LIBRARY_PATH=%s\nADSP_LIBRARY_PATH=%s\noutput_dir=%s\n' "$RUNTIME_DIR" "$RUNTIME_DIR" "$result_dir"
    "${ADB[@]}" shell "$env_cmd '$RUNTIME_DIR/qnn-net-run' --backend '$RUNTIME_DIR/libQnnHtp.so' --model '$RUNTIME_DIR/libQnnModelDlc.so' --dlc_path '$base/models/rf_detr/model.dlc' --input_list '$base/input/rf_detr/input_list.txt' --output_dir '$result_dir' --log_level verbose"
    printf '[OK] qnn-net-run inference exited 0; output_dir=%s\n' "$result_dir"
    "${ADB[@]}" shell "find '$result_dir' -type f"
}
# Run in a subshell so errexit remains active; tee captures all stdout/stderr.
set +e
(set -e; run_model) 2>&1 | tee "$log"
statuses=("${PIPESTATUS[@]}")
set -e
printf 'qnn-net-run task exit code: %s\n' "${statuses[0]}" | tee -a "$log"
(( statuses[1] == 0 )) || exit "${statuses[1]}"
exit "${statuses[0]}"
