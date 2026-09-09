#!/usr/bin/env bash
set -euo pipefail

[[ "${MODEL:-rf_detr}" == rf_detr ]] || { printf '不支持的 MODEL: %s\n' "$MODEL" >&2; exit 1; }
source "$(dirname -- "${BASH_SOURCE[0]}")/device-common.sh"
[[ -n "${RF_DETR_DLC:-}" ]] || die 'RF_DETR_DLC 未设置。'
[[ "$RF_DETR_DLC" == *.dlc && -s "$RF_DETR_DLC" ]] || die "DLC 不存在、为空或扩展名错误: $RF_DETR_DLC"
raw="$PROJECT_ROOT/artifacts/rf_detr/input/image.raw"
[[ -f "$raw" ]] || die '缺少 image.raw，请先执行 make prepare MODEL=rf_detr。'
[[ "$(stat -c %s "$raw")" == 3145728 ]] || die 'image.raw 必须为 3145728 bytes，请重新 prepare。'

model_dir=/data/local/tmp/qnn_mobile/models/rf_detr
input_dir=/data/local/tmp/qnn_mobile/input/rf_detr
output_dir=/data/local/tmp/qnn_mobile/output/rf_detr
# 单独生成设备列表，保留 prepare 生成的 host 列表。
device_list="$PROJECT_ROOT/artifacts/rf_detr/input/input_list.device.txt"
printf 'image:=%s/image.raw\n' "$input_dir" > "$device_list"

"${ADB[@]}" shell "mkdir -p '$model_dir' '$input_dir' '$output_dir'"
"${ADB[@]}" push "$RF_DETR_DLC" "$model_dir/model.dlc"
"${ADB[@]}" push "$raw" "$input_dir/image.raw"
"${ADB[@]}" push "$device_list" "$input_dir/input_list.txt"

# 比对实际传输内容，而不仅检查 adb push 返回值。
verify_file() {
    local host_hash device_hash
    host_hash="$(sha256sum < "$1")"
    host_hash="${host_hash%% *}"
    device_hash="$("${ADB[@]}" shell "sha256sum '$2'")"
    device_hash="${device_hash%% *}"
    [[ "$host_hash" == "$device_hash" ]] || die "SHA256 不匹配: $2"
    printf '[OK] SHA256 %s\n' "$2"
}
verify_file "$RF_DETR_DLC" "$model_dir/model.dlc"
verify_file "$raw" "$input_dir/image.raw"
verify_file "$device_list" "$input_dir/input_list.txt"
printf '\n设备 input_list.txt:\n'
"${ADB[@]}" shell "cat '$input_dir/input_list.txt'"
printf '\n部署完成；输出目录: %s；未执行推理。\n' "$output_dir"
