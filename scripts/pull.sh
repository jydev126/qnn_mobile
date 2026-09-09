#!/usr/bin/env bash
set -euo pipefail
[[ "${MODEL:-rf_detr}" == rf_detr ]] || { printf '不支持 MODEL\n' >&2; exit 1; }
source "$(dirname -- "${BASH_SOURCE[0]}")/device-common.sh"
log="$PROJECT_ROOT/output/rf_detr/qnn-net-run.log"
[[ -f "$log" ]] || die '缺少运行日志。'
remote="$(sed -n 's/^\[OK\] qnn-net-run inference exited 0; output_dir=//p' "$log" | tail -n 1)"
[[ "$remote" =~ ^/data/local/tmp/qnn_mobile/output/rf_detr/run\.[a-zA-Z0-9]+$ ]] || die '日志未记录有效成功运行。'
local_dir="$PROJECT_ROOT/output/rf_detr/${remote##*/}"
mkdir -p "$local_dir"
"${ADB[@]}" pull "$remote/." "$local_dir/"
for file in execution_metadata.yaml Result_0/boxes.raw Result_0/classes.raw Result_0/logits.raw; do
    local_hash="$(sha256sum < "$local_dir/$file")"
    remote_hash="$("${ADB[@]}" shell "sha256sum '$remote/$file'")"
    [[ "${local_hash%% *}" == "${remote_hash%% *}" ]] || die "SHA256 不匹配: $file"
done
printf '%s\n' "$local_dir" > "$PROJECT_ROOT/output/rf_detr/latest-result.txt"
printf '[OK] 拉取及 SHA256 校验通过: %s\n' "$local_dir"
