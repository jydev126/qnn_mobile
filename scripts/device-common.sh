#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_ROOT/config/local.env" ]]; then
    source "$PROJECT_ROOT/config/local.env"
fi
die() { printf '[FAIL] %s\n' "$*" >&2; exit 1; }
command -v adb >/dev/null || die '未找到 adb。'
devices="$(adb devices)"
if [[ -z "${DEVICE_SERIAL:-}" ]]; then
    mapfile -t serials < <(awk '$2 == "device" { print $1 }' <<< "$devices")
    [[ ${#serials[@]} == 1 ]] || die '需要恰好一台在线设备，或在 config/local.env 设置 DEVICE_SERIAL。'
    DEVICE_SERIAL="${serials[0]}"
fi
ADB=(adb -s "$DEVICE_SERIAL")
[[ "$("${ADB[@]}" get-state)" == device ]] || die '指定设备未就绪。'
printf 'Device: %s\n' "$DEVICE_SERIAL"
RUNTIME_DIR=/data/local/tmp/qnn_mobile/runtime
