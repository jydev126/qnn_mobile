#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/device-common.sh"

for prop in ro.product.model ro.board.platform ro.hardware; do
    printf '%s: %s\n' "$prop" "$("${ADB[@]}" shell getprop "$prop")"
done
printf 'Android version: %s\n' "$("${ADB[@]}" shell getprop ro.build.version.release)"
printf 'API level: %s\n' "$("${ADB[@]}" shell getprop ro.build.version.sdk)"
printf 'SELinux: %s\n' "$("${ADB[@]}" shell getenforce)"
