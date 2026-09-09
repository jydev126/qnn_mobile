#!/usr/bin/env bash
set -euo pipefail

printf '尚未实现：%s (MODEL=%s)。当前仅完成工程初始化；未执行模型准备或手机操作。\n' "${1:?缺少命令名}" "${MODEL:-rf_detr}" >&2
exit 2
