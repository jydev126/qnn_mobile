#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "${MODEL:-rf_detr}" == rf_detr ]] || { printf '不支持 MODEL\n' >&2; exit 1; }
if [[ -f "$PROJECT_ROOT/config/local.env" ]]; then source "$PROJECT_ROOT/config/local.env"; fi
: "${RF_DETR_INPUT_IMAGE:?请设置 RF_DETR_INPUT_IMAGE}"
result="$(cat "$PROJECT_ROOT/output/rf_detr/latest-result.txt")"
exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/models/rf_detr/decode.py" --result "$result" --image "$RF_DETR_INPUT_IMAGE" --threshold "${SCORE_THRESHOLD:-0.5}"
