#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_ROOT/config/local.env" ]]; then
    source "$PROJECT_ROOT/config/local.env"
fi
[[ "${MODEL:-rf_detr}" == rf_detr ]] || { printf '不支持的 MODEL: %s\n' "$MODEL" >&2; exit 1; }
[[ -n "${RF_DETR_INPUT_IMAGE:-}" ]] || { printf 'RF_DETR_INPUT_IMAGE 未设置。\n' >&2; exit 1; }
python_cmd="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$python_cmd" ]] || { printf '缺少项目 Python 3.11 venv，请手动准备 .venv 及 numpy、Pillow。\n' >&2; exit 1; }
cd "$PROJECT_ROOT"
exec "$python_cmd" models/rf_detr/prepare.py --image "$RF_DETR_INPUT_IMAGE" --output-dir "$PROJECT_ROOT/artifacts/rf_detr/input"
