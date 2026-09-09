#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ ! -f "$PROJECT_ROOT/config/local.env" ]] || source "$PROJECT_ROOT/config/local.env"
die() { printf '[FAIL] %s\n' "$*" >&2; exit 1; }
[[ -n "${QAIRT_ROOT:-}" && -f "$QAIRT_ROOT/include/QNN/QnnInterface.h" ]] || die '缺少 QAIRT_ROOT / QNN headers。'
[[ -n "${ANDROID_NDK_ROOT:-}" ]] || die '请手动准备 Android NDK，在 config/local.env 填写 ANDROID_NDK_ROOT。'
api="${ANDROID_API:-28}"
[[ "$api" =~ ^[0-9]+$ ]] && (( api >= 28 )) || die 'ANDROID_API 需为 >= 28 的整数。'
compiler="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${api}-clang++"
[[ -x "$compiler" ]] || die "缺少交叉编译器: $compiler"
build_dir="$PROJECT_ROOT/artifacts/cpp"
mkdir -p "$build_dir"
"$compiler" --version | tee "$build_dir/compiler.txt"
# 只动态加载 QNN；静态链接 C++ 标准库，避免额外部署 libc++_shared.so。
"$compiler" -std=c++17 -O2 -g -Wall -Wextra -Werror -static-libstdc++ \
    -I "$QAIRT_ROOT/include/QNN" "$PROJECT_ROOT/cpp/qnn_context_runner.cpp" \
    -ldl -o "$build_dir/qnn-context-runner"
readelf -h "$build_dir/qnn-context-runner"
readelf -d "$build_dir/qnn-context-runner"
sha256sum "$PROJECT_ROOT/cpp/qnn_context_runner.cpp" \
    "$QAIRT_ROOT/include/QNN/QnnInterface.h" "$build_dir/qnn-context-runner" > "$build_dir/build-sha256.txt"
printf '[OK] Android aarch64 executable: %s\n' "$build_dir/qnn-context-runner"
