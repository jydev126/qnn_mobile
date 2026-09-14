#!/usr/bin/env python3
"""Reconstruct Qualcomm-added BEVFormer files from the official patch.

This script does not vendor Qualcomm's third-party source into this repository.
It reads the user's local checkout of `qualcomm/ai-hub-models` and reconstructs
files that the unified diff adds from `/dev/null`.

Example:
    python scripts/extract-qualcomm-bevformer-patch-files.py \
      /path/to/ai-hub-models/src/qai_hub_models/models/bevformer/external_repos/bevformertiny_minimal.diff \
      --output-dir artifacts/qualcomm_bevformer

Default extracted files:
    projects/mmdet3d_plugin/bevformer/modules/deformable_attention.py
    projects/mmdet3d_plugin/bevformer/modules/MultiheadAttention.py
    projects/mmdet3d_plugin/custom_utils.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


DEFAULT_TARGETS = (
    "projects/mmdet3d_plugin/bevformer/modules/deformable_attention.py",
    "projects/mmdet3d_plugin/bevformer/modules/MultiheadAttention.py",
    "projects/mmdet3d_plugin/custom_utils.py",
)


@dataclass
class DiffFile:
    path: str
    is_new_file: bool = False
    source_is_dev_null: bool = False
    lines: list[str] | None = None


def _safe_relative_path(path: str) -> PurePosixPath:
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe diff path: {path!r}")
    return p


def parse_new_files(diff_text: str) -> dict[str, str]:
    """Return exact contents for files added from /dev/null in a unified diff."""
    result: dict[str, str] = {}
    current: DiffFile | None = None
    in_hunk = False

    def finish() -> None:
        nonlocal current, in_hunk
        if (
            current is not None
            and current.is_new_file
            and current.source_is_dev_null
            and current.lines is not None
        ):
            result[current.path] = "".join(current.lines)
        current = None
        in_hunk = False

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git a/"):
            finish()
            # `diff --git a/<old> b/<new>`; paths in this Qualcomm patch do not
            # contain spaces. Use the b/ side as the reconstructed target path.
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 4 or not parts[3].startswith("b/"):
                raise ValueError(f"unexpected diff header: {line.rstrip()}")
            current = DiffFile(path=parts[3][2:], lines=[])
            _safe_relative_path(current.path)
            continue

        if current is None:
            continue

        if line.startswith("new file mode "):
            current.is_new_file = True
            continue

        if line.startswith("--- "):
            current.source_is_dev_null = line.strip() == "--- /dev/null"
            continue

        if line.startswith("+++ "):
            continue

        if line.startswith("@@ "):
            in_hunk = True
            continue

        if line.startswith("\\ No newline at end of file"):
            continue

        if not in_hunk:
            continue

        # A truly new file has only added lines inside hunks. Keep an empty
        # added line as "\n" and preserve the exact line endings from the diff.
        if line.startswith("+"):
            assert current.lines is not None
            current.lines.append(line[1:])
        elif line.startswith("-"):
            raise ValueError(f"new-file section unexpectedly contains deletion: {current.path}")
        elif line.startswith(" "):
            # Context lines are unusual for a file added from /dev/null, but
            # preserving them makes the parser robust to hand-edited patches.
            assert current.lines is not None
            current.lines.append(line[1:])

    finish()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("diff", type=Path, help="path to Qualcomm bevformertiny_minimal.diff")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/qualcomm_bevformer"),
        help="root directory for reconstructed files",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="target path inside the patch; repeat to override the default target set",
    )
    parser.add_argument(
        "--all-new-files",
        action="store_true",
        help="extract every file added from /dev/null instead of the default target set",
    )
    args = parser.parse_args()

    diff_path: Path = args.diff.resolve()
    if not diff_path.is_file():
        raise FileNotFoundError(diff_path)

    files = parse_new_files(diff_path.read_text(encoding="utf-8"))

    if args.all_new_files:
        selected = sorted(files)
    else:
        selected = list(args.targets or DEFAULT_TARGETS)

    missing = [path for path in selected if path not in files]
    if missing:
        available = "\n  ".join(sorted(files))
        raise RuntimeError(
            "requested paths were not found as complete new files in the patch:\n  "
            + "\n  ".join(missing)
            + "\navailable new files:\n  "
            + available
        )

    out_root: Path = args.output_dir.resolve()
    for source_path in selected:
        rel = _safe_relative_path(source_path)
        output_path = out_root.joinpath(*rel.parts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(files[source_path], encoding="utf-8")
        print(f"extracted {source_path} -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
