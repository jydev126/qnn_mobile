"""Prepare RF-DETR small input using the local Qualcomm AI Hub recipe.

Recipe: external/ai-hub-models/src/qai_hub_models/models/rf_detr/model.py
RF_DETR.get_input_spec: image, (1, 3, 512, 512), float32, RGB, [0, 1].
templates/detr/app.py: DETRApp.predict uses PIL Resampling.BILINEAR.
utils/image_processing.py: preprocess_PIL_image uses NCHW float / 255.
RF_DETR.forward applies mean/std normalization inside the model.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error("需要 Python 3.11。")

    with Image.open(args.image) as source:
        resized = source.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    # Explicit little-endian float32, contiguous NCHW, no header or mean/std.
    tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype="<f4")
    if tensor.shape != (1, 3, 512, 512) or tensor.dtype != np.dtype("<f4"):
        raise ValueError("输入 shape/dtype 不符合 recipe。")
    if not np.isfinite(tensor).all() or not (0 <= tensor.min() <= tensor.max() <= 1):
        raise ValueError("输入必须为有限值且在 [0,1] 内。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "image.raw"
    tensor.tofile(raw_path)
    byte_count = raw_path.stat().st_size
    if byte_count != 3145728:
        raise ValueError(f"raw 文件字节数错误: {byte_count}, expected 3145728")
    # Relative path: run qnn-net-run with this directory as its working directory.
    list_path = args.output_dir / "input_list.txt"
    list_path.write_text("image:=image.raw\n", encoding="utf-8")
    print("input tensor: image (依据 Qualcomm RF_DETR.get_input_spec；未读取 DLC metadata)")
    print(f"shape: {list(tensor.shape)}")
    print(f"dtype: {tensor.dtype}")
    print(f"min: {tensor.min()}")
    print(f"max: {tensor.max()}")
    print(f"文件字节数: {byte_count} bytes")
    print(f"raw: {raw_path}")
    print(f"input_list: {list_path} (image:=image.raw；运行目录须为输入文件所在目录)")


if __name__ == "__main__":
    main()
