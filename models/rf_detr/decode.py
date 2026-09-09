"""Decode the verified RF-DETR end-to-end recipe outputs, without extra sigmoid/NMS."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('threshold 必须在 [0,1]')
    metadata = yaml.safe_load((args.result / 'execution_metadata.yaml').read_text())
    if metadata['inferences_completed'] != 1:
        raise ValueError('仅支持当前单图一次推理')
    native = '--use_native_output_files' in metadata['command']
    specs = {s['tensor_name']: s for s in metadata['graphs'][0]['output_tensors']}
    arrays = {}
    for name, shape in [('boxes', [1, 300, 4]), ('logits', [1, 300]), ('classes', [1, 300])]:
        if specs[name]['dimensions'] != shape:
            raise ValueError(f'不支持的输出 shape: {name}')
        dtype = '<i4' if native and name == 'classes' else '<f4'
        values = np.fromfile(args.result / 'Result_0' / f'{name}.raw', dtype=dtype)
        if values.size != np.prod(shape) or not np.isfinite(values).all():
            raise ValueError(f'输出大小/有限值检查失败: {name}')
        arrays[name] = values.reshape(shape)[0]
    boxes, scores, classes = (arrays[n] for n in ('boxes', 'logits', 'classes'))
    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('分数超出 [0,1]，需重新核对 recipe')
    if not np.equal(classes, np.floor(classes)).all():
        raise ValueError('classes 不是整数值')
    classes = classes.astype(np.int32)
    if not (boxes[:, 2:] >= boxes[:, :2]).all():
        raise ValueError('boxes 不符合 xyxy')
    # Read only the literal mapping; do not import the model or download weights.
    label_file = ROOT / 'external/ai-hub-models/src/qai_hub_models/models/templates/detr/coco_label_map.py'
    tree = ast.parse(label_file.read_text())
    labels = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'LABEL_MAP' for t in n.targets))
    with Image.open(args.image) as source:
        original = source.convert('RGB')
    prepared = np.asarray(original.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / np.float32(255)
    expected_raw = np.ascontiguousarray(prepared.transpose(2, 0, 1)[None], dtype='<f4')
    if expected_raw.tobytes() != (ROOT / 'artifacts/rf_detr/input/image.raw').read_bytes():
        raise ValueError('当前图片与 prepare 输入不匹配')
    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    rows = []
    for idx in np.argsort(-scores):
        scaled = boxes[idx] * np.array([original.width / 512, original.height / 512] * 2)
        row = {'index': int(idx), 'class_id': int(classes[idx]),
               'label': labels.get(int(classes[idx]), f'class_{classes[idx]}'),
               'score': float(scores[idx]), 'box_512_xyxy': boxes[idx].tolist(),
               'box_original_xyxy': scaled.tolist()}
        rows.append(row)
        if scores[idx] >= args.threshold:
            x1, y1, x2, y2 = scaled
            draw.rectangle((x1, y1, x2, y2), outline='lime', width=3)
            text = f"{row['label']} {row['score']:.3f}"
            xy = (max(0, min(x1, original.width - 100)), max(0, y1 - 12))
            draw.text(xy, text, fill='yellow', stroke_width=1, stroke_fill='black')
    # Preserve all 300 predictions; threshold is only a presentation setting.
    summary = {'threshold': args.threshold, 'display_count': int((scores >= args.threshold).sum()),
               'score_range': [float(scores.min()), float(scores.max())],
               'raw_storage': 'native' if native else 'float32 (including classes)',
               'postprocess': 'recipe sigmoid scores and pixel xyxy; no extra sigmoid or NMS',
               'detections': rows}
    (args.result / 'detections.json').write_text(json.dumps(summary, indent=2) + '\n')
    annotated.save(args.result / 'detections.png')
    preview = Image.new('RGB', (1024, 542), 'white')
    preview.paste(original.resize((512, 512)), (0, 30))
    preview.paste(annotated.resize((512, 512)), (512, 30))
    title = ImageDraw.Draw(preview)
    title.text((10, 8), 'Original', fill='black')
    title.text((522, 8), f'HTP RF-DETR | score >= {args.threshold} | {summary["display_count"]} detections', fill='black')
    preview.save(args.result / 'comparison.png')
    print(json.dumps({k:v for k,v in summary.items() if k != 'detections'}, indent=2))
    print(args.result / 'comparison.png')


if __name__ == '__main__':
    main()
