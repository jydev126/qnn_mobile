# QNN 文档索引

建议按下面顺序阅读。

## 基础心智模型

1. [`qnn-mental-model.md`](qnn-mental-model.md)：QNN / QAIRT / HTP / Hexagon / Stub / Skel 分工，并与 CUDA/TensorRT 做近似类比。
2. [`qnn-artifact-map.md`](qnn-artifact-map.md)：`.pth / .onnx / .dlc / .bin / .so / executable / .raw` 在整条链路中的位置。
3. [`qnn-tensor-memory.md`](qnn-tensor-memory.md)：tensor metadata、dtype/shape、client buffer、CPU/HTP memory contract。

## 从现成 DLC 走向自己迁模型

4. [`qnn-model-porting.md`](qnn-model-porting.md)：PyTorch 开源模型到 QNN DLC 的拆分、静态化、数值验证方法。
5. [`qnn-operator-support.md`](qnn-operator-support.md)：GridSample、DeformableAttention、Scatter、动态 shape 等特殊算子的处理方法。
6. [`qnn-profiling.md`](qnn-profiling.md)：INIT / Finalize / Execute / accelerator time 的性能口径。

## UniAD 专项

7. [`uniad-qnn-operator-reference.md`](uniad-qnn-operator-reference.md)：逐模块对照 UniAD stage2 与 Qualcomm BEVFormer/RF-DETR 的 QNN 算子参考。

当前最推荐的实验顺序：

```text
UniAD TemporalSelfAttention standalone
→ Detection Decoder single-level MSDA
→ 4-level MSDeformableAttention3D core
→ SpatialCrossAttention rebatch/scatter
→ 一层 BEVFormerEncoder
→ 六层 encoder
→ backbone/neck（单独处理 DCNv2）
→ 其他 heads
```

Qualcomm BEVFormer patch 中新增文件可用：

```bash
python scripts/extract-qualcomm-bevformer-patch-files.py \
  /path/to/ai-hub-models/src/qai_hub_models/models/bevformer/external_repos/bevformertiny_minimal.diff
```

默认重建 `deformable_attention.py`、`MultiheadAttention.py` 和 `custom_utils.py` 到 `artifacts/qualcomm_bevformer/`，不把第三方源码直接提交进本仓库。