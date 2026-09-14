# QNN 学习文档索引

这组文档现在按“先建立硬件/runtime 心智模型，再看 tensor/产物，再做模型迁移和算子实验”的顺序组织。它们都以当前 `qnn_mobile` 的 RF-DETR + S25 Ultra 实测为落点，不把抽象概念和 Qualcomm 内部实现混在一起。

## 推荐阅读顺序

```text
1. qnn-mental-model.md
   QNN / HTP / Hexagon / HVX / HMX / Stub / Skel
   以及和 CUDA/TensorRT 的正确层级类比

2. qnn-artifact-map.md
   .pth/.onnx/.dlc/.bin/.so/executable/.raw
   谁生成、谁读取、在哪侧使用、是否 SoC-specific

3. qnn-tensor-memory.md
   Qnn_Tensor_t 与 client buffer
   当前 std::vector/QNN_TENSORMEMTYPE_RAW 路径
   system LPDDR vs HTP local memory 的边界

4. lifecycle-guide.md
   当前已经跑通的 DLC → Compose → Finalize → Context → Execute

5. qnn-profiling.md
   用当前 RF-DETR 的真实 10 s Finalize / ~60 ms restore / ~160 ms Execute
   理解 INIT、API wall、accelerator event

6. qnn-model-porting.md
   PyTorch/checkpoint → export-friendly graph → QNN DLC
   以及 AI Hub Workbench / 本地 QAIRT 的职责边界

7. qnn-operator-support.md
   GridSample / MSDeformableAttention / DCNv2 / ScatterND / NonZero
   Qualcomm 已有源码参考与具体 decomposition

8. uniad-qnn-operator-reference.md
   把 UniAD stage2 的 TSA/SCA/DCNv2/decoder/seg/motion/planning
   逐模块映射到 Qualcomm BEVFormer / Mask2Former / CenterNet / RF-DETR
```

## 两条主线

### QNN runtime 主线

```text
qnn-mental-model
→ qnn-artifact-map
→ qnn-tensor-memory
→ lifecycle-guide
→ qnn-profiling
```

目标：知道已经生成好的 DLC/Context 怎么在 Android ARM CPU + HTP 上运行，并能正确解释内存和性能。

### UniAD 模型迁移主线

```text
qnn-model-porting
→ qnn-operator-support
→ uniad-qnn-operator-reference
```

目标：把问题从“UniAD 能不能上 QNN”变成一系列可验证的小问题：

```text
DCNv2 能否用 CenterNet-style decomposition 对齐？
TSA 能否映射到 Qualcomm BEVFormer optimized 实现？
4-level MSDA 能否组合 Mask2Former/RF-DETR core？
SCA 的 dynamic camera rebatch 怎么变成 fixed shape + mask？
Planning 哪些留 QNN，哪些明确留 CPU？
```

## 下一步实验顺序

当前文档建议直接进入代码实验：

```text
0. GridSample primitive
1. UniAD DCNv2 vs Qualcomm-style custom_deformconv2d
2. UniAD TemporalSelfAttention single-level
3. Detection decoder single-level MSDA
4. 4-level MSDeformableAttention3D core
5. SpatialCrossAttention fixed rebatch + mask + scatter
6. 一层 BEV encoder
7. 完整 BEV encoder / state
```

每个实验沿用同一验收链：

```text
Original PyTorch
→ Patched PyTorch numeric compare
→ export
→ QNN compile/DLC
→ HTP Finalize
→ HTP Execute
→ numeric + latency + memory
```

这样文档最终会沉淀成一套可复用的 QNN/UniAD 部署实验记录，而不是单纯的概念笔记。
