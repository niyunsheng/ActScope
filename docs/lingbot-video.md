# LingBot Video MoE layer

## Sources

- [Official transformer implementation](https://github.com/Robbyant/lingbot-video/blob/dd5c231e793406c6a8893e9e4307008a9c2adfe4/lingbot_video/transformer_lingbot_video.py): `LingBotVideoBlock`, `LingBotVideoAttention`, `LingBotVideoRouter`, `LingBotVideoGroupedExperts`, `LingBotVideoSparseMoeBlock` and `LingBotVideoMLP`.
- [Released 30B-A3B checkpoint configuration](https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b/blob/f2e538f64afe00cc4ae674db2aeb52e2945edfd5/transformer/config.json), copied into `src/activation/lingbot-config.ts`.

| Parameter | Value |
| --- | --- |
| Hidden size / heads / head dimension | 2048 / 16 / 128 |
| Transformer depth | 48 (only one layer counted) |
| Routed experts / selected per token | 128 / 8 |
| Expert intermediate size | 768 |
| Shared experts / shared intermediate size | 1 / 768 |
| Router | sigmoid, 4 groups, select 2 groups, normalize top-k, scale 2.5 |
| Dense intermediate size | 6144 (not used by this sparse layer) |

Every published layer is sparse (`decoder_sparse_step=1`, no `mlp_only_layers`). Joint attention operates on video and text tokens; T is their combined count, not frames or video tokens alone. The shared expert always runs in addition to the eight routed experts. Routed activation volume follows T×8 assignments, not T×128 experts.

## Accounting contract

This is a **logical training activation estimate**, not an exact PyTorch saved-tensor trace or a measured GPU peak. The public source primarily implements inference. Actual training retention depends on kernels and checkpoint boundaries.

- Main activations are BF16; router arithmetic and DeepEP routing weights are FP32. Unlike the source scatter backend, the graph now models a DeepEP EPBuffer integration. This is a proposed execution abstraction, not a claim that the published LingBot source uses DeepEP.
- Attention retains Q/K/V/O, excluding LSE. RMSNorm retains its input, excluding statistics. Both MLP branches use fused SwiGLU, retaining gate/up rather than a separate SiLU output.
- Group-limited selection produces INT64 indices `[T, 8]`. Gather/normalization/scaling produces FP32 routing weights `[T, 8]` for DeepEP (the original LingBot router casts these to the token dtype). Selection has no gradient; normalization needs scores and indices.
- `deepep_dispatch` takes tokens, indices and weights. It transports weights without multiplying token values. Outputs shown are BF16 expert rows `[R, H]`, FP32 received weights `[R]`, and an opaque EP handle. The optional received expert-index tensor is omitted; grouped-layout metadata is represented by the handle.
- `apply_route_weight` multiplies each expert output by its received weight and emits BF16. Its fused multiply/cast backward retains the BF16 expert output and FP32 weight; an explicit FP32 expert-output activation is unnecessary. FP32 arithmetic inside the fused operation is not a retained tensor.
- `deepep_combine` sums already-weighted BF16 outputs using the handle. Combine backward dispatches gradients using the cached handle. Dispatch backward combines token and weight gradients with that same handle. Neither communication backward needs the original token/weight values.
- The handle and its original top-k indices stay alive across all recomputation levels. Indices are counted once (0.25 MiB at T=4096); the handle is displayed as metadata with **unknown/excluded storage**, not a zero-byte allocation. Full recomputation therefore retains the layer input, indices and handle; totals include only the first two. Replaying dispatch entails communication and must preserve the forward routing/layout; this model does not simulate scheduling or its cost.
- R = T×8 is the unpadded **logical whole-layer** assignment volume, not actual received rows on an EP rank. Per-rank load, padding, capacity, communication buffers, additional permutations/counts and temporary workspaces are not estimated. Received weights are a separate copy from the source weights; source weights are not retained for backward when only the handle is needed.
- External conditioning, positional constants and weights remain outside scope. Patch/text embedding, final projection, refiner and the other 47 layers are excluded. This is saved activation storage, not GPU peak memory.

## DeepEP source

The [official EPBuffer training example](https://github.com/deepseek-ai/DeepEP#interfaces-and-examples), checked 2026-11-10, specifies the expanded layout, FP32 routing weights, BF16 combine input and saved-handle backward helpers. This graph targets that interface, not the legacy low-latency weighted-combine API.

Tests cover mixed precision, index lifetime, handle exclusion, all eight recomputation policies and finite-difference router/weighting gradients. They do not execute DeepEP GPU kernels.
