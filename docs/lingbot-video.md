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

- Main activations are BF16. Router input/logits/sigmoid scores are FP32. The default scatter path casts expert outputs to FP32 before weighting; that main value is counted for the route-weight gradient.
- Attention uses the existing FA abstraction: Q/K/V/O retained, LSE excluded. RMSNorm uses an ideal fused abstraction retaining input; the source's unfused FP32 internal tensors/statistics are excluded.
- Routed and shared MLPs use the same fused SwiGLU abstraction as H3: backward retains gate/up and reconstructs SiLU(gate). No separate SiLU output is counted. The public eager implementation uses separate SiLU and multiplication, so its retention can be higher than this estimate.
- Group-limited selection produces INT64 top-k indices `[T, 8]`; gather/normalization/scaling produces BF16 weights `[T, 8]`. Selection itself has no gradient. Gather/normalization retains scores and indices; dispatch needs the index/permutation mapping, and weighted combine needs indices, weights and expert values. The logical index tensor is counted once (0.25 MiB at T=4096); concrete permutation/group/count buffers are excluded. Recomputing selection assumes deterministic replay, including tie-breaking.
- Weighted combine retains expert values and routing weights. Pure addition needs no input values for backward. Both post-attention and post-FFN norms are included.
- Count active assignments without expert alignment padding, capacity, dropped tokens or sequence padding. Reordered copies, auxiliary normalization reductions, additional routing permutation/count buffers, communication and temporary workspaces are excluded. No expert parallelism or offload is modeled.
- External conditioning (token-wise scales, shifts, gates and their tanh state), positional constants and weights must remain available. Patch/text embedding, final projection, refiner and the other 47 layers are outside scope.

All four recomputation levels use the shared dependency engine. Full recomputation retains only the layer input **within this ledger**. Tests cover checkpoint dimensions, mixed precision, shared/routed volumes, all eight policy combinations and finite-difference router/weighted-sum gradients. They do not execute official GPU kernels.

## DeepEP comparison

The graph above follows LingBot's logical scatter path, **not a DeepEP memory estimate**. The [DeepEP training example](https://github.com/deepseek-ai/DeepEP#interfaces-and-examples) (EPBuffer API, checked 2026-11-10) saves an EPHandle from dispatch for both backward operations. The original top-k indices must stay unchanged while the handle is in use. The handle also contains communication/layout metadata, so its size cannot be represented by the index tensor alone.

In the expanded layout, dispatch delivers token rows and FP32 routing weights. Expert computation applies the weights before combine; combine receives BF16 weighted expert outputs and sums them. Combine backward uses cached dispatch; dispatch backward uses combine, including the routing-weight gradients. Weight multiplication backward still needs the unweighted expert output and routing weights. A standalone FP32 expert-output tensor is not required by the communication interface. None of these distributed buffers, receiver-side copies, padding or handles are included in this graph.
