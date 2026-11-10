# ActScope

## Supported models

- MiniMax H3 · DiT
- LingBot Video · MoE 30B-A3B
- Ant Ling 3.0 Flash · KDA + MoE, gated MLA + MoE, KDA + dense FFN

## Ling 3.0 Flash

Select **Ant Ling 3.0 Flash**, then choose **Layer type**. The three views cover the
42 decoder layers (1-based):

| Layers | Attention | FFN |
| --- | --- | --- |
| 1–2 | KDA | Dense SwiGLU, intermediate 6144 |
| 6, 12, 18, 24, 30, 36, 42 | Gated MLA | MoE |
| All remaining layers | KDA | MoE |

Each layer is `x → RMSNorm → attention → +x → RMSNorm → FFN → +residual`.
Hidden size is 2560. The model card reports 124B total / 5.1B active parameters;
these are parameter counts, not activation-memory measurements.

- **KDA:** Q/K/V projections from 2560 to 4096, each followed by a causal
  depthwise convolution (kernel 4) and SiLU. The 32 heads have dimension 128.
  `f_proj` supplies a per-channel decay gate; `b_proj` supplies FP32 sigmoid beta
  per head. KDA includes Q/K L2 normalization and the diagonal decay transform
  internally. A per-head RMSNorm and sigmoid output gate from `g_proj` precede
  the 4096 → 2560 output projection. There is no RoPE or softmax matrix here.
- **Gated MLA:** direct Q projection (no Q LoRA), Q/K dimensions 128 non-RoPE +
  64 RoPE = 192, V dimension 128, 32 heads. KV compression produces a rank-512
  latent and a shared 64-dimensional positional key. RMSNorm and KV expansion
  produce the non-positional K and V. Causal attention is followed by a
  head-wise sigmoid gate and a 4096 → 2560 output projection. Both the FP32
  sigmoid result and its BF16 gate are represented for their backward consumers.
- **MoE:** FP32 sigmoid router, 512 routed experts, top-8 selection plus one
  shared expert; each expert is a 768-wide SwiGLU FFN. Bias-adjusted scores select
  4 of 8 groups (group score is the sum of its best two experts) and then top-8
  experts. The weights gather the original, unbiased sigmoid scores, normalize
  them and multiply by 2.5. The routed result is added to the shared branch.

### Sources and scope of the estimate

The [official model card](https://huggingface.co/inclusionAI/Ling-3.0-flash) and
[checkpoint config](https://huggingface.co/inclusionAI/Ling-3.0-flash/blob/fcc60a52b34c69c6271d0ac8a8cb0815020b6a43/config.json)
are cross-checked against the
[official implementation](https://huggingface.co/inclusionAI/Ling-3.0-flash/blob/fcc60a52b34c69c6271d0ac8a8cb0815020b6a43/modeling_bailing_moe_v3.py).
The config and implementation are pinned to `fcc60a52b34c69c6271d0ac8a8cb0815020b6a43`.

This is a **logical saved-main-activation ledger for training**, not measured GPU
memory, inference KV-cache usage, or peak allocated memory. T is token count;
activations use token-major shapes and BF16 unless marked FP32/INT64. We exclude
weights, gradients, optimizer state, normalization statistics, attention LSE,
workspace, padding, communication buffers and embedding/LM/MTP heads.

- KDA counts the inputs to the chunk operation. Its backward abstraction replays
  recurrence from a zero initial state; internal chunk states, transformed gates,
  intermediate matrices and convolution caches are excluded. Actual FLA kernels
  may save additional state even under “No recomputation.” The KDA output remains
  a dependency of the following gated norm. KDA uses the attention recomputation
  tier, labeled **KDA** in the UI; this is not a throughput or hardware-bound claim.
- MLA uses **expanded training Q/K/V and an FA-style backward assumption** that
  saves Q, K, V and O, with no T² tensor in this ledger. The pinned reference uses
  eager attention; its allocations, optional V padding and absorbed inference
  cache layouts are not simulated. Positional-key broadcast is folded into K
  assembly. Packed projections use disjoint slices without counting a duplicate
  packed allocation.
- Fused RMSNorm, gated RMSNorm, convolution + SiLU, and SwiGLU rules reconstruct
  intermediate values within backward. This describes the chosen abstraction,
  not every unfused PyTorch autograd save. Residual additions require no input
  values for backward. Short depthwise convolution uses the memory recomputation
  tier; internal work is not profiled.
- MoE uses `R = T × 8` unpadded logical assignments, not a per-rank EP estimate.
  Route indices are retained at every tier to preserve discrete selections.
  Multiplication and routed reduction are FP32, followed by BF16 conversion,
  matching the reference training path. No DeepEP backend is assumed.
- The config includes late-layer expert SwiGLU clamp limits. The pinned reference
  `BailingMoeV3MLP.forward` does not apply them; these views follow that unclipped
  reference. Fused backends that apply the limits can require different saved
  intermediates. The separate MTP layer is outside these 42 decoder-layer views.

### Validation

Run `npm test` and `npm run build`. For UI checks, start
`npm run dev -- --port 5190 --strictPort`, then run `npm run test:browser`.
The browser script uses installed Google Chrome by default;
`ACTSCOPE_BROWSER_EXECUTABLE=/path/to/chrome npm run test:browser` supports a
separate Chrome for Testing download. `ACTSCOPE_URL` can override the app URL.
