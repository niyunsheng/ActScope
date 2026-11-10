# ActScope

An interactive single-layer activation memory visualizer showing backward dependencies and how recomputation changes saved tensors. Built with React and TypeScript as a static website.

Supports a single **MiniMax H3 DiT** or **LingBot Video MoE 30B-A3B** layer, selected beside ActScope. Refiner layers are excluded. Total token count T is editable; main activations use BF16 (LingBot router and weighted-combine inputs also include FP32). Attention activations use THD layout `[T, heads, head_dim]`; other activations use `[T, width]`, with no separate batch dimension.

## Run locally

```sh
npm ci
npm run dev -- --port 5190
```

Open the [local app](http://127.0.0.1:5190/).

- Compare four cumulative strategies: no recomputation → memory-bound → memory-bound + GEMM → full recomputation (adding FA).
- Yellow tensors are saved; white tensors are not retained.
- Hover over an operator to highlight its backward dependencies. The toolbar shows `Bwd needs` and, for GEMMs, the weight shape `[out, in]`.
- Hover over a tensor to highlight its backward consumers. The toolbar shows its shape, size in MiB, retention status, and `Bwd used by`.
- Details and highlights clear when the pointer leaves; clicking does not pin them.
- Scroll to zoom, drag the graph or nodes to pan, or use Fit and Reset.

## Scope

Estimates **main activations retained for backward** for one layer on one GPU without sharding. This is not peak GPU memory. Parameters, gradients, optimizer states, conditioning branches, norm statistics, FA LSE, and temporary workspaces are excluded.

H3 follows a user-provided configuration and diagram; pre-norm and Q/K RMSNorm types are verified against the [Diffusers implementation](https://github.com/huggingface/diffusers/blob/33b3d3704242a37be8cc6be010df97aeef9dc643/src/diffusers/models/transformers/transformer_minimax_h3.py). Assumptions include standard MHA, fixed RoPE, fused SwiGLU, zero attention dropout, and trainable modulation and residual gates. See [model.ts](src/activation/model.ts) for operator dependencies and [h3-config.ts](src/activation/h3-config.ts) for dimensions. The graph displays H3’s packed FC1 as separate `gate_proj` and `up_proj` logical projections sharing one input; this does not change activation totals. Activation totals have not been validated against a runtime saved-tensor trace or GPU measurements.

LingBot uses the [official implementation and checkpoint config](docs/lingbot-video.md): 128 routed experts, top-8 routing, and one shared expert. T includes video and text tokens. Its source references, mixed-precision accounting and training assumptions are documented there.

## Development and deployment

Models are defined in `src/activation/model.ts`; the analysis engine, rendering, and viewport controls are shared. To add a model, define a `Model` and register it in `models`.

```sh
npm test              # Analysis and numerical gradient tests
npm run test:browser  # Requires the local server and Chrome
npm run build         # Type checking and production build
```

Deploy `dist/` to GitHub Pages. Changes must go through a feature branch and PR; `main` requires the `validate` check and disallows direct or forced pushes, including administrators.
