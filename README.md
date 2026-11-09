# ActScope

An interactive single-layer activation memory visualizer showing backward dependencies and how recomputation changes saved tensors. Built with React and TypeScript as a static website.

Currently supports **one MiniMax H3 DiT layer**, excluding the text refiner. Total token count T is editable; activation dtype is fixed to BF16. Attention activations use THD layout `[T, heads, head_dim]`; other activations use `[T, width]`, with no separate batch dimension.

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

The model follows a user-provided configuration and diagram. Assumptions include standard MHA, fixed RoPE, fused SwiGLU, zero attention dropout, and trainable modulation and residual gates. See [model.ts](src/activation/model.ts) for operator dependencies and [h3-config.ts](src/activation/h3-config.ts) for dimensions. Estimates have not been validated against the official H3 implementation or GPU measurements.

## Development and deployment

Models are defined in `src/activation/model.ts`; the analysis engine, rendering, and viewport controls are shared. To add a model, define a `Model` and register it in `models`.

```sh
npm test              # Analysis and numerical gradient tests
npm run test:browser  # Requires the local server and Chrome
npm run build         # Type checking and production build
```

Deploy `dist/` to GitHub Pages.
