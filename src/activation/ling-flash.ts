import { hidden, tensorFactory, operator } from './nodes';
import { createGrid } from './layout';
import type { Model, Parameters, Tensor, Operator } from './model';
import { lingFlashConfig as c } from './ling-flash-config';

// BailingMoeV3DecoderLayer / KimiDeltaAttention / MultiLatentAttention / SparseMoeBlock:
// https://huggingface.co/inclusionAI/Ling-3.0-flash/blob/fcc60a52b34c69c6271d0ac8a8cb0815020b6a43/modeling_bailing_moe_v3.py
// Main-activation ledger only. Backend assumptions and omitted state: README.md.
const H = c.hidden_size, N = c.num_attention_heads, D = c.head_dim;
const E = c.num_experts, K = c.num_experts_per_tok, I = c.moe_intermediate_size;
const S = c.moe_shared_expert_intermediate_size * c.num_shared_experts;
const heads = (width: number) => (p: Parameters) => [p.sequence, N, width];
const width = (n: number) => (p: Parameters) => [p.sequence, n];
const routedHidden = (p: Parameters) => [p.sequence * K, H];
const routedFF = (p: Parameters) => [p.sequence * K, I];
const t = tensorFactory('BF16');
const grid = createGrid({ main: 600, q: 200, k: 400, v: 600, decay: 800, beta: 1000, gate: 1200,
  qRope: 350, latent: 650, kRope: 850, kNope: 550, mlaV: 750, mlaGate: 1100,
  router: 180, expertGate: 520, expertUp: 680, shared: 1050, sharedGate: 970, sharedUp: 1130 }, { all: 0 });
const at = (column: keyof typeof grid.columns, row: number) => grid.point(column, row, 'all');

function makeLayer(mla: boolean, dense: boolean): Model {
  const tensors: Tensor[] = [t('input', 'Layer input', at('main', 0)), t('norm1', 'Attention RMSNorm', at('main', 2))];
  const operators: Operator[] = [operator('n1', 'rmsnorm', 'norm', ['input'], ['norm1'], ['input'], at('main', 1))];
  if (mla) {
    // Packed projections are represented by disjoint slices, with no extra split allocations.
    tensors.push(
      t('qNope', 'Q non-RoPE', at('q', 4), heads(c.qk_nope_head_dim)),
      t('qPos', 'Q positional', at('qRope', 4), heads(c.qk_rope_head_dim)),
      t('latent', 'KV latent', at('latent', 4), width(c.kv_lora_rank)),
      t('kPos', 'Shared K positional', at('kRope', 4), p => [p.sequence, 1, c.qk_rope_head_dim]),
      t('gateLogits', 'Head gate logits', at('mlaGate', 4), width(N)),
      t('latentNorm', 'Normalized KV latent', at('latent', 6), width(c.kv_lora_rank)),
      t('qRot', 'Q RoPE slice', at('qRope', 6), heads(c.qk_rope_head_dim)),
      t('kRot', 'Shared K RoPE slice', at('kRope', 6), p => [p.sequence, 1, c.qk_rope_head_dim]),
      t('gateFP32', 'Head sigmoid FP32', at('mlaGate', 6), width(N), 'FP32'),
      t('attnGate', 'Sigmoid head gate BF16', at('mlaGate', 8), width(N)),
      t('q', 'Q assembled', at('q', 8), heads(c.qk_head_dim)),
      t('kNope', 'K non-RoPE', at('kNope', 8), heads(c.qk_nope_head_dim)),
      t('v', 'V', at('mlaV', 8), heads(c.v_head_dim)),
      t('k', 'K assembled', at('k', 10), heads(c.qk_head_dim)),
      t('attention', 'MLA attention output', at('main', 12), heads(c.v_head_dim)),
      t('gated', 'Gated attention output', at('main', 14), heads(c.v_head_dim)),
      t('out', 'Attention projection', at('main', 16)),
    );
    operators.push(
      operator('qp', 'q_proj', 'linear', ['norm1'], ['qNope', 'qPos'], ['norm1'], at('q', 3), { weightShape: [N*c.qk_head_dim, H] }),
      operator('kvA', 'kv_a_proj', 'linear', ['norm1'], ['latent', 'kPos'], ['norm1'], at('latent', 3), { weightShape: [c.kv_lora_rank+c.qk_rope_head_dim, H] }),
      operator('gateProj', 'g_proj', 'linear', ['norm1'], ['gateLogits'], ['norm1'], at('mlaGate', 3), { weightShape: [N, H] }),
      operator('kvNorm', 'kv_rmsnorm', 'norm', ['latent'], ['latentNorm'], ['latent'], at('latent', 5)),
      operator('qRoPE', 'rope', 'rope', ['qPos'], ['qRot'], [], at('qRope', 5)),
      operator('kRoPE', 'rope', 'rope', ['kPos'], ['kRot'], [], at('kRope', 5)),
      // Keep the FP32 sigmoid for its derivative and the BF16 gate for multiply.
      operator('gateSigmoid', 'sigmoid_fp32', 'sigmoid', ['gateLogits'], ['gateFP32'], ['gateFP32'], at('mlaGate', 5)),
      operator('gateCast', 'cast_bf16', 'cast', ['gateFP32'], ['attnGate'], [], at('mlaGate', 7)),
      operator('packQ', 'concat_q', 'reshape', ['qNope', 'qRot'], ['q'], [], at('q', 7)),
      operator('kvB', 'kv_b_proj', 'linear', ['latentNorm'], ['kNope', 'v'], ['latentNorm'], at('latent', 7), { weightShape: [N*(c.qk_nope_head_dim+c.v_head_dim), c.kv_lora_rank] }),
      operator('packK', 'broadcast_concat_k', 'reshape', ['kNope', 'kRot'], ['k'], [], at('k', 9)),
      // Explicit FA-style training abstraction, not the reference eager T² allocator.
      operator('fa', 'causal_mla_attention', 'attention', ['q', 'k', 'v'], ['attention'], ['q', 'k', 'v', 'attention'], at('main', 11)),
      operator('attnWeight', 'head_gate_multiply', 'multiply', ['attention', 'attnGate'], ['gated'], ['attention', 'attnGate'], at('main', 13), { inputLanes: { attnGate: 1250 } }),
      operator('o', 'o_proj', 'linear', ['gated'], ['out'], ['gated'], at('main', 15), { weightShape: [H, N*c.v_head_dim] }),
    );
  } else {
    for (const [id, column] of [['q','q'], ['k','k'], ['v','v']] as const) {
      tensors.push(t(id, `${id.toUpperCase()} projection`, at(column, 4), heads(D)), t(`${id}Conv`, `${id.toUpperCase()} conv + SiLU`, at(column, 6), heads(D)));
      operators.push(
        operator(`${id}p`, `${id}_proj`, 'linear', ['norm1'], [id], ['norm1'], at(column, 3), { weightShape: [N*D, H] }),
        // Fused depthwise causal convolution + SiLU; replay pre-SiLU values in backward.
        operator(`${id}Conv`, 'causal_conv_silu', 'convolution', [id], [`${id}Conv`], [id], at(column, 5)),
      );
    }
    tensors.push(
      t('decay', 'Decay gate logits', at('decay', 4), heads(D)),
      t('betaLogits', 'Beta logits', at('beta', 4), width(N)),
      t('beta', 'Beta sigmoid FP32', at('beta', 6), width(N), 'FP32'),
      t('outputGate', 'Output gate logits', at('gate', 4), heads(D)),
      t('attention', 'KDA output', at('main', 8), heads(D)),
      t('gated', 'Gated RMSNorm output', at('main', 10), heads(D)),
      t('out', 'Attention projection', at('main', 12)),
    );
    operators.push(
      operator('fProj', 'f_proj', 'linear', ['norm1'], ['decay'], ['norm1'], at('decay', 3), { weightShape: [N*D, H] }),
      operator('bProj', 'b_proj', 'linear', ['norm1'], ['betaLogits'], ['norm1'], at('beta', 3), { weightShape: [N, H] }),
      operator('betaSigmoid', 'sigmoid_fp32', 'sigmoid', ['betaLogits'], ['beta'], ['beta'], at('beta', 5)),
      operator('gateProj', 'g_proj', 'linear', ['norm1'], ['outputGate'], ['norm1'], at('gate', 3), { weightShape: [N*D, H] }),
      // Q/K L2 norm, diagonal decay transform and recurrent-state replay are internal.
      // Keep the main inputs, not a fictitious softmax matrix or FA Q/K/V/O rule.
      operator('kda', 'chunk_kda', 'kda', ['qConv', 'kConv', 'vConv', 'decay', 'beta'], ['attention'], ['qConv', 'kConv', 'vConv', 'decay', 'beta'], at('main', 7)),
      operator('gatedNorm', 'gated_rmsnorm', 'gatedNorm', ['attention', 'outputGate'], ['gated'], ['attention', 'outputGate'], at('main', 9), { inputLanes: { outputGate: 1280 } }),
      operator('o', 'o_proj', 'linear', ['gated'], ['out'], ['gated'], at('main', 11), { weightShape: [H, N*D] }),
    );
  }
  const endAttention = mla ? 16 : 12;
  const start = endAttention + 5;
  tensors.push(t('residual', 'Attention residual', at('main', endAttention+2)), t('norm2', 'FFN RMSNorm', at('main', endAttention+4)));
  operators.push(
    operator('resA', 'residual_add', 'add', ['input', 'out'], ['residual'], [], at('main', endAttention+1), { inputLanes: { input: 1320 } }),
    operator('n2', 'rmsnorm', 'norm', ['residual'], ['norm2'], ['residual'], at('main', endAttention+3)),
  );
  if (dense) {
    tensors.push(
      t('gate', 'Dense gate', at('expertGate', start+1), width(c.intermediate_size)),
      t('up', 'Dense up', at('expertUp', start+1), width(c.intermediate_size)),
      t('product', 'Dense SwiGLU output', at('main', start+3), width(c.intermediate_size)),
      t('ffnOut', 'Dense FFN output', at('main', start+5)),
    );
    operators.push(
      operator('w1', 'gate_proj', 'linear', ['norm2'], ['gate'], ['norm2'], at('expertGate', start), { weightShape: [c.intermediate_size, H] }),
      operator('w3', 'up_proj', 'linear', ['norm2'], ['up'], ['norm2'], at('expertUp', start), { weightShape: [c.intermediate_size, H] }),
      operator('sw', 'swiglu', 'swiglu', ['gate', 'up'], ['product'], ['gate', 'up'], at('main', start+2)),
      operator('w2', 'down_proj', 'linear', ['product'], ['ffnOut'], ['product'], at('main', start+4), { weightShape: [H, c.intermediate_size] }),
    );
  } else {
    tensors.push(
      t('routerInput', 'Router input FP32', at('router', start+1), hidden, 'FP32'),
      t('logits', 'Router logits', at('router', start+3), width(E), 'FP32'),
      t('scores', 'Sigmoid scores', at('router', start+5), width(E), 'FP32'),
      { ...t('indices', 'Top-k indices', at('router', start+7), width(K), 'INT64'), retain: true },
      t('weights', 'Normalized route weights', at('router', start+9), width(K), 'FP32'),
      t('dispatch', 'Dispatched tokens', at('main', start+9), routedHidden),
      t('gate', 'Expert gate', at('expertGate', start+11), routedFF),
      t('up', 'Expert up', at('expertUp', start+11), routedFF),
      t('product', 'Expert SwiGLU output', at('main', start+13), routedFF),
      t('expertOut', 'Expert output', at('main', start+15), routedHidden),
      t('weighted', 'Weighted output FP32', at('main', start+17), routedHidden, 'FP32'),
      t('routedFP32', 'Routed sum FP32', at('main', start+19), hidden, 'FP32'),
      t('routed', 'Routed sum BF16', at('main', start+21)),
      t('sharedGate', 'Shared gate', at('sharedGate', start+1), width(S)),
      t('sharedUp', 'Shared up', at('sharedUp', start+1), width(S)),
      t('sharedProduct', 'Shared SwiGLU output', at('shared', start+3), width(S)),
      t('sharedOut', 'Shared output', at('shared', start+5)),
      t('ffnOut', 'MoE output', at('main', start+23)),
    );
    operators.push(
      operator('castRouter', 'cast_fp32', 'cast', ['norm2'], ['routerInput'], [], at('router', start)),
      operator('router', 'router_gemm', 'linear', ['routerInput'], ['logits'], ['routerInput'], at('router', start+2), { weightShape: [E, H] }),
      operator('sigmoid', 'sigmoid', 'sigmoid', ['logits'], ['scores'], ['scores'], at('router', start+4)),
      // Expert bias affects discrete selection only; weight gather uses unbiased scores.
      operator('topk', 'biased_group_topk', 'route', ['scores'], ['indices'], [], at('router', start+6), { forwardOnly: true }),
      operator('route', 'gather_norm_scale', 'route', ['scores', 'indices'], ['weights'], ['scores', 'indices'], at('router', start+8), { inputLanes: { scores: 80 } }),
      operator('dispatch', 'dispatch', 'dispatch', ['norm2', 'indices'], ['dispatch'], ['indices'], at('main', start+8)),
      operator('w1', 'expert_gate_proj', 'linear', ['dispatch'], ['gate'], ['dispatch'], at('expertGate', start+10), { weightShape: [E, I, H] }),
      operator('w3', 'expert_up_proj', 'linear', ['dispatch'], ['up'], ['dispatch'], at('expertUp', start+10), { weightShape: [E, I, H] }),
      operator('sw', 'swiglu', 'swiglu', ['gate', 'up'], ['product'], ['gate', 'up'], at('main', start+12)),
      operator('w2', 'expert_down_proj', 'linear', ['product'], ['expertOut'], ['product'], at('main', start+14), { weightShape: [E, H, I] }),
      operator('weight', 'apply_route_weight', 'multiply', ['expertOut', 'weights'], ['weighted'], ['expertOut', 'weights'], at('main', start+16), { inputLanes: { weights: 300 } }),
      operator('combine', 'combine_sum', 'combine', ['weighted', 'indices'], ['routedFP32'], ['indices'], at('main', start+18), { inputLanes: { indices: 320 } }),
      operator('castRouted', 'cast_bf16', 'cast', ['routedFP32'], ['routed'], [], at('main', start+20)),
      operator('sharedW1', 'shared_gate_proj', 'linear', ['norm2'], ['sharedGate'], ['norm2'], at('sharedGate', start), { weightShape: [S, H], inputBranches: { norm2: grid.columns.shared } }),
      operator('sharedW3', 'shared_up_proj', 'linear', ['norm2'], ['sharedUp'], ['norm2'], at('sharedUp', start), { weightShape: [S, H], inputBranches: { norm2: grid.columns.shared } }),
      operator('sharedSw', 'swiglu', 'swiglu', ['sharedGate', 'sharedUp'], ['sharedProduct'], ['sharedGate', 'sharedUp'], at('shared', start+2)),
      operator('sharedW2', 'shared_down_proj', 'linear', ['sharedProduct'], ['sharedOut'], ['sharedProduct'], at('shared', start+4), { weightShape: [H, S] }),
      operator('addExperts', 'add_shared', 'add', ['routed', 'sharedOut'], ['ffnOut'], [], at('main', start+22), { inputLanes: { sharedOut: 1250 } }),
    );
  }
  const lastRow = start + (dense ? 7 : 25);
  tensors.push(t('output', 'Layer output', at('main', lastRow)));
  operators.push(operator('resF', 'residual_add', 'add', ['residual', 'ffnOut'], ['output'], [], at('main', lastRow-1), { inputLanes: { residual: 1320 } }));
  return {
    id: mla ? 'ling-flash-mla-moe' : dense ? 'ling-flash-kda-dense' : 'ling-flash-kda-moe',
    name: mla ? 'Gated MLA + MoE · 7 layers' : dense ? 'KDA + Dense · layers 1–2' : 'KDA + MoE · 33 layers',
    attentionLabel: mla ? 'FA' : 'KDA', activationDtype: 'BF16 / FP32',
    intermediateLabel: dense ? 'FFN intermediate' : 'Expert intermediate',
    memoryDescription: mla ? 'Norm, RoPE, routing, SwiGLU, gates' : 'Norm, short conv, routing, SwiGLU, gates',
    attentionDimensions: mla ? [
      { label: 'Q / K head dimension', value: c.qk_head_dim }, { label: 'V head dimension', value: c.v_head_dim },
      { label: 'KV latent rank', value: c.kv_lora_rank }, { label: 'RoPE dimension', value: c.qk_rope_head_dim },
    ] : [{ label: 'Head dimension', value: D }, { label: 'Q / K / V width', value: N*D }, { label: 'Conv kernel', value: c.short_conv_kernel_size }],
    defaults: { sequence: 4096, hidden: H, intermediate: dense ? c.intermediate_size : I, heads: N, kvHeads: N, headDim: mla ? c.qk_head_dim : D, bytes: 2 },
    facts: [{ label: 'Model layers', value: c.num_hidden_layers }, { label: 'KDA : MLA', value: '5 : 1' },
      ...(dense ? [] : [{ label: 'Routed experts', value: E }, { label: 'Top-k experts', value: K }, { label: 'Shared experts', value: c.num_shared_experts }, { label: 'Groups / selected', value: `${c.n_group} / ${c.topk_group}` }])],
    note: `${mla ? 'Layers 6, 12, 18, 24, 30, 36, 42: gated MLA + MoE; expanded training Q/K/V with an FA-style backend assumption.' : dense ? 'Layers 1–2: KDA + dense SwiGLU FFN.' : 'KDA layers from layer 3 onward, excluding multiples of 6: KDA + MoE; routed rows = T × 8.'} Logical saved main activations, not measured GPU memory. Kernel auxiliary state, KDA recurrent/chunk state, caches, weights and communication buffers are excluded.`,
    sources: [{ label: 'Official model card', href: 'https://huggingface.co/inclusionAI/Ling-3.0-flash' },
      { label: 'Pinned config & implementation', href: 'https://huggingface.co/inclusionAI/Ling-3.0-flash/tree/fcc60a52b34c69c6271d0ac8a8cb0815020b6a43' }],
    layout: grid.size(1400, lastRow), boundary: ['input'], tensors, operators,
    regions: dense ? [] : [
      { label: 'Router · FP32', x: 30, y: grid.y(start)-22, width: 300, height: grid.y(start+10)-grid.y(start) },
      { label: 'Routed experts · top-8 / 512', x: 450, y: grid.y(start+10)-22, width: 300, height: grid.y(start+20)-grid.y(start+10) },
      { label: 'Shared expert', x: 900, y: grid.y(start)-22, width: 300, height: grid.y(start+6)-grid.y(start) },
    ],
  };
}

export const lingFlashKdaMoe = makeLayer(false, false);
export const lingFlashMlaMoe = makeLayer(true, false);
export const lingFlashKdaDense = makeLayer(false, true);
export const lingFlash: Model = {
  ...lingFlashKdaMoe, id: 'ant-ling-3-flash', name: 'Ant Ling 3.0 Flash',
  variants: [lingFlashKdaMoe, lingFlashMlaMoe, lingFlashKdaDense],
};
