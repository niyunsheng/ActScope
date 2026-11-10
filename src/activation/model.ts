import { createGrid } from './layout';
import { operatorLabels, tensorLabels } from './labels';
import { lingbot } from './lingbot';
import { h3Config } from './h3-config';
export type RuleKey = 'linear' | 'norm' | 'modulate' | 'rope' | 'attention' | 'swiglu' | 'residual' | 'cast' | 'sigmoid' | 'route' | 'dispatch' | 'silu' | 'multiply' | 'combine' | 'add';
export type Category = 'gemm' | 'fa' | 'memory';
export type Policy = Record<Category, boolean>;
export interface Parameters { sequence: number; hidden: number; intermediate: number; heads: number; kvHeads: number; headDim: number; bytes: number }
export interface Tensor { metadata?: string; retain?: boolean; dtype?: 'BF16' | 'FP32' | 'INT64'; id: string; label: string; shape: (p: Parameters) => number[]; formula: string; x: number; y: number }
export interface Operator { forwardOnly?: boolean; inputBranches?: Record<string, number>; weightShape?: number[]; inputLanes?: Record<string, number>; rule: RuleKey; id: string; label: string; category: Category; inputs: string[]; outputs: string[]; backward: string[]; x: number; y: number }
export interface Model { regions?: { label: string; x: number; y: number; width: number; height: number }[]; layout?: { width: number; height: number }; facts?: { label: string; value: string | number }[]; note?: string; id: string; name: string; tensors: Tensor[]; operators: Operator[]; boundary: string[]; defaults: Parameters }
const hidden = (p: Parameters) => [p.sequence, p.hidden];
const q = (p: Parameters) => [p.sequence, p.heads, p.headDim];
const kv = (p: Parameters) => [p.sequence, p.kvHeads, p.headDim];
const ff = (p: Parameters) => [p.sequence, p.intermediate];
const grid = createGrid({ left: 90, main: 250, right: 410, gate: 170, up: 330 }, { attention: 0, ffn: 16 });
const t = (id: string, label: string, position: { x: number; y: number }, shape = hidden, formula = 'T × H'): Tensor => ({ id, label, ...position, shape, formula });
const categories: Record<RuleKey, Category> = {
  linear: 'gemm', attention: 'fa', norm: 'memory', modulate: 'memory', rope: 'memory', swiglu: 'memory', residual: 'memory', cast: 'memory', sigmoid: 'memory', route: 'memory', dispatch: 'memory', silu: 'memory', multiply: 'memory', combine: 'memory', add: 'memory',
};
const op = (id: string, label: string, rule: RuleKey, inputs: string[], outputs: string[], backward: string[], position: { x: number; y: number }): Operator =>
  ({ id, label, rule, category: categories[rule], inputs, outputs, backward, ...position });
// User-supplied MiniMaxH3Transformer3DModel config and DiT diagram.
// Config has no separate KV-head count: assume standard MHA, Nkv = Nq = 56.
// Token-major layout: attention tensors use [T, heads, headDim]; other activations use [T, width].
// T is the total token count; BF16 is 2 bytes.
// Conditioning, weights, positional constants, RNG and kernel auxiliary state are outside this main-activation ledger.
export const h3: Model = {
  id: 'minimax-h3-dit', name: 'MiniMax H3 · DiT',
  layout: grid.size(520, 28),
  defaults: { sequence: 4096, hidden: h3Config.hidden_size, intermediate: h3Config.ffn_dim, heads: h3Config.num_attention_heads, kvHeads: h3Config.num_attention_heads, headDim: h3Config.attention_head_dim, bytes: 2 },
  boundary: ['input'],
  tensors: [
    t('input', tensorLabels.input, grid.point('main', 0, 'attention')), t('norm1', tensorLabels.norm1, grid.point('main', 2, 'attention')), t('mod1', tensorLabels.mod1, grid.point('main', 4, 'attention')),
    t('q', tensorLabels.q, grid.point('left', 6, 'attention'), q, 'T × Nq × D'), t('k', tensorLabels.k, grid.point('main', 6, 'attention'), kv, 'T × Nkv × D'), t('v', tensorLabels.v, grid.point('right', 6, 'attention'), kv, 'T × Nkv × D'),
    t('qn', tensorLabels.qn, grid.point('left', 8, 'attention'), q, 'T × Nq × D'), t('kn', tensorLabels.kn, grid.point('main', 8, 'attention'), kv, 'T × Nkv × D'),
    t('qr', tensorLabels.qr, grid.point('left', 10, 'attention'), q, 'T × Nq × D'), t('kr', tensorLabels.kr, grid.point('main', 10, 'attention'), kv, 'T × Nkv × D'),
    t('attention', tensorLabels.attention, grid.point('main', 12, 'attention'), q, 'T × Nq × D'), t('out', tensorLabels.out, grid.point('main', 14, 'attention')),
    t('residual', tensorLabels.residual, grid.point('main', 0, 'ffn')), t('norm2', tensorLabels.norm2, grid.point('main', 2, 'ffn')), t('mod2', tensorLabels.mod2, grid.point('main', 4, 'ffn')),
    t('gate', 'Gate', grid.point('gate', 6, 'ffn'), ff, 'T × I'), t('up', 'Up', grid.point('up', 6, 'ffn'), ff, 'T × I'),
    t('swiglu', 'SwiGLU output', grid.point('main', 8, 'ffn'), ff, 'T × I'), t('fc2', 'Down projection output', grid.point('main', 10, 'ffn')), t('output', tensorLabels.output, grid.point('main', 12, 'ffn')),
  ],
  operators: [
    // RMSNorm type verified in Diffusers transformer_minimax_h3.py at
    // 33b3d3704242a37be8cc6be010df97aeef9dc643: MiniMaxH3TransformerBlock / Attention.
    // Backward needs input X + external affine weights and rstd (auxiliary not counted); no Y.
    op('n1', operatorLabels.n1, 'norm', ['input'], ['norm1'], ['input'], grid.point('main', 1, 'attention')),
    // Y = X*(1+scale)+shift. Backward needs norm1 for dscale and external scale for dX.
    op('m1', operatorLabels.m1, 'modulate', ['norm1'], ['mod1'], ['norm1'], grid.point('main', 3, 'attention')),
    // Q = mod1 Wqᵀ. Backward needs mod1 for dWq, external Wq for dX; no Q.
    op('qp', operatorLabels.qp, 'linear', ['mod1'], ['q'], ['mod1'], grid.point('left', 5, 'attention')),
    // K = mod1 Wkᵀ. Backward shares mod1 with Q/V; do not count that input three times.
    op('kp', operatorLabels.kp, 'linear', ['mod1'], ['k'], ['mod1'], grid.point('main', 5, 'attention')),
    // V = mod1 Wvᵀ. Backward needs mod1, not V; FA.backward independently needs V.
    op('vp', operatorLabels.vp, 'linear', ['mod1'], ['v'], ['mod1'], grid.point('right', 5, 'attention')),
    // Q norm backward needs pre-norm Q and norm statistics; Q-norm output is not required.
    op('qn', operatorLabels.qn, 'norm', ['q'], ['qn'], ['q'], grid.point('left', 7, 'attention')),
    // K norm backward needs pre-norm K and norm statistics; K-norm output is not required.
    op('kn', operatorLabels.kn, 'norm', ['k'], ['kn'], ['k'], grid.point('main', 7, 'attention')),
    // Fixed RoPE: dX = Rᵀ dY. No main activation dependency; sin/cos must remain available.
    op('qr', operatorLabels.qr, 'rope', ['qn'], ['qr'], [], grid.point('left', 9, 'attention')),
    // Fixed RoPE: dX = Rᵀ dY. No K-norm output required by backward itself.
    op('kr', operatorLabels.kr, 'rope', ['kn'], ['kr'], [], grid.point('main', 9, 'attention')),
    // FA.backward needs qr, kr, v, attention O, plus LSE (outside this ledger); dropout=0.
    op('fa', operatorLabels.fa, 'attention', ['qr', 'kr', 'v'], ['attention'], ['qr', 'kr', 'v', 'attention'], grid.point('main', 11, 'attention')),
    // O projection backward needs attention for dW; FA.backward shares the same attention storage.
    op('o', operatorLabels.o, 'linear', ['attention'], ['out'], ['attention'], grid.point('main', 13, 'attention')),
    // Y = input + gate*out. Backward needs out for dgate, external gate for dbranch; not input.
    { ...op('r1', operatorLabels.residual, 'residual', ['input', 'out'], ['residual'], ['out'], grid.point('main', 15, 'attention')), inputLanes: { input: 492 } },
    // FFN norm backward needs residual and norm statistics; no norm2 output.
    op('n2', operatorLabels.n2, 'norm', ['residual'], ['norm2'], ['residual'], grid.point('main', 1, 'ffn')),
    // FFN modulation backward needs norm2 for scale gradient; external scale is also required.
    op('m2', operatorLabels.m2, 'modulate', ['norm2'], ['mod2'], ['norm2'], grid.point('main', 3, 'ffn')),
    // Logical views of the packed FC1: both projections share mod2 storage.
    // Gate/up are disjoint output slices; splitting the graph does not add activation bytes.
    op('gate_proj', 'gate_proj', 'linear', ['mod2'], ['gate'], ['mod2'], grid.point('gate', 5, 'ffn')),
    op('up_proj', 'up_proj', 'linear', ['mod2'], ['up'], ['mod2'], grid.point('up', 5, 'ffn')),
    // Fused SwiGLU backward needs gate/up; reconstruct SiLU(gate) inside backward, not cached Y.
    op('sw', 'swiglu', 'swiglu', ['gate', 'up'], ['swiglu'], ['gate', 'up'], grid.point('main', 7, 'ffn')),
    // FC2 backward needs swiglu for dW; weights are external and its output is not required.
    op('f2', 'down_proj', 'linear', ['swiglu'], ['fc2'], ['swiglu'], grid.point('main', 9, 'ffn')),
    // Y = residual + gate*fc2. Backward needs fc2 for dgate, not residual; replay needs both.
    { ...op('r2', operatorLabels.residual, 'residual', ['residual', 'fc2'], ['output'], ['fc2'], grid.point('main', 11, 'ffn')), inputLanes: { residual: 470 } },
  ],
};
export const models = [h3, lingbot];
