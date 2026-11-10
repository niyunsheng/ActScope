import { operatorLabels, tensorLabels } from './labels';
import { lingbot } from './lingbot';
import { h3Config } from './h3-config';
export type RuleKey = 'linear' | 'norm' | 'modulate' | 'rope' | 'attention' | 'swiglu' | 'residual' | 'cast' | 'sigmoid' | 'route' | 'dispatch' | 'silu' | 'multiply' | 'combine' | 'add';
export type Category = 'gemm' | 'fa' | 'memory';
export type Policy = Record<Category, boolean>;
export interface Parameters { sequence: number; hidden: number; intermediate: number; heads: number; kvHeads: number; headDim: number; bytes: number }
export interface Tensor { dtype?: 'BF16' | 'FP32'; id: string; label: string; shape: (p: Parameters) => number[]; formula: string; x: number; y: number }
export interface Operator { weightShape?: number[]; inputLanes?: Record<string, number>; rule: RuleKey; id: string; label: string; category: Category; inputs: string[]; outputs: string[]; backward: string[]; x: number; y: number }
export interface Model { layout?: { width: number; height: number }; facts?: { label: string; value: string | number }[]; note?: string; id: string; name: string; tensors: Tensor[]; operators: Operator[]; boundary: string[]; defaults: Parameters }
const hidden = (p: Parameters) => [p.sequence, p.hidden];
const q = (p: Parameters) => [p.sequence, p.heads, p.headDim];
const kv = (p: Parameters) => [p.sequence, p.kvHeads, p.headDim];
const ff = (p: Parameters) => [p.sequence, p.intermediate];
const t = (id: string, label: string, x: number, y: number, shape = hidden, formula = 'T × H'): Tensor => ({ id, label, x, y, shape, formula });
const categories: Record<RuleKey, Category> = {
  linear: 'gemm', attention: 'fa', norm: 'memory', modulate: 'memory', rope: 'memory', swiglu: 'memory', residual: 'memory', cast: 'memory', sigmoid: 'memory', route: 'memory', dispatch: 'memory', silu: 'memory', multiply: 'memory', combine: 'memory', add: 'memory',
};
const op = (id: string, label: string, rule: RuleKey, inputs: string[], outputs: string[], backward: string[], x: number, y: number): Operator =>
  ({ id, label, rule, category: categories[rule], inputs, outputs, backward, x, y });
// User-supplied MiniMaxH3Transformer3DModel config and DiT diagram.
// Config has no separate KV-head count: assume standard MHA, Nkv = Nq = 56.
// Token-major layout: attention tensors use [T, heads, headDim]; other activations use [T, width].
// T is the total token count; BF16 is 2 bytes.
// Conditioning, weights, positional constants, RNG and kernel auxiliary state are outside this main-activation ledger.
export const h3: Model = {
  id: 'minimax-h3-dit', name: 'MiniMax H3 · DiT',
  defaults: { sequence: 4096, hidden: h3Config.hidden_size, intermediate: h3Config.ffn_dim, heads: h3Config.num_attention_heads, kvHeads: h3Config.num_attention_heads, headDim: h3Config.attention_head_dim, bytes: 2 },
  boundary: ['input'],
  tensors: [
    t('input', tensorLabels.input, 250, 35), t('norm1', tensorLabels.norm1, 250, 135), t('mod1', tensorLabels.mod1, 250, 235),
    t('q', tensorLabels.q, 90, 340, q, 'T × Nq × D'), t('k', tensorLabels.k, 250, 340, kv, 'T × Nkv × D'), t('v', tensorLabels.v, 410, 340, kv, 'T × Nkv × D'),
    t('qn', tensorLabels.qn, 90, 440, q, 'T × Nq × D'), t('kn', tensorLabels.kn, 250, 440, kv, 'T × Nkv × D'),
    t('qr', tensorLabels.qr, 90, 540, q, 'T × Nq × D'), t('kr', tensorLabels.kr, 250, 540, kv, 'T × Nkv × D'),
    t('attention', tensorLabels.attention, 250, 650, q, 'T × Nq × D'), t('out', tensorLabels.out, 250, 750),
    t('residual', tensorLabels.residual, 250, 850), t('norm2', tensorLabels.norm2, 250, 950), t('mod2', tensorLabels.mod2, 250, 1050),
    t('gate', 'Gate', 200, 1150, ff, 'T × I'), t('up', 'Up', 340, 1150, ff, 'T × I'),
    t('swiglu', 'SwiGLU output', 250, 1250, ff, 'T × I'), t('fc2', 'FC2 output', 250, 1350), t('output', tensorLabels.output, 250, 1450),
  ],
  operators: [
    // RMSNorm type verified in Diffusers transformer_minimax_h3.py at
    // 33b3d3704242a37be8cc6be010df97aeef9dc643: MiniMaxH3TransformerBlock / Attention.
    // Backward needs input X + external affine weights and rstd (auxiliary not counted); no Y.
    op('n1', operatorLabels.n1, 'norm', ['input'], ['norm1'], ['input'], 250, 85),
    // Y = X*(1+scale)+shift. Backward needs norm1 for dscale and external scale for dX.
    op('m1', operatorLabels.m1, 'modulate', ['norm1'], ['mod1'], ['norm1'], 250, 185),
    // Q = mod1 Wqᵀ. Backward needs mod1 for dWq, external Wq for dX; no Q.
    op('qp', operatorLabels.qp, 'linear', ['mod1'], ['q'], ['mod1'], 90, 285),
    // K = mod1 Wkᵀ. Backward shares mod1 with Q/V; do not count that input three times.
    op('kp', operatorLabels.kp, 'linear', ['mod1'], ['k'], ['mod1'], 250, 285),
    // V = mod1 Wvᵀ. Backward needs mod1, not V; FA.backward independently needs V.
    op('vp', operatorLabels.vp, 'linear', ['mod1'], ['v'], ['mod1'], 410, 285),
    // Q norm backward needs pre-norm Q and norm statistics; Q-norm output is not required.
    op('qn', operatorLabels.qn, 'norm', ['q'], ['qn'], ['q'], 90, 390),
    // K norm backward needs pre-norm K and norm statistics; K-norm output is not required.
    op('kn', operatorLabels.kn, 'norm', ['k'], ['kn'], ['k'], 250, 390),
    // Fixed RoPE: dX = Rᵀ dY. No main activation dependency; sin/cos must remain available.
    op('qr', operatorLabels.qr, 'rope', ['qn'], ['qr'], [], 90, 490),
    // Fixed RoPE: dX = Rᵀ dY. No K-norm output required by backward itself.
    op('kr', operatorLabels.kr, 'rope', ['kn'], ['kr'], [], 250, 490),
    // FA.backward needs qr, kr, v, attention O, plus LSE (outside this ledger); dropout=0.
    op('fa', operatorLabels.fa, 'attention', ['qr', 'kr', 'v'], ['attention'], ['qr', 'kr', 'v', 'attention'], 250, 595),
    // O projection backward needs attention for dW; FA.backward shares the same attention storage.
    op('o', operatorLabels.o, 'linear', ['attention'], ['out'], ['attention'], 250, 700),
    // Y = input + gate*out. Backward needs out for dgate, external gate for dbranch; not input.
    op('r1', operatorLabels.residual, 'residual', ['input', 'out'], ['residual'], ['out'], 250, 800),
    // FFN norm backward needs residual and norm statistics; no norm2 output.
    op('n2', operatorLabels.n2, 'norm', ['residual'], ['norm2'], ['residual'], 250, 900),
    // FFN modulation backward needs norm2 for scale gradient; external scale is also required.
    op('m2', operatorLabels.m2, 'modulate', ['norm2'], ['mod2'], ['norm2'], 250, 1000),
    // Packed FC1 backward needs mod2; gate/up output slices are not dependencies of FC1.backward.
    op('f1', 'fc1 · gate + up', 'linear', ['mod2'], ['gate', 'up'], ['mod2'], 250, 1100),
    // Fused SwiGLU backward needs gate/up; reconstruct SiLU(gate) inside backward, not cached Y.
    op('sw', 'swiglu', 'swiglu', ['gate', 'up'], ['swiglu'], ['gate', 'up'], 250, 1200),
    // FC2 backward needs swiglu for dW; weights are external and its output is not required.
    op('f2', 'fc2', 'linear', ['swiglu'], ['fc2'], ['swiglu'], 250, 1300),
    // Y = residual + gate*fc2. Backward needs fc2 for dgate, not residual; replay needs both.
    op('r2', operatorLabels.residual, 'residual', ['residual', 'fc2'], ['output'], ['fc2'], 250, 1400),
  ],
};
export const models = [h3, lingbot];
