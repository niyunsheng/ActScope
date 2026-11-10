import { createGrid } from './layout';
import { operatorLabels, tensorLabels } from './labels';
import type { Model, Operator, Parameters, RuleKey, Tensor } from './model';
import { lingbotConfig as c } from './lingbot-config';

// Architecture: Robbyant/lingbot-video @ dd5c231e793406c6a8893e9e4307008a9c2adfe4,
// lingbot_video/transformer_lingbot_video.py: LingBotVideoBlock / SparseMoeBlock.
// Logical training ledger, not the inference implementation's allocator trace.
const H = c.hidden_size, I = c.moe_intermediate_size, E = c.num_experts, K = c.num_experts_per_tok;
const hidden = (p: Parameters) => [p.sequence, p.hidden];
const attention = (p: Parameters) => [p.sequence, p.heads, p.headDim];
const routedHidden = (p: Parameters) => [p.sequence * K, p.hidden];
const routedFF = (p: Parameters) => [p.sequence * K, p.intermediate];
const sharedFF = (p: Parameters) => [p.sequence, p.intermediate * c.n_shared_experts];
const grid = createGrid({ main: 520, left: 360, right: 680, router: 170, gate: 440, up: 600, shared: 910, sharedGate: 830, sharedUp: 990 }, { attention: 0, ffn: 18, branches: 25 });
const tensor = (id: string, label: string, position: { x: number; y: number }, shape = hidden, dtype: NonNullable<Tensor['dtype']> = 'BF16'): Tensor =>
  ({ id, label, ...position, shape, dtype, formula: '' });
const operator = (id: string, label: string, rule: RuleKey, inputs: string[], outputs: string[], backward: string[], position: { x: number; y: number }, weightShape?: number[], inputLanes?: Record<string, number>, inputBranches?: Operator['inputBranches']): Operator =>
  ({ id, label, rule, category: rule === 'linear' ? 'gemm' : rule === 'attention' ? 'fa' : 'memory', inputs, outputs, backward, ...position, weightShape, inputLanes, inputBranches });

export const lingbot: Model = {
  id: 'lingbot-video-moe', name: 'LingBot Video · MoE',
  layout: grid.size(1200, 44),
  regions: [
    { label: 'Router', x: 40, y: grid.y(23), width: 260, height: grid.y(39) - grid.y(23) },
    { label: 'Routed experts', x: 350, y: grid.y(23), width: 340, height: grid.y(39) - grid.y(23) },
    { label: 'Shared expert', x: 750, y: grid.y(23), width: 340, height: grid.y(39) - grid.y(23) },
  ],
  defaults: { sequence: 4096, hidden: H, intermediate: I, heads: c.num_attention_heads, kvHeads: c.num_attention_heads, headDim: H / c.num_attention_heads, bytes: 2 },
  facts: [{label:'EP backend',value:'DeepEP'}, {label:'Routed experts',value:E}, {label:'Top-k experts',value:K}, {label:'Groups',value:c.n_group}, {label:'Selected groups',value:c.topk_group}, {label:'Shared experts',value:c.n_shared_experts}],
  note: 'DeepEP expanded-layout abstraction: R = T × 8 unpadded expert rows, not a per-rank EP estimate. Handle retained; its storage, communication buffers and padding are excluded.',
  boundary: ['input'],
  tensors: [
    tensor('input', tensorLabels.input,grid.point('main', 0, 'attention')), tensor('norm1', tensorLabels.norm1,grid.point('main', 2, 'attention')), tensor('mod1', tensorLabels.mod1,grid.point('main', 4, 'attention')),
    tensor('q', tensorLabels.q,grid.point('left', 6, 'attention'),attention), tensor('k', tensorLabels.k,grid.point('main', 6, 'attention'),attention), tensor('v', tensorLabels.v,grid.point('right', 6, 'attention'),attention),
    tensor('qn', tensorLabels.qn,grid.point('left', 8, 'attention'),attention), tensor('kn', tensorLabels.kn,grid.point('main', 8, 'attention'),attention),
    tensor('qr', tensorLabels.qr,grid.point('left', 10, 'attention'),attention), tensor('kr', tensorLabels.kr,grid.point('main', 10, 'attention'),attention),
    tensor('attention', tensorLabels.attention,grid.point('main', 12, 'attention'),attention), tensor('out', tensorLabels.out,grid.point('main', 14, 'attention')),
    tensor('postAttn','Post-attention norm',grid.point('main', 16, 'attention')), tensor('residual', tensorLabels.residual,grid.point('main', 0, 'ffn')),
    tensor('norm2', tensorLabels.norm2,grid.point('main', 2, 'ffn')), tensor('mod2', tensorLabels.mod2,grid.point('main', 4, 'ffn')),
    tensor('routerInput','Router input FP32',grid.point('router', 1, 'branches'),hidden,'FP32'),
    tensor('logits','Router logits',grid.point('router', 3, 'branches'),p=>[p.sequence,E],'FP32'),
    tensor('scores','Sigmoid scores',grid.point('router', 5, 'branches'),p=>[p.sequence,E],'FP32'),
    { ...tensor('indices','Top-k indices',grid.point('router', 7, 'branches'),p=>[p.sequence,K],'INT64'), retain: true },
    tensor('weights','Top-k weights',grid.point('router', 9, 'branches'),p=>[p.sequence,K],'FP32'),
    { ...tensor('handle','EP handle',grid.point('router', 1, 'branches'),()=>[]), metadata: 'Routing/layout metadata · size excluded', retain: true },
    tensor('dispatch','Dispatched tokens',grid.point('main', 1, 'branches'),routedHidden),
    tensor('gate','Expert gate',grid.point('gate', 3, 'branches'),routedFF), tensor('up','Expert up',grid.point('up', 3, 'branches'),routedFF),
    tensor('product','Expert SwiGLU output',grid.point('main', 5, 'branches'),routedFF),
    tensor('expertOut','Expert output',grid.point('main', 7, 'branches'),routedHidden),
    tensor('weighted','Weighted output BF16',grid.point('main', 11, 'branches'),routedHidden),
    tensor('routed','Routed sum',grid.point('main', 13, 'branches')),
    tensor('sharedGate','Shared gate',grid.point('sharedGate', 1, 'branches'),sharedFF), tensor('sharedUp','Shared up',grid.point('sharedUp', 1, 'branches'),sharedFF),
    tensor('sharedProduct','Shared SwiGLU output',grid.point('shared', 3, 'branches'),sharedFF),
    tensor('sharedOut','Shared output',grid.point('shared', 5, 'branches')),
    tensor('moeOut','MoE output',grid.point('main', 15, 'branches')), tensor('postFFN','Post-FFN norm',grid.point('main', 17, 'branches')), tensor('output', tensorLabels.output,grid.point('main', 19, 'branches')),
  ],
  operators: [
    // Fused RMSNorm abstraction: save X; internal FP32 statistics/workspaces excluded.
    operator('n1', operatorLabels.n1,'norm',['input'],['norm1'],['input'],grid.point('main', 1, 'attention')),
    operator('m1', operatorLabels.m1,'modulate',['norm1'],['mod1'],['norm1'],grid.point('main', 3, 'attention')),
    operator('qp', operatorLabels.qp,'linear',['mod1'],['q'],['mod1'],grid.point('left', 5, 'attention'),[H,H]),
    operator('kp', operatorLabels.kp,'linear',['mod1'],['k'],['mod1'],grid.point('main', 5, 'attention'),[H,H]),
    operator('vp', operatorLabels.vp,'linear',['mod1'],['v'],['mod1'],grid.point('right', 5, 'attention'),[H,H]),
    operator('qn', operatorLabels.qn,'norm',['q'],['qn'],['q'],grid.point('left', 7, 'attention')),
    operator('kn', operatorLabels.kn,'norm',['k'],['kn'],['k'],grid.point('main', 7, 'attention')),
    operator('qr', operatorLabels.qr,'rope',['qn'],['qr'],[],grid.point('left', 9, 'attention')),
    operator('kr', operatorLabels.kr,'rope',['kn'],['kr'],[],grid.point('main', 9, 'attention')),
    // FA backend assumption, dropout=0. Q/K/V/O retained; LSE is excluded.
    operator('fa', operatorLabels.fa,'attention',['qr','kr','v'],['attention'],['qr','kr','v','attention'],grid.point('main', 11, 'attention')),
    operator('o', operatorLabels.o,'linear',['attention'],['out'],['attention'],grid.point('main', 13, 'attention'),[H,H]),
    operator('postA','post_attn_rmsnorm','norm',['out'],['postAttn'],['out'],grid.point('main', 15, 'attention')),
    operator('resA', operatorLabels.residual,'residual',['input','postAttn'],['residual'],['postAttn'],grid.point('main', 17, 'attention'),undefined,{input:1160}),
    operator('n2', operatorLabels.n2,'norm',['residual'],['norm2'],['residual'],grid.point('main', 1, 'ffn')),
    operator('m2', operatorLabels.m2,'modulate',['norm2'],['mod2'],['norm2'],grid.point('main', 3, 'ffn')),
    // Router computes linear and sigmoid in FP32. Cast backward needs no values.
    operator('castRouter','cast_fp32','cast',['mod2'],['routerInput'],[],grid.point('router', 0, 'branches'),undefined),
    operator('router','router_gemm','linear',['routerInput'],['logits'],['routerInput'],grid.point('router', 2, 'branches'),[E,H]),
    operator('sigmoid','sigmoid','sigmoid',['logits'],['scores'],['scores'],grid.point('router', 4, 'branches')),
    // Discrete choice has no gradient. Gather/normalization needs the selected
    // indices plus scores; replay assumes deterministic selection, including ties.
    { ...operator('topk','group_limited_topk','route',['scores'],['indices'],[],grid.point('router', 6, 'branches')), forwardOnly: true },
    operator('route','gather_norm_scale','route',['scores','indices'],['weights'],['scores','indices'],grid.point('router', 8, 'branches'),undefined,{scores:90}),
    // Logical compute dependencies: weight transport/reordering is implicit.
    // Dispatch shows token routing only; the weight value flows to its consumer.
    // Align branch entries; the index dependency returns upward through a side lane.
    operator('dispatch','dispatch','dispatch',['mod2','indices'],['dispatch','handle'],['handle'],grid.point('main', 0, 'branches'),undefined,{indices:310}),
    // Fused SwiGLU abstraction, matching H3: backward retains gate/up,
    // reconstructing SiLU(gate) rather than retaining its separate output.
    operator('w1','expert_gate_proj','linear',['dispatch'],['gate'],['dispatch'],grid.point('gate', 2, 'branches'),[E,I,H]),
    operator('w3','expert_up_proj','linear',['dispatch'],['up'],['dispatch'],grid.point('up', 2, 'branches'),[E,I,H]),
    operator('sw','swiglu','swiglu',['gate','up'],['product'],['gate','up'],grid.point('main', 4, 'branches')),
    operator('w2','expert_down_proj','linear',['product'],['expertOut'],['product'],grid.point('main', 6, 'branches'),[E,H,I]),
    // Fused multiply/cast: BF16 expert values and FP32 routing weights are
    // sufficient for backward; no full FP32 expert-output allocation is required.
    operator('weight','apply_route_weight','multiply',['expertOut','weights'],['weighted'],['expertOut','weights'],grid.point('main', 10, 'branches'),undefined,{weights:grid.columns.router}),
    // Communication combine sums already-weighted BF16 outputs. Its backward
    // is cached dispatch with the forward handle, not another weighted multiply.
    operator('combine','combine','combine',['weighted','handle'],['routed'],['handle'],grid.point('main', 12, 'branches'),undefined),
    operator('sharedW1','shared_gate_proj','linear',['mod2'],['sharedGate'],['mod2'],grid.point('sharedGate', 0, 'branches'),[I*c.n_shared_experts,H],undefined,{mod2:grid.columns.shared}),
    operator('sharedW3','shared_up_proj','linear',['mod2'],['sharedUp'],['mod2'],grid.point('sharedUp', 0, 'branches'),[I*c.n_shared_experts,H],undefined,{mod2:grid.columns.shared}),
    operator('sharedSw','swiglu','swiglu',['sharedGate','sharedUp'],['sharedProduct'],['sharedGate','sharedUp'],grid.point('shared', 2, 'branches')),
    operator('sharedW2','shared_down_proj','linear',['sharedProduct'],['sharedOut'],['sharedProduct'],grid.point('shared', 4, 'branches'),[H,I*c.n_shared_experts]),
    // Addition backward needs only upstream gradients, not either input's value.
    operator('add','add','add',['routed','sharedOut'],['moeOut'],[],grid.point('main', 14, 'branches'),undefined,{sharedOut:grid.columns.shared}),
    operator('postF','post_ffn_rmsnorm','norm',['moeOut'],['postFFN'],['moeOut'],grid.point('main', 16, 'branches')),
    operator('resF', operatorLabels.residual,'residual',['residual','postFFN'],['output'],['postFFN'],grid.point('main', 18, 'branches'),undefined,{residual:1130}),
  ],
};
