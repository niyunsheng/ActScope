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
const tensor = (id: string, label: string, x: number, y: number, shape = hidden, dtype: NonNullable<Tensor['dtype']> = 'BF16'): Tensor =>
  ({ id, label, x, y, shape, dtype, formula: '' });
const operator = (id: string, label: string, rule: RuleKey, inputs: string[], outputs: string[], backward: string[], x: number, y: number, weightShape?: number[], inputLanes?: Record<string, number>, inputRoutes?: Operator['inputRoutes']): Operator =>
  ({ id, label, rule, category: rule === 'linear' ? 'gemm' : rule === 'attention' ? 'fa' : 'memory', inputs, outputs, backward, x, y, weightShape, inputLanes, inputRoutes });

export const lingbot: Model = {
  id: 'lingbot-video-moe', name: 'LingBot Video · MoE',
  layout: { width: 1200, height: 3020 },
  regions: [
    { label: 'Router', x: 40, y: 1200, width: 260, height: 1410 },
    { label: 'Routed experts', x: 350, y: 1200, width: 340, height: 1410 },
    { label: 'Shared expert', x: 750, y: 1200, width: 340, height: 1410 },
  ],
  defaults: { sequence: 4096, hidden: H, intermediate: I, heads: c.num_attention_heads, kvHeads: c.num_attention_heads, headDim: H / c.num_attention_heads, bytes: 2 },
  facts: [{label:'Routed experts',value:E}, {label:'Top-k experts',value:K}, {label:'Groups',value:c.n_group}, {label:'Selected groups',value:c.topk_group}, {label:'Shared experts',value:c.n_shared_experts}],
  note: 'T = video + text tokens. Logical activation estimate; includes INT64 top-k indices; excludes permutation buffers, expert padding, norm/FA auxiliary state and conditioning.',
  boundary: ['input'],
  tensors: [
    tensor('input', tensorLabels.input,520,35), tensor('norm1', tensorLabels.norm1,520,135), tensor('mod1', tensorLabels.mod1,520,235),
    tensor('q', tensorLabels.q,360,340,attention), tensor('k', tensorLabels.k,520,340,attention), tensor('v', tensorLabels.v,680,340,attention),
    tensor('qn', tensorLabels.qn,360,440,attention), tensor('kn', tensorLabels.kn,520,440,attention),
    tensor('qr', tensorLabels.qr,360,540,attention), tensor('kr', tensorLabels.kr,520,540,attention),
    tensor('attention', tensorLabels.attention,520,650,attention), tensor('out', tensorLabels.out,520,750),
    tensor('postAttn','Post-attention norm',520,850), tensor('residual', tensorLabels.residual,520,950),
    tensor('norm2', tensorLabels.norm2,520,1050), tensor('mod2', tensorLabels.mod2,520,1150),
    tensor('routerInput','Router input FP32',170,1350,hidden,'FP32'),
    tensor('logits','Router logits',170,1450,p=>[p.sequence,E],'FP32'),
    tensor('scores','Sigmoid scores',170,1550,p=>[p.sequence,E],'FP32'),
    tensor('indices','Top-k indices',170,1650,p=>[p.sequence,K],'INT64'),
    tensor('weights','Top-k weights',170,1750,p=>[p.sequence,K]),
    tensor('dispatch','Dispatched tokens',520,1900,routedHidden),
    tensor('gate','Expert gate',440,2000,routedFF), tensor('up','Expert up',600,2000,routedFF),
    tensor('product','Expert SwiGLU output',520,2150,routedFF),
    tensor('expertOut','Expert output',520,2300,routedHidden),
    tensor('expertFloat','Expert output FP32',520,2400,routedHidden,'FP32'),
    tensor('routed','Routed sum',520,2600),
    tensor('sharedGate','Shared gate',830,1350,sharedFF), tensor('sharedUp','Shared up',990,1350,sharedFF),
    tensor('sharedProduct','Shared SwiGLU output',910,1500,sharedFF),
    tensor('sharedOut','Shared output',910,1650),
    tensor('moeOut','MoE output',520,2700), tensor('postFFN','Post-FFN norm',520,2800), tensor('output', tensorLabels.output,520,2950),
  ],
  operators: [
    // Fused RMSNorm abstraction: save X; internal FP32 statistics/workspaces excluded.
    operator('n1', operatorLabels.n1,'norm',['input'],['norm1'],['input'],520,85),
    operator('m1', operatorLabels.m1,'modulate',['norm1'],['mod1'],['norm1'],520,185),
    operator('qp', operatorLabels.qp,'linear',['mod1'],['q'],['mod1'],360,285,[H,H]),
    operator('kp', operatorLabels.kp,'linear',['mod1'],['k'],['mod1'],520,285,[H,H]),
    operator('vp', operatorLabels.vp,'linear',['mod1'],['v'],['mod1'],680,285,[H,H]),
    operator('qn', operatorLabels.qn,'norm',['q'],['qn'],['q'],360,390),
    operator('kn', operatorLabels.kn,'norm',['k'],['kn'],['k'],520,390),
    operator('qr', operatorLabels.qr,'rope',['qn'],['qr'],[],360,490),
    operator('kr', operatorLabels.kr,'rope',['kn'],['kr'],[],520,490),
    // FA backend assumption, dropout=0. Q/K/V/O retained; LSE is excluded.
    operator('fa', operatorLabels.fa,'attention',['qr','kr','v'],['attention'],['qr','kr','v','attention'],520,595),
    operator('o', operatorLabels.o,'linear',['attention'],['out'],['attention'],520,700,[H,H]),
    operator('postA','post_attn_rmsnorm','norm',['out'],['postAttn'],['out'],520,800),
    operator('resA', operatorLabels.residual,'residual',['input','postAttn'],['residual'],['postAttn'],520,900,undefined,{input:1160}),
    operator('n2', operatorLabels.n2,'norm',['residual'],['norm2'],['residual'],520,1000),
    operator('m2', operatorLabels.m2,'modulate',['norm2'],['mod2'],['norm2'],520,1100),
    // Router computes linear and sigmoid in FP32. Cast backward needs no values.
    operator('castRouter','cast_fp32','cast',['mod2'],['routerInput'],[],170,1300,undefined,undefined,{mod2:[[520,1160],[520,1180],[170,1180],[170,1284]]}),
    operator('router','router_gemm','linear',['routerInput'],['logits'],['routerInput'],170,1400,[E,H]),
    operator('sigmoid','sigmoid','sigmoid',['logits'],['scores'],['scores'],170,1500),
    // Discrete choice has no gradient. Gather/normalization needs the selected
    // indices plus scores; replay assumes deterministic selection, including ties.
    operator('topk','group_limited_topk','route',['scores'],['indices'],[],170,1600),
    operator('route','gather_norm_scale','route',['scores','indices'],['weights'],['scores','indices'],170,1700,undefined,undefined,{scores:[[170,1560],[90,1560],[90,1700],[108,1700]]}),
    // Indices stand in for the logical permutation; concrete permutation buffers are excluded.
    operator('dispatch','dispatch','dispatch',['mod2','indices'],['dispatch'],['indices'],520,1850,undefined,undefined,{mod2:[[520,1160],[520,1834]],indices:[[170,1660],[310,1660],[310,1850],[458,1850]]}),
    // Fused SwiGLU abstraction, matching H3: backward retains gate/up,
    // reconstructing SiLU(gate) rather than retaining its separate output.
    operator('w1','expert_gate_proj','linear',['dispatch'],['gate'],['dispatch'],440,1950,[E,I,H]),
    operator('w3','expert_up_proj','linear',['dispatch'],['up'],['dispatch'],600,1950,[E,I,H]),
    operator('sw','swiglu','swiglu',['gate','up'],['product'],['gate','up'],520,2100),
    operator('w2','expert_down_proj','linear',['product'],['expertOut'],['product'],520,2250,[E,H,I]),
    operator('castExpert','cast_fp32','cast',['expertOut'],['expertFloat'],[],520,2350),
    // Weighted scatter uses FP32 expert values for d(route weights); normalized
    // BF16 weights for d(expert values). Indices identify the destination token slots.
    operator('combine','combine','combine',['expertFloat','weights','indices'],['routed'],['expertFloat','weights','indices'],520,2550,undefined,undefined,{weights:[[170,1760],[170,1790],[330,1790],[330,2550],[458,2550]],indices:[[170,1660],[310,1660],[310,2530],[440,2530],[440,2550],[458,2550]]}),
    operator('sharedW1','shared_gate_proj','linear',['mod2'],['sharedGate'],['mod2'],830,1300,[I*c.n_shared_experts,H],undefined,{mod2:[[520,1160],[520,1180],[910,1180],[910,1260],[830,1260],[830,1284]]}),
    operator('sharedW3','shared_up_proj','linear',['mod2'],['sharedUp'],['mod2'],990,1300,[I*c.n_shared_experts,H],undefined,{mod2:[[520,1160],[520,1180],[910,1180],[910,1260],[990,1260],[990,1284]]}),
    operator('sharedSw','swiglu','swiglu',['sharedGate','sharedUp'],['sharedProduct'],['sharedGate','sharedUp'],910,1450),
    operator('sharedW2','shared_down_proj','linear',['sharedProduct'],['sharedOut'],['sharedProduct'],910,1600,[H,I*c.n_shared_experts]),
    // Addition backward needs only upstream gradients, not either input's value.
    operator('add','add','add',['routed','sharedOut'],['moeOut'],[],520,2650,undefined,undefined,{sharedOut:[[910,1660],[910,2650],[582,2650]]}),
    operator('postF','post_ffn_rmsnorm','norm',['moeOut'],['postFFN'],['moeOut'],520,2750),
    operator('resF', operatorLabels.residual,'residual',['residual','postFFN'],['output'],['postFFN'],520,2900,undefined,{residual:1130}),
  ],
};
