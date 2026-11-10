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
const tensor = (id: string, label: string, x: number, y: number, shape = hidden, dtype: 'BF16' | 'FP32' = 'BF16'): Tensor =>
  ({ id, label, x, y, shape, dtype, formula: '' });
const operator = (id: string, label: string, rule: RuleKey, inputs: string[], outputs: string[], backward: string[], x: number, y: number, weightShape?: number[], inputLanes?: Record<string, number>): Operator =>
  ({ id, label, rule, category: rule === 'linear' ? 'gemm' : rule === 'attention' ? 'fa' : 'memory', inputs, outputs, backward, x, y, weightShape, inputLanes });

export const lingbot: Model = {
  id: 'lingbot-video-moe', name: 'LingBot Video · MoE',
  layout: { width: 1100, height: 2870 },
  defaults: { sequence: 4096, hidden: H, intermediate: I, heads: c.num_attention_heads, kvHeads: c.num_attention_heads, headDim: H / c.num_attention_heads, bytes: 2 },
  facts: [{label:'Routed experts',value:E}, {label:'Top-k',value:K}, {label:'Shared experts',value:c.n_shared_experts}],
  note: 'T = video + text tokens. Logical activation estimate; excludes routing indices, expert padding, norm/FA auxiliary state and conditioning.',
  boundary: ['input'],
  tensors: [
    tensor('input', tensorLabels.input,360,35), tensor('norm1', tensorLabels.norm1,360,135), tensor('mod1', tensorLabels.mod1,360,235),
    tensor('q', tensorLabels.q,200,340,attention), tensor('k', tensorLabels.k,360,340,attention), tensor('v', tensorLabels.v,520,340,attention),
    tensor('qn', tensorLabels.qn,200,440,attention), tensor('kn', tensorLabels.kn,360,440,attention),
    tensor('qr', tensorLabels.qr,200,540,attention), tensor('kr', tensorLabels.kr,360,540,attention),
    tensor('attention', tensorLabels.attention,360,650,attention), tensor('out', tensorLabels.out,360,750),
    tensor('postAttn','Post-attention norm',360,850), tensor('residual', tensorLabels.residual,360,950),
    tensor('norm2', tensorLabels.norm2,360,1050), tensor('mod2', tensorLabels.mod2,360,1150),
    tensor('routerInput','Router input FP32',260,1250,hidden,'FP32'),
    tensor('logits','Router logits',260,1350,p=>[p.sequence,E],'FP32'),
    tensor('scores','Sigmoid scores',260,1450,p=>[p.sequence,E],'FP32'),
    tensor('weights','Top-k weights',260,1550,p=>[p.sequence,K]),
    tensor('dispatch','Dispatched tokens',260,1650,routedHidden),
    tensor('gate','Expert gate',160,1750,routedFF), tensor('up','Expert up',360,1750,routedFF),
    tensor('silu','Expert SiLU',160,1850,routedFF), tensor('product','Expert product',260,1950,routedFF),
    tensor('expertOut','Expert output',260,2050,routedHidden),
    tensor('expertFloat','Expert output FP32',260,2150,routedHidden,'FP32'),
    tensor('routed','Routed sum',260,2350),
    tensor('sharedGate','Shared gate',710,1750,sharedFF), tensor('sharedUp','Shared up',910,1750,sharedFF),
    tensor('sharedSilu','Shared SiLU',710,1850,sharedFF), tensor('sharedProduct','Shared product',810,1950,sharedFF),
    tensor('sharedOut','Shared output',810,2050),
    tensor('moeOut','MoE output',500,2450), tensor('postFFN','Post-FFN norm',500,2550), tensor('output', tensorLabels.output,500,2800),
  ],
  operators: [
    // Fused RMSNorm abstraction: save X; internal FP32 statistics/workspaces excluded.
    operator('n1', operatorLabels.n1,'norm',['input'],['norm1'],['input'],360,85),
    operator('m1', operatorLabels.m1,'modulate',['norm1'],['mod1'],['norm1'],360,185),
    operator('qp', operatorLabels.qp,'linear',['mod1'],['q'],['mod1'],200,285,[H,H]),
    operator('kp', operatorLabels.kp,'linear',['mod1'],['k'],['mod1'],360,285,[H,H]),
    operator('vp', operatorLabels.vp,'linear',['mod1'],['v'],['mod1'],520,285,[H,H]),
    operator('qn', operatorLabels.qn,'norm',['q'],['qn'],['q'],200,390),
    operator('kn', operatorLabels.kn,'norm',['k'],['kn'],['k'],360,390),
    operator('qr', operatorLabels.qr,'rope',['qn'],['qr'],[],200,490),
    operator('kr', operatorLabels.kr,'rope',['kn'],['kr'],[],360,490),
    // FA backend assumption, dropout=0. Q/K/V/O retained; LSE is excluded.
    operator('fa', operatorLabels.fa,'attention',['qr','kr','v'],['attention'],['qr','kr','v','attention'],360,595),
    operator('o', operatorLabels.o,'linear',['attention'],['out'],['attention'],360,700,[H,H]),
    operator('postA','post_attn_rmsnorm','norm',['out'],['postAttn'],['out'],360,800),
    operator('resA', operatorLabels.residual,'residual',['input','postAttn'],['residual'],['postAttn'],360,900,undefined,{input:1040}),
    operator('n2', operatorLabels.n2,'norm',['residual'],['norm2'],['residual'],360,1000),
    operator('m2', operatorLabels.m2,'modulate',['norm2'],['mod2'],['norm2'],360,1100),
    // Router computes linear and sigmoid in FP32. Cast backward needs no values.
    operator('castRouter','cast_fp32','cast',['mod2'],['routerInput'],[],260,1200),
    operator('router','router_gemm','linear',['routerInput'],['logits'],['routerInput'],260,1300,[E,H]),
    operator('sigmoid','sigmoid','sigmoid',['logits'],['scores'],['scores'],260,1400),
    // Composite gather + normalize + scale (2.5). Indices/bias are external metadata.
    // Choice: 4 groups, top 2 groups, top 8 experts. Gradients use bias-free sigmoid scores.
    operator('route','group_topk_norm','route',['scores'],['weights'],['scores'],260,1500),
    // Weights encode the routing decision for replay; dispatch backward needs indices only.
    operator('dispatch','dispatch','dispatch',['mod2','weights'],['dispatch'],[],260,1600,undefined,{mod2:80}),
    // Eager expert path: separate SiLU and multiply, not a fused SwiGLU backward.
    operator('w1','expert_w1','linear',['dispatch'],['gate'],['dispatch'],160,1700,[E,I,H]),
    operator('w3','expert_w3','linear',['dispatch'],['up'],['dispatch'],360,1700,[E,I,H]),
    operator('silu','silu','silu',['gate'],['silu'],['gate'],160,1800),
    operator('mul','multiply','multiply',['silu','up'],['product'],['silu','up'],260,1900),
    operator('w2','expert_w2','linear',['product'],['expertOut'],['product'],260,2000,[E,H,I]),
    operator('castExpert','cast_fp32','cast',['expertOut'],['expertFloat'],[],260,2100),
    // Weighted scatter uses FP32 expert values for d(route weights); normalized
    // BF16 weights for d(expert values). Cast result to BF16; indices excluded.
    operator('combine','weighted_sum','combine',['expertFloat','weights'],['routed'],['expertFloat','weights'],260,2300,undefined,{weights:60}),
    operator('sharedW1','shared_gate_proj','linear',['mod2'],['sharedGate'],['mod2'],710,1700,[I*c.n_shared_experts,H],{mod2:650}),
    operator('sharedW3','shared_up_proj','linear',['mod2'],['sharedUp'],['mod2'],910,1700,[I*c.n_shared_experts,H],{mod2:1070}),
    operator('sharedSilu','silu','silu',['sharedGate'],['sharedSilu'],['sharedGate'],710,1800),
    operator('sharedMul','multiply','multiply',['sharedSilu','sharedUp'],['sharedProduct'],['sharedSilu','sharedUp'],810,1900),
    operator('sharedW2','shared_down_proj','linear',['sharedProduct'],['sharedOut'],['sharedProduct'],810,2000,[H,I*c.n_shared_experts]),
    // Addition backward needs only upstream gradients, not either input's value.
    operator('add','add','add',['routed','sharedOut'],['moeOut'],[],500,2400),
    operator('postF','post_ffn_rmsnorm','norm',['moeOut'],['postFFN'],['moeOut'],500,2500),
    operator('resF', operatorLabels.residual,'residual',['residual','postFFN'],['output'],['postFFN'],500,2750,undefined,{residual:1010}),
  ],
};
