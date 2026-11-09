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
    tensor('input','Block input',360,35), tensor('norm1','Attention norm',360,135), tensor('mod1','Attention input',360,235),
    tensor('q','Q',160,340,attention), tensor('k','K',360,340,attention), tensor('v','V',560,340,attention),
    tensor('qn','Q norm',160,440,attention), tensor('kn','K norm',360,440,attention),
    tensor('qr','Q RoPE',160,540,attention), tensor('kr','K RoPE',360,540,attention),
    tensor('attention','FA output',360,650,attention), tensor('out','Attention projection',360,750),
    tensor('postAttn','Post-attention norm',360,850), tensor('residual','Attention residual',360,950),
    tensor('norm2','FFN norm',360,1050), tensor('mod2','MoE input',360,1150),
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
    tensor('moeOut','MoE output',500,2450), tensor('postFFN','Post-FFN norm',500,2550), tensor('output','Block output',500,2800),
  ],
  operators: [
    // Fused RMSNorm abstraction: save X; internal FP32 statistics/workspaces excluded.
    operator('n1','RMSNorm','norm',['input'],['norm1'],['input'],360,85),
    operator('m1','modulate','modulate',['norm1'],['mod1'],['norm1'],360,185),
    operator('qp','q_proj','linear',['mod1'],['q'],['mod1'],160,285,[H,H]),
    operator('kp','k_proj','linear',['mod1'],['k'],['mod1'],360,285,[H,H]),
    operator('vp','v_proj','linear',['mod1'],['v'],['mod1'],560,285,[H,H]),
    operator('qn','Q RMSNorm','norm',['q'],['qn'],['q'],160,390),
    operator('kn','K RMSNorm','norm',['k'],['kn'],['k'],360,390),
    operator('qr','Q RoPE','rope',['qn'],['qr'],[],160,490),
    operator('kr','K RoPE','rope',['kn'],['kr'],[],360,490),
    // FA backend assumption, dropout=0. Q/K/V/O retained; LSE is excluded.
    operator('fa','FA','attention',['qr','kr','v'],['attention'],['qr','kr','v','attention'],360,595),
    operator('o','o_proj','linear',['attention'],['out'],['attention'],360,700,[H,H]),
    operator('postA','post-attn RMSNorm','norm',['out'],['postAttn'],['out'],360,800),
    operator('resA','gated residual','residual',['input','postAttn'],['residual'],['postAttn'],360,900,undefined,{input:1040}),
    operator('n2','RMSNorm','norm',['residual'],['norm2'],['residual'],360,1000),
    operator('m2','modulate','modulate',['norm2'],['mod2'],['norm2'],360,1100),
    // Router computes linear and sigmoid in FP32. Cast backward needs no values.
    operator('castRouter','cast FP32','cast',['mod2'],['routerInput'],[],260,1200),
    operator('router','router GEMM','linear',['routerInput'],['logits'],['routerInput'],260,1300,[E,H]),
    operator('sigmoid','sigmoid','sigmoid',['logits'],['scores'],['scores'],260,1400),
    // Composite gather + normalize + scale (2.5). Indices/bias are external metadata.
    // Choice: 4 groups, top 2 groups, top 8 experts. Gradients use bias-free sigmoid scores.
    operator('route','group top-k / norm','route',['scores'],['weights'],['scores'],260,1500),
    // Weights encode the routing decision for replay; dispatch backward needs indices only.
    operator('dispatch','dispatch','dispatch',['mod2','weights'],['dispatch'],[],260,1600,undefined,{mod2:80}),
    // Eager expert path: separate SiLU and multiply, not a fused SwiGLU backward.
    operator('w1','expert w1','linear',['dispatch'],['gate'],['dispatch'],160,1700,[E,I,H]),
    operator('w3','expert w3','linear',['dispatch'],['up'],['dispatch'],360,1700,[E,I,H]),
    operator('silu','SiLU','silu',['gate'],['silu'],['gate'],160,1800),
    operator('mul','multiply','multiply',['silu','up'],['product'],['silu','up'],260,1900),
    operator('w2','expert w2','linear',['product'],['expertOut'],['product'],260,2000,[E,H,I]),
    operator('castExpert','cast FP32','cast',['expertOut'],['expertFloat'],[],260,2100),
    // Weighted scatter uses FP32 expert values for d(route weights); normalized
    // BF16 weights for d(expert values). Cast result to BF16; indices excluded.
    operator('combine','weighted sum','combine',['expertFloat','weights'],['routed'],['expertFloat','weights'],260,2300,undefined,{weights:60}),
    operator('sharedW1','shared gate_proj','linear',['mod2'],['sharedGate'],['mod2'],710,1700,[I*c.n_shared_experts,H],{mod2:650}),
    operator('sharedW3','shared up_proj','linear',['mod2'],['sharedUp'],['mod2'],910,1700,[I*c.n_shared_experts,H],{mod2:1070}),
    operator('sharedSilu','SiLU','silu',['sharedGate'],['sharedSilu'],['sharedGate'],710,1800),
    operator('sharedMul','multiply','multiply',['sharedSilu','sharedUp'],['sharedProduct'],['sharedSilu','sharedUp'],810,1900),
    operator('sharedW2','shared down_proj','linear',['sharedProduct'],['sharedOut'],['sharedProduct'],810,2000,[H,I*c.n_shared_experts]),
    // Addition backward needs only upstream gradients, not either input's value.
    operator('add','routed + shared','add',['routed','sharedOut'],['moeOut'],[],500,2400),
    operator('postF','post-FFN RMSNorm','norm',['moeOut'],['postFFN'],['moeOut'],500,2500),
    operator('resF','gated residual','residual',['residual','postFFN'],['output'],['postFFN'],500,2750,undefined,{residual:1010}),
  ],
};
