import { describe, expect, it } from 'vitest';
import { analyze, none } from './engine';
import { lingFlash, lingFlashKdaMoe, lingFlashMlaMoe, lingFlashKdaDense } from './ling-flash';
import { lingFlashConfig as c } from './ling-flash-config';

const levels = [none, { memory: true, gemm: false, fa: false }, { memory: true, gemm: true, fa: false }, { memory: true, gemm: true, fa: true }];
const variants = [lingFlashKdaMoe, lingFlashMlaMoe, lingFlashKdaDense];
const H = 2560, N = 32, D = 128, K = 8, E = 512, I = 768;
// Independent scalar accounting: router input/scores/weights/indices, routed inputs
// and outputs, three SwiGLU tensors per routed/shared expert assignment.
const moeBytesPerToken = H*4 + E*4 + K*4 + K*8 + 2*K*H*2 + 3*K*I*2 + 3*I*2;
const kdaBytesPerToken = 4*H*2 + 10*N*D*2 + N*4;
const mlaBytesPerToken = 4*H*2 + 2*512*2 + 2*N*192*2 + 3*N*128*2 + N*4 + N*2;

describe('Ling 3.0 Flash checkpoint architecture', () => {
  it('covers the 42-layer 5:1 schedule and the first two dense layers', () => {
    const counts = { dense: 0, kdaMoe: 0, mlaMoe: 0 };
    for (let i = 0; i < c.num_hidden_layers; i++) {
      if (i < c.first_k_dense_replace) counts.dense++;
      else if ((i+1) % c.layer_group_size === 0) counts.mlaMoe++;
      else counts.kdaMoe++;
    }
    expect(counts).toEqual({ dense: 2, kdaMoe: 33, mlaMoe: 7 });
    expect(lingFlash.variants).toEqual(variants);
    expect([c.hidden_size,c.num_attention_heads,c.num_experts,c.num_experts_per_tok,c.num_shared_experts]).toEqual([H,N,E,K,1]);
    expect(c.q_lora_rank).toBeNull();
    expect(c.no_kda_lora).toBe(true);
    expect(c.attention_dropout).toBe(0);
  });

  it('models KDA convolutions, FP32 beta, diagonal decay and gated per-head norm', () => {
    const model = lingFlashKdaMoe, result = analyze(model, model.defaults, none);
    const row = (id: string) => result.rows.find(row => row.id === id)!;
    const op = (id: string) => model.operators.find(op => op.id === id)!;
    expect(row('qConv').dimensions).toEqual([4096,N,D]);
    expect(row('decay').dimensions).toEqual([4096,N,D]);
    expect(row('outputGate').dimensions).toEqual([4096,N,D]);
    expect(row('beta').dimensions).toEqual([4096,N]);
    expect(row('beta').bytes).toBe(4096*N*4);
    expect(op('kda').category).toBe('fa');
    expect(op('kda').backward).toEqual(['qConv','kConv','vConv','decay','beta']);
    expect(op('gatedNorm').backward).toEqual(['attention','outputGate']);
    expect(op('qp').weightShape).toEqual([4096,2560]);
    expect(op('bProj').weightShape).toEqual([32,2560]);
    expect(c.short_conv_kernel_size).toBe(4);
    expect(model.operators.some(op => op.rule === 'rope' || op.rule === 'attention')).toBe(false);
    expect(result.total).toBe(4096*(kdaBytesPerToken+moeBytesPerToken));
  });

  it('keeps MLA latent, shared positional key and expanded Q/K/V dimensions distinct', () => {
    const model = lingFlashMlaMoe, result = analyze(model, model.defaults, none);
    const row = (id: string) => result.rows.find(row => row.id === id)!;
    const op = (id: string) => model.operators.find(op => op.id === id)!;
    expect(row('latent').dimensions).toEqual([4096,512]);
    expect(row('kPos').dimensions).toEqual([4096,1,64]);
    expect(row('kRot').bytes).toBe(4096*64*2);
    expect(row('q').dimensions).toEqual([4096,32,192]);
    expect(row('k').dimensions).toEqual([4096,32,192]);
    expect(row('v').dimensions).toEqual([4096,32,128]);
    expect(row('attnGate').dimensions).toEqual([4096,32]);
    expect(row('attnGate').dtype).toBe('BF16');
    expect(row('gateFP32').bytes).toBe(4096*32*4);
    expect(op('gateSigmoid').backward).toEqual(['gateFP32']);
    expect(op('qp').weightShape).toEqual([6144,2560]);
    expect(op('kvA').weightShape).toEqual([576,2560]);
    expect(op('kvB').weightShape).toEqual([8192,512]);
    expect(op('o').weightShape).toEqual([2560,4096]);
    expect(op('fa').backward).toEqual(['q','k','v','attention']);
    expect(op('attnWeight').backward).toEqual(['attention','attnGate']);
    // Non-RoPE / RoPE slices are disjoint; no full duplicate packed Q/KV tensor.
    expect(row('qNope').bytes+row('qPos').bytes).toBe(row('q').bytes);
    expect(result.total).toBe(4096*(mlaBytesPerToken+moeBytesPerToken));
  });

  it('uses 8 routed assignments, one shared expert and unbiased normalized sigmoid weights', () => {
    const model = lingFlashKdaMoe, result = analyze(model, model.defaults, none);
    const row = (id: string) => result.rows.find(row => row.id === id)!;
    const op = (id: string) => model.operators.find(op => op.id === id)!;
    expect(row('dispatch').dimensions).toEqual([4096*8,2560]);
    expect(row('gate').dimensions).toEqual([4096*8,768]);
    expect(row('sharedGate').dimensions).toEqual([4096,768]);
    expect(row('scores').bytes).toBe(4096*512*4);
    expect(row('weights').bytes).toBe(4096*8*4);
    expect(row('indices').bytes).toBe(4096*8*8);
    expect(row('weighted').dtype).toBe('FP32');
    expect(row('routed').dtype).toBe('BF16');
    expect(op('w1').weightShape).toEqual([512,768,2560]);
    expect(op('topk').forwardOnly).toBe(true);
    expect(op('route').inputs).toEqual(['scores','indices']);
    expect(op('weight').backward).toEqual(['expertOut','weights']);
    expect(op('combine').backward).toEqual(['indices']);
    expect([c.n_group,c.topk_group,c.routed_scaling_factor]).toEqual([8,4,2.5]);
    expect(model.operators.filter(op => op.rule === 'add').every(op => op.backward.length === 0)).toBe(true);
    expect(model.operators.some(op => op.rule === 'modulate' || op.rule === 'residual')).toBe(false);
  });

  it('uses a 6144-wide dense FFN in layers 1–2 without any routing storage', () => {
    const model = lingFlashKdaDense, result = analyze(model, model.defaults, none);
    expect(model.defaults.intermediate).toBe(6144);
    expect(result.rows.find(row => row.id === 'gate')!.dimensions).toEqual([4096,6144]);
    expect(model.operators.find(op => op.id === 'w1')!.weightShape).toEqual([6144,2560]);
    expect(result.rows.some(row => ['indices','dispatch','sharedGate'].includes(row.id))).toBe(false);
    expect(result.total).toBe(4096*(kdaBytesPerToken+3*6144*2));
  });

  for (const model of variants) it(`${model.name}: complete backward frontiers for every recomputation policy`, () => {
    for (let mask=0; mask<8; mask++) {
      const policy = { memory: !!(mask&1), gemm: !!(mask&2), fa: !!(mask&4) };
      const result = analyze(model, model.defaults, policy);
      const saved = new Set(result.rows.filter(row => row.keep).map(row => row.id));
      const available = (id: string): boolean => {
        if (saved.has(id)) return true;
        const producer = model.operators.find(op => op.outputs.includes(id));
        return !!producer && policy[producer.category] && producer.inputs.every(available);
      };
      for (const op of model.operators) for (const id of op.backward) expect(available(id)).toBe(true);
      expect(analyze(model, {...model.defaults,sequence:8192}, policy).total).toBe(result.total*2);
    }
    const results = levels.map(policy => analyze(model, model.defaults, policy));
    results.slice(1).forEach((r,i) => expect(r.total).toBeLessThan(results[i].total));
    const ids = (index: number) => results[index].rows.filter(row => row.keep).map(row => row.id).sort();
    const routing = model === lingFlashKdaDense ? [] : ['indices'];
    expect(ids(2)).toEqual(['input','attention',...routing].sort());
    expect(ids(3)).toEqual(['input',...routing].sort());
  });
});
