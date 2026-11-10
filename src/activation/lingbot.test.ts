import { describe, expect, it } from 'vitest';
import { analyze, none } from './engine';
import { lingbot } from './lingbot';
import { lingbotConfig } from './lingbot-config';
const p = lingbot.defaults;
const levels = [none, { memory:true,gemm:false,fa:false }, { memory:true,gemm:true,fa:false }, { memory:true,gemm:true,fa:true }];

describe('LingBot Video MoE checkpoint and logical activation ledger', () => {
  it('uses checkpoint MoE dimensions, not constructor dense defaults', () => {
    expect([p.hidden,p.heads,p.headDim,p.intermediate]).toEqual([2048,16,128,768]);
    expect([lingbotConfig.num_experts,lingbotConfig.num_experts_per_tok,lingbotConfig.n_shared_experts]).toEqual([128,8,1]);
    expect(lingbotConfig.decoder_sparse_step).toBe(1);
    expect(lingbotConfig.mlp_only_layers).toEqual([]);
    expect(lingbot.operators.find(op=>op.id==='w1')!.weightShape).toEqual([128,768,2048]);
    expect(lingbot.operators.find(op=>op.id==='sharedW1')!.weightShape).toEqual([768,2048]);
  });
  it('counts active token assignments, shared experts and FP32 values separately', () => {
    const result = analyze(lingbot,p,none);
    const row = (id:string) => result.rows.find(row=>row.id===id)!;
    expect(row('dispatch').dimensions).toEqual([4096*8,2048]);
    expect(row('gate').dimensions).toEqual([4096*8,768]);
    expect(row('sharedGate').dimensions).toEqual([4096,768]);
    expect(row('indices').dimensions).toEqual([4096,8]);
    expect(row('indices').bytes).toBe(4096*8*8);
    expect(row('indices').keep).toBe(true);
    expect(row('scores').bytes).toBe(4096*128*4);
    expect(row('routerInput').bytes).toBe(4096*2048*4);
    expect(row('expertFloat').bytes).toBe(4096*8*2048*4);
    // 16 hidden BF16 values, router FP32 input+scores, BF16 top-k weights,
    // dispatched inputs, three routed and three shared intermediate values,
    // and FP32 expert output used by the weighted-sum backward.
    const perToken = 16*2048*2 + 2048*4 + 128*4 + 8*8 + 8*2 + 8*2048*2 + 3*8*768*2 + 3*768*2 + 8*2048*4;
    expect(result.total).toBe(4096*perToken);
    expect(result.rows.some(row=>row.id==='silu' || row.id==='sharedSilu')).toBe(false);
    expect(lingbot.operators.find(op=>op.id==='sw')!.backward).toEqual(['gate','up']);
    expect(lingbot.operators.find(op=>op.id==='sharedSw')!.backward).toEqual(['sharedGate','sharedUp']);
  });
  it('preserves backward frontiers for all policies and scales with joint token count', () => {
    for (let mask=0;mask<8;mask++) {
      const policy={memory:!!(mask&1),gemm:!!(mask&2),fa:!!(mask&4)};
      const result=analyze(lingbot,p,policy);
      const kept=new Set(result.rows.filter(row=>row.keep).map(row=>row.id));
      function available(id:string):boolean {
        if (kept.has(id)) return true;
        const producer=lingbot.operators.find(op=>op.outputs.includes(id));
        return !!producer && policy[producer.category] && producer.inputs.every(available);
      }
      for (const op of lingbot.operators) for (const id of op.backward) expect(available(id)).toBe(true);
      expect(analyze(lingbot,{...p,sequence:8192},policy).total).toBe(result.total*2);
    }
    const results=levels.map(policy=>analyze(lingbot,p,policy));
    expect(results[2].rows.filter(row=>row.keep).map(row=>row.id)).toEqual(['input','attention']);
    expect(results[3].rows.filter(row=>row.keep).map(row=>row.id)).toEqual(['input']);
    expect(results[2].total).toBe(32*1024**2);
    expect(results[3].total).toBe(16*1024**2);
    results.slice(1).forEach((r,i)=>expect(r.total).toBeLessThan(results[i].total));
  });
  it('models post norms, shared branch and weighted routing dependencies', () => {
    const op=(id:string)=>lingbot.operators.find(op=>op.id===id)!;
    expect(op('postA').backward).toEqual(['out']);
    expect(op('postF').backward).toEqual(['moeOut']);
    expect(op('combine').backward).toEqual(['expertFloat','weights','indices']);
    expect(op('add').backward).toEqual([]);
    expect(op('sigmoid').backward).toEqual(['scores']);
    expect(op('route').backward).toEqual(['scores','indices']);
    expect(op('topk').backward).toEqual([]);
    expect(op('dispatch').backward).toEqual(['indices']);
  });
  it('checks normalized sigmoid route and weighted-sum gradients away from top-k ties', () => {
    // Fixed selected experts: output = sum_i 2.5*sigmoid(z_i)/sum(sigmoid(z))*expert_i.
    // Choice indices are discrete metadata; no gradient through the selection.
    const sigmoid=(x:number)=>1/(1+Math.exp(-x));
    const x=[.2,-.4,1.3,-.7];
    const scores=x.slice(0,2).map(sigmoid), sum=scores[0]+scores[1];
    const weighted=(scores[0]*x[2]+scores[1]*x[3])/sum;
    const gradient=[...scores.map((s,i)=>2.5*(x[i+2]-weighted)/sum*s*(1-s)),...scores.map(s=>2.5*s/sum)];
    const f=(a:number[])=>{const s=a.slice(0,2).map(sigmoid);return 2.5*(s[0]*a[2]+s[1]*a[3])/(s[0]+s[1]);};
    x.forEach((_,i)=>{const plus=[...x],minus=[...x];plus[i]+=1e-5;minus[i]-=1e-5;expect(gradient[i]).toBeCloseTo((f(plus)-f(minus))/2e-5,7);});
  });
});
