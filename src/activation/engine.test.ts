import { describe, expect, it } from 'vitest';
import { analyze, none } from './engine';
import { h3, type Policy } from './model';
const p = h3.defaults;
const ids = (policy: Policy) => analyze(h3, p, policy).rows.filter(r => r.keep).map(r => r.id);
describe('H3 DiT activation checkpoint dependencies', () => {
  it('matches the user diagram without recomputation, including shared QKV input', () => {
    expect(ids(none)).toEqual(['input','norm1','mod1','q','k','v','qr','kr','attention','out','residual','norm2','mod2','gate','up','swiglu','fc2']);
    const rows = analyze(h3,p,none).rows;
    expect(rows.find(t => t.id === 'q')!.dimensions).toEqual([p.sequence,p.heads,p.headDim]);
    expect(rows.find(t => t.id === 'input')!.dimensions).toEqual([p.sequence,p.hidden]);
    const hiddenBytes = p.sequence*p.hidden*p.bytes;
    const attnBytes = p.sequence*p.heads*p.headDim*p.bytes;
    expect(analyze(h3,p,none).total).toBe(8*hiddenBytes + 6*attnBytes + 3*p.sequence*p.intermediate*p.bytes);
  });
  it('matches the memory-bound-only diagram and GEMM+memory diagram', () => {
    expect(ids({gemm:false,fa:false,memory:true})).toEqual(['input','q','k','v','attention','out','gate','up','fc2']);
    expect(ids({gemm:true,fa:false,memory:true})).toEqual(['input','attention']);
  });
  it('splits packed H3 gate/up logically without duplicating input storage', () => {
    const result = analyze(h3,p,none);
    const input = result.rows.find(row=>row.id==='mod2')!;
    expect(input.reasons).toEqual(['gate_proj.backward','up_proj.backward']);
    expect(input.retained).toBe(p.sequence*p.hidden*p.bytes);
    const policies = [none, {gemm:false,fa:false,memory:true}, {gemm:true,fa:false,memory:true}, {gemm:true,fa:true,memory:true}];
    expect(policies.map(policy=>analyze(h3,p,policy).total/1024**2)).toEqual([1008,574,98,42]);
  });
  it('all-on retains only the layer boundary', () => {
    expect(ids({gemm:true,fa:true,memory:true})).toEqual(['input']);
  });
  it('FA-only drops its output but retains Q/K/V backward dependencies', () => {
    const saved=ids({gemm:false,fa:true,memory:false});
    expect(saved).not.toContain('attention');
    for (const id of ['qr','kr','v']) expect(saved).toContain(id);
  });
  it('all 8 policies have complete retained dependency frontiers and linear sequence scaling', () => {
    for(let mask=0;mask<8;mask++) {
      const policy={gemm:!!(mask&1),fa:!!(mask&2),memory:!!(mask&4)};
      const r=analyze(h3,p,policy), saved=new Set(r.rows.filter(t=>t.keep).map(t=>t.id));
      function reachable(id:string):boolean {
        if(saved.has(id))return true;
        const op=h3.operators.find(o=>o.outputs.includes(id));
        return !!op && policy[op.category] && op.inputs.every(reachable);
      }
      for(const op of h3.operators)for(const input of op.backward)expect(reachable(input)).toBe(true);
      expect(analyze(h3,{...p,sequence:p.sequence*2},policy).total).toBe(r.total*2);
      expect(r.total).toBeLessThanOrEqual(analyze(h3,p,none).total);
    }
  });
  it('rejects invalid shapes and supports GQA widths independently of hidden size', () => {
    expect(()=>analyze(h3,{...p,sequence:0},none)).toThrow();
    expect(()=>analyze(h3,{...p,kvHeads:3},none)).toThrow();
    const r=analyze(h3,{...p,kvHeads:8},none);
    expect(r.rows.find(t=>t.id==='k')!.bytes*7).toBe(r.rows.find(t=>t.id==='q')!.bytes);
  });
});

describe('model dependency integrity', () => {
  it('rejects duplicate tensors and missing / cyclic producers before any policy runs', () => {
    expect(() => analyze({...h3,tensors:[...h3.tensors,h3.tensors[0]]},p,none)).toThrow(/Duplicate tensor/);
    expect(() => analyze({...h3,operators:h3.operators.slice(1)},p,none)).toThrow(/producer/);
    const ops=h3.operators.map(o=>({...o}));ops[0].inputs=['output'];
    expect(() => analyze({...h3,operators:ops},p,none)).toThrow(/cyclic producer/);
  });
  it('rejects unrelated backward references and invalid tensor shapes', () => {
    const ops=h3.operators.map(o=>({...o}));ops[0].backward=['fc2'];
    expect(() => analyze({...h3,operators:ops},p,none)).toThrow(/backward dependency/);
    const ts=h3.tensors.map(t=>({...t}));ts[0].shape=()=>[1.5,2];
    expect(() => analyze({...h3,tensors:ts},p,none)).toThrow(/Invalid shape/);
  });
});
