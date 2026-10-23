import { describe,it,expect } from 'vitest';
import { calculate,defaults,expression,shapeOf } from './engine';
import { preset } from './presets';
describe('safe expressions',()=>{
 it('resolves precedence and parentheses',()=>{expect(expression('(H + 2) * 3 / 2',{H:4})).toBe(9);expect(shapeOf('[H, 2*H]',{H:4})).toEqual([4,8]);});
 it('rejects code, unknown variables, fractional dimensions and overflow',()=>{for(const expr of ['window.alert(1)','H.x','2**3','1/0','missing']) expect(()=>expression(expr,{})).toThrow();expect(()=>shapeOf('3/2',{})).toThrow();});
});
describe('memory accounting',()=>{
 it('matches hand-counted Qwen3 dense decoder parameters',()=>{const d=calculate(preset('dense'),{...defaults,gpus:1,ep:1}); const attention=2*4096*4096+2*1024*4096;const ffn=3*4096*12288; const norms=2*4096+2*128;expect(d.globalParams).toBe((attention+ffn+norms)*36);expect(d.total).toBe(d.globalParams*18);});
 it('uses distinct dense and expert state groups',()=>{const d=calculate(preset('moe'),{...defaults,gpus:128,pp:4,ep:32});expect(d.dp).toBe(32);expect(d.edp).toBe(1); const router=d.rows.find(t=>t.name==='moe.router.weight')!;expect(router.param).toBe(128*4096*2*24);expect(router.optim).toBe(128*4096*12/32*24);const expert=d.rows.find(t=>t.kind==='expert')!;expect(expert.localCount).toBe(expert.count/32);expect(expert.so).toBe(1);});
 it('full shards both classes without double dividing EP',()=>{const m=preset('moe');const d=calculate(m,{...defaults,gpus:128,ep:32,shard:'full'});expect(d.total).toBeCloseTo(d.globalParams*18/128);});
 it('rejects splitting part of an attention head',()=>{expect(()=>calculate(preset('moe'),{...defaults,tp:8})).toThrow(/KV head/);});
 it('rejects impossible groups and dimensions',()=>{expect(()=>calculate(preset('dense'),{...defaults,gpus:7,tp:2})).toThrow();expect(()=>calculate(preset('moe'),{...defaults,ep:3})).toThrow();const m=preset('custom');m.tensors[0].shape='7, hidden_size';expect(()=>calculate(m,{...defaults,gpus:2,tp:2,ep:1})).toThrow();});
 it('does not divide replicated weights by TP; frozen params have no gradients or optimizer',()=>{const m=preset('custom');m.tensors[0].kind='replicated';m.tensors[0].trainable=false;const d=calculate(m,{...defaults,tp:2,ep:1,shard:'none'});expect(d.total).toBe(1024*4096*2);});
 it('counts one storage once for multiple saving operators and partitions policies',()=>{const m=preset('custom');m.activations=[{id:'act',name:'input',savedBy:'fc1.backward,fc2.backward',shape:'tokens_per_rank, hidden_size',bytes:2,policy:'offload'}];const d=calculate(m,defaults);expect(d.activation).toEqual({all:2097152,keep:0,recompute:0,offload:2097152});});
});
