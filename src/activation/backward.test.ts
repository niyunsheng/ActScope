import { describe, it, expect } from 'vitest';
import { h3 } from './model';
// Independent finite-difference checks of the derivatives underlying the graph.
// These validate mathematical dependencies, not CUDA kernel storage allocations.
const dot = (a:number[],b:number[]) => a.reduce((s,x,i)=>s+x*b[i],0);
const sigmoid = (x:number) => 1/(1+Math.exp(-x));
const silu = (x:number) => x*sigmoid(x);
const siluPrime = (x:number) => sigmoid(x)*(1+x*(1-sigmoid(x)));
function check(f:(x:number[])=>number,x:number[],gradient:number[]) {
  x.forEach((_,i)=>{
    const plus=x.slice(),minus=x.slice(),eps=1e-5;
    plus[i]+=eps;minus[i]-=eps;
    expect(gradient[i]).toBeCloseTo((f(plus)-f(minus))/(2*eps),7);
  });
}
describe('backward activation dependencies: numerical derivatives',()=>{
  it('linear weight gradient uses X, input gradient uses W, neither uses output',()=>{
    const x=[.3,-.8],w=[.6,.2],dy=.7;
    check(v=>dot(v.slice(0,2),v.slice(2))*dy,[...x,...w],[...w.map(v=>v*dy),...x.map(v=>v*dy)]);
    for(const op of h3.operators.filter(o=>o.rule==='linear'))expect(op.backward).toEqual(op.inputs);
  });
  it('modulation needs X for trainable scale, not shift/output values',()=>{
    const x=[.3,-.8],scale=.4,shift=.2,dy=[.7,-.2];
    check(v=>dot(v.slice(0,2).map(x=>x*(1+v[2])+v[3]),dy),[...x,scale,shift],
      [...dy.map(g=>g*(1+scale)),dot(dy,x),dy[0]+dy[1]]);
    for(const op of h3.operators.filter(o=>o.rule==='modulate'))expect(op.backward).toEqual(op.inputs);
  });
  it('gated residual needs branch for gate grad but no residual value for backward',()=>{
    const residual=[.3,-.8],branch=[.6,.2],gate=.4,dy=[.7,-.2];
    check(v=>dot(v.slice(0,2).map((x,i)=>x+v[4]*v[2+i]),dy),[...residual,...branch,gate],
      [...dy,...dy.map(g=>g*gate),dot(dy,branch)]);
    for(const op of h3.operators.filter(o=>o.rule==='residual'))expect(op.backward).toEqual([op.inputs[1]]);
  });
  it('fixed RoPE backward is the transpose rotation and does not depend on X',()=>{
    const c=Math.cos(.4),s=Math.sin(.4),dy=[.7,-.2];
    check(x=>dot([c*x[0]-s*x[1],s*x[0]+c*x[1]],dy),[.3,-.8],[c*dy[0]+s*dy[1],-s*dy[0]+c*dy[1]]);
    for(const op of h3.operators.filter(o=>o.rule==='rope'))expect(op.backward).toEqual([]);
  });
  it('fused SwiGLU backward uses gate and up without retaining SiLU output',()=>{
    const gate=-.8,up=.4,dy=.7;
    check(x=>dy*silu(x[0])*x[1],[gate,up],[dy*up*siluPrime(gate),dy*silu(gate)]);
    expect(h3.operators.find(o=>o.rule==='swiglu')!.backward).toEqual(['gate','up']);
  });
  for(const kind of ['rms','layer'])it(`${kind} norm uses X, statistics and gamma without output`,()=>{
    const x=[.3,-.8,.6],gamma=[.4,.7,.2],dy=[.7,-.2,.5],eps=.01,n=x.length;
    const mean=kind==='layer'?x.reduce((a,b)=>a+b)/n:0;
    const centered=x.map(v=>v-mean),rstd=1/Math.sqrt(dot(centered,centered)/n+eps);
    const g=dy.map((v,i)=>v*gamma[i]),gm=kind==='layer'?g.reduce((a,b)=>a+b)/n:0;
    const dx=g.map((v,i)=>rstd*(v-gm)-centered[i]*rstd**3*dot(g,centered)/n);
    const dg=dy.map((v,i)=>v*centered[i]*rstd);
    check(v=>{
      const a=v.slice(0,n),w=v.slice(n),m=kind==='layer'?a.reduce((a,b)=>a+b)/n:0;
      const z=a.map(t=>t-m),r=1/Math.sqrt(dot(z,z)/n+eps);
      return dot(z.map((t,i)=>t*r*w[i]),dy);
    },[...x,...gamma],[...dx,...dg]);
    for(const op of h3.operators.filter(o=>o.rule==='norm'))expect(op.backward).toEqual(op.inputs);
  });
  it('attention backward reconstructs probabilities with Q/K/LSE and uses O for row reduction',()=>{
    const q=[.3,-.8],k=[.6,.2,-.4,.7],v=[.5,-.3,.8,.2],dy=[.7,-.2],scale=1/Math.sqrt(2);
    function forward(a:number[]){
      const logits=[dot(a.slice(0,2),a.slice(2,4))*scale,dot(a.slice(0,2),a.slice(4,6))*scale];
      const lse=Math.log(Math.exp(logits[0])+Math.exp(logits[1]));
      const p=logits.map(s=>Math.exp(s-lse));
      return {lse,p,o:[p[0]*a[6]+p[1]*a[8],p[0]*a[7]+p[1]*a[9]]};
    }
    const all=[...q,...k,...v],{o,lse}=forward(all);
    const p=[dot(q,k.slice(0,2)),dot(q,k.slice(2))].map(s=>Math.exp(s*scale-lse));
    const delta=dot(dy,o),ds=p.map((p,j)=>p*(dot(dy,v.slice(2*j,2*j+2))-delta));
    const dq=[0,1].map(i=>scale*(ds[0]*k[i]+ds[1]*k[2+i]));
    const dk=ds.flatMap(s=>q.map(q=>s*q*scale)),dv=p.flatMap(p=>dy.map(d=>p*d));
    check(a=>dot(forward(a).o,dy),all,[...dq,...dk,...dv]);
    expect(h3.operators.find(o=>o.rule==='attention')!.backward).toEqual(['qr','kr','v','attention']);
  });
});
