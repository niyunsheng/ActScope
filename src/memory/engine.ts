import { z } from 'zod';

export const documentSchema = z.object({
  version: z.literal(1), name: z.string().min(1).max(100),
  variables: z.array(z.object({ name: z.string().regex(/^[A-Za-z_][A-Za-z0-9_]*$/), value: z.number().positive().finite() })).max(100),
  repeats: z.string().min(1).max(200),
  tensors: z.array(z.object({ id: z.string(), name: z.string().min(1), shape: z.string().min(1).max(500), kind: z.enum(['replicated', 'tp', 'expert']), axis: z.number().int().min(0), shardUnit: z.string().max(200).optional(), trainable: z.boolean() })).max(300),
  activations: z.array(z.object({ id: z.string(), name: z.string().min(1), savedBy: z.string(), perToken: z.string().min(1).max(500).optional(), shape: z.string().min(1).max(500), bytes: z.number().int().min(1).max(8), policy: z.enum(['keep', 'recompute', 'offload']) })).max(300),
});
export type Model = z.infer<typeof documentSchema>;
export type Kind = Model['tensors'][number]['kind'];
export type Settings = { gpus: number; tp: number; ep: number; etp: number; pp: number; paramBytes: number; gradBytes: number; optimizerBytes: number; shard: 'none' | 'optim' | 'grads' | 'full' };
export const defaults: Settings = { gpus: 8, tp: 1, ep: 8, etp: 1, pp: 1, paramBytes: 2, gradBytes: 4, optimizerBytes: 12, shard: 'optim' };

// Small arithmetic parser: no eval, property access, calls or executable expressions.
export function expression(source: string, vars: Record<string, number>): number {
  if (source.length > 500) throw new Error('表达式过长');
  const tokens = source.match(/[A-Za-z_][A-Za-z0-9_]*|(?:\d+(?:\.\d*)?|\.\d+)|[()+*/-]/g) || [];
  if (tokens.join('') !== source.replace(/\s/g, '')) throw new Error('表达式只支持变量、数字、括号和 + − × /');
  let i = 0;
  function atom(): number {
    const t = tokens[i++];
    if (t === '(') { const n = sum(); if (tokens[i++] !== ')') throw new Error('括号不匹配'); return n; }
    if (t === '-') return -atom();
    if (t === '+') return atom();
    if (!t) throw new Error('表达式不完整');
    if (/^\d|^\./.test(t)) return Number(t);
    if (!Object.hasOwn(vars, t)) throw new Error(`未定义变量：${t}`);
    return vars[t];
  }
  function product(): number { let n = atom(); while (tokens[i] === '*' || tokens[i] === '/') { const op = tokens[i++]; const r = atom(); n = op === '*' ? n * r : n / r; } return n; }
  function sum(): number { let n = product(); while (tokens[i] === '+' || tokens[i] === '-') { const op = tokens[i++]; const r = product(); n = op === '+' ? n + r : n - r; } return n; }
  const n = sum();
  if (i !== tokens.length || !Number.isFinite(n)) throw new Error('表达式无效或结果溢出');
  return n;
}
function integer(n: number, label: string) { if (!Number.isSafeInteger(n) || n <= 0) throw new Error(`${label}必须是正安全整数`); return n; }
export function shapeOf(shape: string, vars: Record<string, number>) {
  const s = shape.trim().replace(/^\[/, '').replace(/\]$/, '');
  return s.split(',').map(dim => integer(expression(dim, vars), '维度'));
}
export function variables(model: Model) {
  const vars: Record<string, number> = Object.create(null);
  model.variables.forEach(v => { if (Object.hasOwn(vars, v.name)) throw new Error(`变量重复：${v.name}`); vars[v.name] = v.value; });
  return vars;
}
export function calculate(model: Model, s: Settings) {
  documentSchema.parse(model);
  const ids = [...model.tensors, ...model.activations].map(t => t.id);
  if (new Set(ids).size !== ids.length) throw new Error('Tensor / 激活 ID 不能重复');
  const vars = variables(model);
  for (const k of ['gpus', 'tp', 'ep', 'etp', 'pp'] as const) integer(s[k], k);
  const dp = integer(s.gpus / (s.pp * s.tp), 'Dense DP（GPU / PP / TP）');
  const edp = integer(s.gpus / (s.pp * s.ep * s.etp), 'Expert DP（GPU / PP / EP / ETP）');
  const layers = integer(expression(model.repeats, vars), '模块重复次数');
  // Balanced stage ownership is explicit; no fractional layer memory.
  const localLayers = Math.ceil(layers / s.pp);
  if (s.pp > layers) throw new Error('PP 不能大于模块重复次数');
  const rows = model.tensors.map(t => {
    const dims = shapeOf(t.shape, vars);
    const count = integer(dims.reduce((a,b) => a*b, 1), '参数量');
    const local = [...dims];
    const divisor = t.kind === 'expert' ? s.etp : t.kind === 'tp' ? s.tp : 1;
    if (t.kind !== 'replicated') {
      if (t.axis >= dims.length) throw new Error(`${t.name}：分片维度超出 shape`);
      if (t.kind === 'expert' && t.axis === 0) throw new Error(`${t.name}：第 0 维用于 EP，ETP 请选其他维度`);
      local[t.axis] = integer(local[t.axis] / divisor, `${t.name} 的 TP/ETP 本地维度`);
      const unit = integer(expression(t.shardUnit || '1', vars), '分片粒度');
      if (local[t.axis] % unit) throw new Error(`${t.name}：本地维度须为 ${unit} 的倍数；当前不支持 KV head 复制`);
    }
    if (t.kind === 'expert') local[0] = integer(local[0] / s.ep, `${t.name} 的本地 expert 数`);
    const localCount = local.reduce((a,b) => a*b, 1);
    const group = t.kind === 'expert' ? edp : dp;
    const sp = s.shard === 'full' ? group : 1;
    const sg = s.shard === 'full' || s.shard === 'grads' ? group : 1;
    const so = s.shard !== 'none' ? group : 1;
    const param = localCount*s.paramBytes/sp*localLayers;
    const grad = t.trainable ? localCount*s.gradBytes/sg*localLayers : 0;
    const optim = t.trainable ? localCount*s.optimizerBytes/so*localLayers : 0;
    return { ...t, dims, local, count, localCount, param, grad, optim, total: param+grad+optim, group, sp, sg, so };
  });
  const acts = model.activations.map(a => { const dims = a.perToken ? [] : shapeOf(a.shape, vars); const tokens = integer(vars.tokens_per_rank, 'tokens_per_rank'); const size = a.perToken ? integer(expression(a.perToken, vars), '每 token 字节数') * tokens : integer(dims.reduce((x,y) => x*y, 1), '激活元素数')*a.bytes; return { ...a, dims, size, perTokenBytes: size / tokens }; });
  const totals = rows.reduce((r,t) => ({param:r.param+t.param, grad:r.grad+t.grad, optim:r.optim+t.optim}), {param:0,grad:0,optim:0});
  return { rows, acts, totals, total:totals.param+totals.grad+totals.optim, layers, localLayers, dp, edp, globalParams:rows.reduce((n,t) => n+t.count, 0)*layers,
    activation: { all:acts.reduce((n,a)=>n+a.size,0), keep:acts.filter(a=>a.policy==='keep').reduce((n,a)=>n+a.size,0), recompute:acts.filter(a=>a.policy==='recompute').reduce((n,a)=>n+a.size,0), offload:acts.filter(a=>a.policy==='offload').reduce((n,a)=>n+a.size,0) } };
}
export function bytes(n: number) { if (!n) return '0 B'; const i = Math.min(4, Math.floor(Math.log(n)/Math.log(1024))); return `${(n/1024**i).toLocaleString('en-US',{maximumFractionDigits:2,minimumFractionDigits: i ? 2:0})} ${['B','KiB','MiB','GiB','TiB'][i]}`; }
export function number(n:number) { return n.toLocaleString('en-US', {maximumFractionDigits:0}); }
