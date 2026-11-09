import type { Model, Parameters, Policy } from './model';
export const none: Policy = { gemm: false, fa: false, memory: false };
export function validate(p: Parameters): string | null {
  if (Object.values(p).some(v => !Number.isSafeInteger(v) || v <= 0)) return 'All parameters must be positive integers.';
  if (![2, 4].includes(p.bytes)) return 'Activation dtype must use 2 or 4 bytes.';
  if (p.heads % p.kvHeads) return 'Q heads must be a multiple of KV heads.';
  if (p.headDim % 2) return 'RoPE head dimension must be even.';
  return null;
}
export function analyze(model: Model, p: Parameters, policy: Policy) {
  const error = validate(p); if (error) throw new Error(error);
  const tensors = new Map(model.tensors.map(t => [t.id, t]));
  if (tensors.size !== model.tensors.length) throw new Error('Duplicate tensor ID');
  const known = (id: string) => { if (!tensors.has(id)) throw new Error(`Unknown tensor: ${id}`); };
  for (const id of model.boundary) known(id);
  const defined = new Set(model.boundary);
  for (const op of model.operators) {
    for (const id of op.inputs) { known(id); if (!defined.has(id)) throw new Error(`Missing or cyclic producer: ${id}`); }
    for (const id of op.outputs) { known(id); if (defined.has(id)) throw new Error(`Duplicate producer: ${id}`); defined.add(id); }
    for (const id of op.backward) { known(id); if (![...op.inputs, ...op.outputs].includes(id)) throw new Error(`Invalid backward dependency: ${id}`); }
  }
  if (defined.size !== tensors.size) throw new Error('Tensor has no producer or boundary');
  const producers = new Map(model.operators.flatMap(op => op.outputs.map(id => [id, op] as const)));
  const saved = new Map<string, Set<string>>();
  const recomputed = new Set<string>();
  const walked = new Set<string>();
  function demand(id: string, reason: string) {
    const producer = producers.get(id);
    if (model.boundary.includes(id) || !producer || !policy[producer.category]) {
      if (!saved.has(id)) saved.set(id, new Set());
      saved.get(id)!.add(reason); return;
    }
    recomputed.add(id);
    if (walked.has(id)) return;
    for (const input of producer.inputs) demand(input, `Input for recomputing ${producer.label}`);
    walked.add(id);
  }
  for (const id of model.boundary) {
    saved.set(id, new Set(['Layer boundary: recomputation input']));
  }
  for (const op of model.operators) for (const id of op.backward) demand(id, `${op.label}.backward`);
  const rows = model.tensors.map(t => {
    const shape = t.shape(p);
    if (shape.some(d => !Number.isSafeInteger(d) || d <= 0)) throw new Error(`Invalid shape: ${t.id}`);
    const bytes = shape.reduce((a, b) => a * b, p.bytes);
    if (!Number.isSafeInteger(bytes) || bytes <= 0) throw new Error('Tensor size exceeds the safe numeric range.');
    const keep = saved.has(t.id);
    const retained = keep ? bytes : 0;
    return { ...t, dimensions: shape, bytes, keep, retained, recompute: recomputed.has(t.id), reasons: [...(saved.get(t.id) ?? [])] };
  });
  return { rows, total: rows.reduce((sum, r) => sum + r.retained, 0), savedCount: rows.filter(r => r.keep).length };
}
export type Analysis = ReturnType<typeof analyze>;
export const formatBytes = (n: number) => n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(2)} GiB` : `${(n / 1024 ** 2).toFixed(2)} MiB`;
