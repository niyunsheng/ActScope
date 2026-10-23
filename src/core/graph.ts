import { z } from 'zod';

export const dtypeBytes = { bf16: 2, fp16: 2, fp32: 4, int32: 4, int64: 8, bool: 1 } as const;
const id = z.string().min(1);
const integer = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const position = z.object({ x: z.number().finite(), y: z.number().finite() });
const evidence = z.enum(['hypothesis', 'source_verified', 'runtime_observed']);
const operator = z.object({ id, name: id, module: id, phase: z.enum(['forward', 'backward']), position });
const tensor = z.object({
  id, name: id, module: id, shape: z.array(integer), dtype: z.enum(['bf16', 'fp16', 'fp32', 'int32', 'int64', 'bool']),
  storageId: id, producer: id.nullable(), consumers: z.array(id), savedBy: z.array(id),
  evidence, note: z.string(), position,
  stride: z.array(integer).optional(), storageOffset: integer.optional(),
});
export const graphSchema = z.object({
  schemaVersion: z.union([z.literal(1), z.literal(2)]), name: id,
  baseline: z.object({ megatronCommit: id, modelRevision: id, model: id, batch: integer.positive(), sequence: integer.positive(), tp: integer.positive(), cp: integer.positive(), ep: integer.positive(), note: z.string() }),
  operators: z.array(operator), tensors: z.array(tensor),
  storages: z.array(z.object({ id, bytes: integer, evidence })),
}).superRefine((g, ctx) => {
  const issue = (message: string) => ctx.addIssue({ code: 'custom', message });
  const ids = [...g.operators, ...g.tensors].map(x => x.id);
  if (new Set(ids).size !== ids.length) issue('算子和 Tensor ID 必须全局唯一');
  if (new Set(g.storages.map(s => s.id)).size !== g.storages.length) issue('Storage ID 重复');
  const ops = new Map(g.operators.map(o => [o.id, o]));
  const storages = new Map(g.storages.map(s => [s.id, s]));
  for (const t of g.tensors) {
    const bytes = logicalBytes(t);
    if (!Number.isSafeInteger(bytes)) issue(`${t.id}: 大小超过安全整数范围`);
    if (g.schemaVersion === 1 && (t.stride !== undefined || t.storageOffset !== undefined)) issue(`${t.id}: stride/offset 需要 schemaVersion 2`);
    if (t.stride && t.stride.length !== t.shape.length) issue(`${t.id}: stride 和 shape 的维数不同`);
    const span = storageSpanBytes(t);
    if (!Number.isSafeInteger(span)) issue(`${t.id}: storage 地址跨度超出安全整数范围`);
    if (!storages.has(t.storageId)) issue(`${t.id}: storage 不存在`);
    else if (span > storages.get(t.storageId)!.bytes) issue(`${t.id}: Tensor 的实际地址跨度超出 storage`);
    for (const ref of [t.producer, ...t.consumers].filter((x): x is string => x !== null)) {
      if (ops.get(ref)?.phase !== 'forward') issue(`${t.id}: ${ref} 不是有效的前向算子`);
    }
    for (const ref of t.savedBy) if (ops.get(ref)?.phase !== 'backward') issue(`${t.id}: ${ref} 不是有效的反向保存者`);
  }
  if (!Number.isSafeInteger(g.storages.reduce((n, s) => n + s.bytes, 0))) issue('Storage 总量超出安全整数范围');
});

export type Graph = z.infer<typeof graphSchema>;
export type Tensor = Graph['tensors'][number];
export function logicalBytes(t: Pick<Tensor, 'shape' | 'dtype'>) {
  return t.shape.reduce((n, d) => n * d, 1) * dtypeBytes[t.dtype];
}
export function storageSpanBytes(t: Pick<Tensor, 'shape' | 'dtype' | 'stride' | 'storageOffset'>) {
  if (t.shape.some(d => d === 0)) return 0;
  if (!t.stride) return (t.storageOffset ?? 0) * dtypeBytes[t.dtype] + logicalBytes(t);
  const lastElement = (t.storageOffset ?? 0) + t.shape.reduce((n, d, i) => n + (d - 1) * t.stride![i], 0);
  return (lastElement + 1) * dtypeBytes[t.dtype];
}
export function savedStorageBytes(g: Graph) {
  const retained = new Set(g.tensors.filter(t => t.savedBy.length > 0).map(t => t.storageId));
  return g.storages.reduce((n, s) => n + (retained.has(s.id) ? s.bytes : 0), 0);
}
export const formatBytes = (bytes: number) => bytes < 1024 ? `${bytes} B` : bytes < 2 ** 20 ? `${(bytes / 1024).toFixed(2)} KiB` : `${(bytes / 2 ** 20).toFixed(2)} MiB`;
export function parseGraph(input: unknown): Graph { return graphSchema.parse(input); }
