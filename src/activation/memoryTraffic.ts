import type { Model, Parameters } from './model';

// Logical main-tensor traffic, not measured device-memory transactions. Each distinct
// input is read once and each output written once per materialized kernel.
// Gradients use their tensor's dtype. Affine/conditioning weights, statistics,
// partial reductions, cache reuse, cross-op fusion and launch latency excluded.
export function memoryTraffic(model: Model, p: Parameters) {
  const tensors = new Map(model.tensors.map(t => [t.id, t]));
  const bytes = (id: string) => {
    const t = tensors.get(id)!;
    return t.metadata ? 0 : t.shape(p).reduce((a, b) => a*b, t.dtype === 'INT64' ? 8 : t.dtype === 'FP32' ? 4 : t.dtype === 'BF16' ? 2 : p.bytes);
  };
  const sum = (ids: string[]) => [...new Set(ids)].reduce((n, id) => n + bytes(id), 0);
  const differentiable = (id: string) => !tensors.get(id)!.metadata && tensors.get(id)!.dtype !== 'INT64';
  const rows = model.operators.filter(op => op.category === 'memory' && op.rule !== 'dispatch' && op.rule !== 'combine').map(op => {
    const input = sum(op.inputs), output = sum(op.outputs);
    let forward = input + output;
    let backward = op.forwardOnly ? 0 : sum(op.backward) + sum(op.outputs.filter(differentiable)) + sum(op.inputs.filter(differentiable));
    if (op.rule === 'route' && !op.forwardOnly) {
      // Gather reads only selected FP32 scores. Backward writes a dense scores
      // gradient, including zeros for unselected experts; indices have no grad.
      const indices = op.inputs.filter(id => !differentiable(id));
      forward = output * 2 + sum(indices);
      backward = output * 2 + sum(indices) + sum(op.inputs.filter(differentiable));
    }
    // Retained top-k indices do not require rerunning discrete selection.
    const replay = op.outputs.every(id => tensors.get(id)!.retain) ? 0 : forward;
    return { id: op.id, forward, backward, replay };
  });
  return {
    rows,
    forwardMemory: rows.reduce((n, r) => n + r.forward, 0),
    backwardMemory: rows.reduce((n, r) => n + r.backward, 0),
    replayMemory: rows.reduce((n, r) => n + r.replay, 0),
  };
}
