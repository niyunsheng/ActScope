import type { Model, Parameters, Policy } from './model';

export const kernels = [
  { id: 'forwardGemm', label: 'Forward GEMM', reference: 220 },
  { id: 'forwardFa', label: 'Forward FA', reference: 215 },
  { id: 'backwardGemm', label: 'Backward GEMM', reference: 216 },
  { id: 'backwardFa', label: 'Backward FA', reference: 165 },
] as const;
export type Kernel = typeof kernels[number]['id'];
export type KernelValues = Record<Kernel, number>;
export const referenceRates = Object.fromEntries(kernels.map(k => [k.id, k.reference])) as KernelValues;
const width = (shape: number[]) => shape.slice(1).reduce((a, b) => a * b, 1);

// Logical single-sequence matmul FLOPs. Backward uses effective work (2× forward),
// so FA throughput must also use effective FLOPs, excluding internal QK replay.
export function computeFlops(model: Model, params: Parameters): KernelValues {
  const shapes = new Map(model.tensors.map(t => [t.id, t.shape(params)]));
  let gemm = 0, fa = 0;
  for (const op of model.operators) {
    if (op.category === 'gemm') {
      const input = shapes.get(op.inputs[0])!;
      const outputWidth = op.weightShape?.at(-2) ?? op.outputs.reduce((n, id) => n + width(shapes.get(id)!), 0);
      // Expert inputs already contain T×top-k rows; never multiply by E again.
      gemm += 2 * input[0] * width(input) * outputWidth;
    } else if (op.category === 'fa') {
      const [q, k, v] = op.inputs.map(id => shapes.get(id)!);
      fa += 2 * q[0] * k[0] * q[1] * (q[2] + v[2]);
    }
  }
  return { forwardGemm: gemm, forwardFa: fa, backwardGemm: 2 * gemm, backwardFa: 2 * fa };
}

export function computeTimes(flops: KernelValues, rates: KernelValues, policy: Policy) {
  if (kernels.some(k => !Number.isFinite(rates[k.id]) || rates[k.id] <= 0)) return null;
  const ms = Object.fromEntries(kernels.map(k => [k.id, flops[k.id] / rates[k.id] / 1e9])) as KernelValues;
  const forward = ms.forwardGemm + ms.forwardFa;
  const baseBackward = ms.backwardGemm + ms.backwardFa;
  // Explicit cost scenario: each selected GEMM/FA is replayed once during backward.
  // Memory-bound work and communication time are excluded.
  const recompute = (policy.gemm ? ms.forwardGemm : 0) + (policy.fa ? ms.forwardFa : 0);
  const backward = baseBackward + recompute;
  const total = forward + backward;
  return { ms, forward, backward, recompute, total, relative: total / (forward + baseBackward) * 100 };
}
