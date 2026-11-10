import { memoryTraffic } from './memoryTraffic';
import { positive } from './hardware';
import type { Model, Parameters, Policy } from './model';

export const kernels = [
  { id: 'gemm', label: 'GEMM', unit: 'TFLOP/s', reference: 220 },
  { id: 'forwardFa', label: 'FA Forward', unit: 'TFLOP/s', reference: 215 },
  { id: 'backwardFa', label: 'FA Backward', unit: 'TFLOP/s', reference: 165 },
  { id: 'memory', label: 'Memory-bound', unit: 'GB/s', reference: 672 },
] as const;
export type KernelRates = Record<typeof kernels[number]['id'], number>;
export type KernelValues = Record<'forwardGemm' | 'forwardFa' | 'backwardGemm' | 'backwardFa', number>;
export const referenceRates = { ...Object.fromEntries(kernels.map(k => [k.id, k.reference])) } as KernelRates;
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

export function computeTimes(flops: KernelValues, rates: KernelRates, policy: Policy, traffic: Pick<ReturnType<typeof memoryTraffic>, 'forwardMemory' | 'backwardMemory' | 'replayMemory'>) {
  if (Object.values(rates).some(rate => !positive(rate))) return null;
  const ms = {
    forwardGemm: flops.forwardGemm / rates.gemm / 1e9,
    backwardGemm: flops.backwardGemm / rates.gemm / 1e9,
    forwardFa: flops.forwardFa / rates.forwardFa / 1e9,
    backwardFa: flops.backwardFa / rates.backwardFa / 1e9,
    forwardMemory: traffic.forwardMemory / rates.memory / 1e6,
    backwardMemory: traffic.backwardMemory / rates.memory / 1e6,
  };
  const forward = ms.forwardGemm + ms.forwardFa + ms.forwardMemory;
  const baseBackward = ms.backwardGemm + ms.backwardFa + ms.backwardMemory;
  // One replay of each selected compute op and non-retained local memory op.
  // Communication is excluded; throughput stays constant across strategies.
  const recompute = (policy.gemm ? ms.forwardGemm : 0) + (policy.fa ? ms.forwardFa : 0) + (policy.memory ? traffic.replayMemory / rates.memory / 1e6 : 0);
  const backward = baseBackward + recompute;
  const total = forward + backward;
  return { ms, forward, backward, recompute, total, relative: total / (forward + baseBackward) * 100 };
}
