import measurement from '../../benchmarks/data/h3-33630.json';
import type { KernelRates } from './compute';

type MeasurementRow = {
  group: string; direction: string; count: number; median_ms: number;
  flops?: number; logical_bytes?: number;
};

// Sum per-shape medians with the model's multiplicities. These are isolated
// probes, not an end-to-end layer measurement or measured DRAM transactions.
const rows: MeasurementRow[] = measurement.rows;
export function measuredWork(group: string, forward: boolean) {
  const selected = rows.filter(r => r.group === group && (r.direction === 'forward') === forward);
  const ms = selected.reduce((sum, r) => sum + r.count * r.median_ms, 0);
  const work = selected.reduce((sum, r) => sum + r.count * (r.flops ?? r.logical_bytes ?? 0), 0);
  return { ms, work, rate: work / ms / (group === 'memory' ? 1e6 : 1e9) };
}
export const measuredGroups = [
  { id: 'gemm', label: 'GEMM', unit: 'TFLOP/s' },
  { id: 'fa', label: 'FA4', unit: 'TFLOP/s' },
  { id: 'memory', label: 'Memory probes', unit: 'logical GB/s' },
].map(group => ({ ...group, forward: measuredWork(group.id, true), backward: measuredWork(group.id, false) }));

const pooled = (id: string, divisor: number) => {
  const { forward, backward } = measuredGroups.find(group => group.id === id)!;
  return (forward.work + backward.work) / (forward.ms + backward.ms) / divisor;
};
const unroundedRates: KernelRates = {
  gemm: pooled('gemm', 1e9),
  forwardFa: measuredWork('fa', true).rate,
  backwardFa: measuredWork('fa', false).rate,
  memory: pooled('memory', 1e6),
};
export const measuredRates = Object.fromEntries(
  Object.entries(unroundedRates).map(([key, value]) => [key, Number(value.toFixed(1))]),
) as KernelRates;
export const measurementWorkload = measurement.workload;
export const measurementMethod = measurement.method;
