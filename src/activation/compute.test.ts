import { describe, expect, it } from 'vitest';
import { h3, models } from './model';
import { lingbot } from './lingbot';
import { computeFlops, computeTimes, referenceRates } from './compute';
import { memoryTraffic } from './memoryTraffic';
import { none } from './engine';

describe('kernel cost estimates', () => {
  it('matches the H3 33,630-token workload and normalized elapsed times', () => {
    const T = 33630, H = 5376, A = 7168, F = 14336;
    const flops = computeFlops(h3, { ...h3.defaults, sequence: T });
    expect(flops.forwardGemm).toBe(2 * T * H * (4 * A + 3 * F));
    expect(flops.forwardFa).toBe(4 * T * T * A);
    expect(flops.backwardGemm).toBe(2 * flops.forwardGemm);
    expect(flops.backwardFa).toBe(2 * flops.forwardFa);
    const traffic = memoryTraffic(h3, { ...h3.defaults, sequence: T });
    const base = computeTimes(flops, referenceRates, none, traffic)!;
    expect(base.ms.backwardGemm).toBe(flops.backwardGemm / 216 / 1e9);
    expect(base.ms.forwardMemory).toBe(traffic.forwardMemory / referenceRates.memory / 1e6);
    expect(base.total).toBe(base.forward + base.backward);
    expect(base.relative).toBe(100);
    const memory = computeTimes(flops, referenceRates, { ...none, memory: true }, traffic)!;
    expect(memory.recompute).toBe(traffic.replayMemory / referenceRates.memory / 1e6);
    expect(memory.relative).toBeGreaterThan(100);
    const gemm = computeTimes(flops, referenceRates, { ...none, memory: true, gemm: true }, traffic)!;
    expect(gemm.forward).toBe(base.forward);
    expect(gemm.backward).toBe(base.backward + base.ms.forwardGemm + memory.recompute);
    expect(gemm.relative).toBeGreaterThan(memory.relative);
    const full = computeTimes(flops, referenceRates, { memory: true, gemm: true, fa: true }, traffic)!;
    expect(full.recompute).toBe(base.forward);
    expect(full.relative).toBeGreaterThan(gemm.relative);
  });
  it('counts only selected expert assignments, plus shared experts and router GEMM', () => {
    const { sequence: T, hidden: H, intermediate: I } = lingbot.defaults;
    expect(computeFlops(lingbot, lingbot.defaults).forwardGemm).toBe(8*T*H*H + 6*T*8*H*I + 6*T*H*I + 2*T*H*128);
  });
  it('scales GEMM linearly and full attention quadratically with sequence length', () => {
    for (const model of models) {
      const a = computeFlops(model, model.defaults);
      const b = computeFlops(model, { ...model.defaults, sequence: model.defaults.sequence * 2 });
      expect(b.forwardGemm).toBe(2*a.forwardGemm);
      expect(b.forwardFa).toBe(4*a.forwardFa);
    }
  });
  it('responds to throughput edits and rejects invalid throughput', () => {
    const flops = computeFlops(h3, h3.defaults);
    const traffic = memoryTraffic(h3, h3.defaults);
    const a = computeTimes(flops, referenceRates, none, traffic)!;
    const b = computeTimes(flops, { ...referenceRates, gemm: 440 }, none, traffic)!;
    expect(b.ms.forwardGemm).toBe(a.ms.forwardGemm / 2);
    expect(b.ms.backwardGemm).toBe(a.ms.backwardGemm);
    expect(b.ms.forwardMemory).toBe(a.ms.forwardMemory);
    for (const rate of [0, -1, NaN, Infinity]) expect(computeTimes(flops, { ...referenceRates, forwardFa: rate }, none, traffic)).toBeNull();
  });
});
