import { describe, expect, it } from 'vitest';
import { h3, models } from './model';
import { lingbot } from './lingbot';
import { computeFlops, computeTimes, referenceRates } from './compute';
import { none } from './engine';

describe('kernel cost estimates', () => {
  it('matches the H3 33,630-token workload and normalized elapsed times', () => {
    const T = 33630, H = 5376, A = 7168, F = 14336;
    const flops = computeFlops(h3, { ...h3.defaults, sequence: T });
    expect(flops.forwardGemm).toBe(2 * T * H * (4 * A + 3 * F));
    expect(flops.forwardFa).toBe(4 * T * T * A);
    expect(flops.backwardGemm).toBe(2 * flops.forwardGemm);
    expect(flops.backwardFa).toBe(2 * flops.forwardFa);
    const base = computeTimes(flops, referenceRates, none)!;
    expect(base.total / 1000).toBeCloseTo(.902, 3);
    expect(base.relative).toBe(100);
    expect(computeTimes(flops, referenceRates, { ...none, memory: true })).toEqual(base);
    const gemm = computeTimes(flops, referenceRates, { ...none, memory: true, gemm: true })!;
    expect(gemm.forward).toBe(base.forward);
    expect(gemm.backward).toBe(base.backward + base.ms.forwardGemm);
    expect(gemm.relative).toBeCloseTo(113.1, 1);
    const full = computeTimes(flops, referenceRates, { memory: true, gemm: true, fa: true })!;
    expect(full.recompute).toBe(base.forward);
    expect(full.relative).toBeCloseTo(129.8, 1);
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
    const a = computeTimes(flops, referenceRates, none)!;
    const b = computeTimes(flops, { ...referenceRates, forwardGemm: 440 }, none)!;
    expect(b.ms.forwardGemm).toBe(a.ms.forwardGemm / 2);
    expect(b.backward).toBe(a.backward);
    for (const rate of [0, -1, NaN, Infinity]) expect(computeTimes(flops, { ...referenceRates, forwardFa: rate }, none)).toBeNull();
  });
});
