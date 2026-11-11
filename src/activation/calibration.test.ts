import { describe, expect, it } from 'vitest';
import measurement from '../../benchmarks/data/h3-33630.json';
import { measuredGroups, measuredRates, measuredWork } from './calibration';
import { computeFlops, computeTimes } from './compute';
import { memoryTraffic } from './memoryTraffic';
import { h3 } from './model';
import { none } from './engine';

describe('H3 measurement compatibility', () => {
  const params = { ...h3.defaults, sequence: measurement.workload.tokens };
  const flops = computeFlops(h3, params);
  const traffic = memoryTraffic(h3, params);

  it('uses the customer image FA4 build with recorded source identity and verified GPU selection', () => {
    expect(measurement.environment.fa4.version).toBe('0.0.0+h3fl2va.bwdv2.gb011817.pkg1');
    expect(measurement.environment.container.image_ref).toBe('oppenheimli/aigc:share.h3.ofld.v3');
    expect(measurement.environment.container.image_digest).toBe('oppenheimli/aigc@sha256:8594ffb9da1e05a1ef598270b45b28dba78f426468cbdb8be57ee5efc10df793');
    expect(measurement.environment.fa4.entry_files_sha256['interface.py']).toBe('8a3ca80665b6099c0a5c1a55f67c227b8c762df403ace3ba931677e50b7295d8');
    expect(measurement.environment.fa4.python_source_tree_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(measurement.environment.gpu_selection_verified).toBe(true);
  });

  it('covers the exact frontend GEMM, attention and logical memory work', () => {
    expect(measuredWork('gemm', true).work).toBe(flops.forwardGemm);
    expect(measuredWork('gemm', false).work).toBe(flops.backwardGemm);
    expect(measuredWork('fa', true).work).toBe(flops.forwardFa);
    expect(measuredWork('fa', false).work).toBe(flops.backwardFa);
    expect(measuredWork('memory', true).work).toBe(traffic.forwardMemory);
    expect(measuredWork('memory', false).work).toBe(traffic.backwardMemory);
  });

  it('contains a completed run, correctness checks and all finite timing samples', () => {
    expect(measurement.finished_at_utc).toBeTruthy();
    expect(measurement.fa_correctness).toContain('passed');
    expect(measurement.memory_correctness).toContain('passed');
    expect(measurement.rows).toHaveLength(27);
    for (const row of measurement.rows) {
      expect(row.samples_ms).toHaveLength(measurement.method.samples);
      for (const ms of row.samples_ms) expect(Number.isFinite(ms) && ms > 0).toBe(true);
    }
  });

  it('preserves summed probe time at the measured workload while keeping replay estimated', () => {
    const baseline = computeTimes(flops, measuredRates, none, traffic)!;
    const probeSum = measuredGroups.reduce((sum, group) => sum + group.forward.ms + group.backward.ms, 0);
    // Editable inputs are rounded to 0.1; aggregation should remain within 1 ms.
    expect(Math.abs(baseline.total - probeSum)).toBeLessThan(1);
    const full = computeTimes(flops, measuredRates, { memory: true, gemm: true, fa: true }, traffic)!;
    expect(full.recompute).toBeCloseTo(baseline.forward, 9);
    expect(full.relative).toBeGreaterThan(100);
  });
});
