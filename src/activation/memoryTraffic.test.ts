import { expect, it } from 'vitest';
import { h3 } from './model';
import { lingbot } from './lingbot';
import { memoryTraffic } from './memoryTraffic';
import { computeFlops, computeTimes, referenceRates } from './compute';
import { none } from './engine';

it('counts SwiGLU two-input/one-output traffic separately from saved storage', () => {
  const p = { ...h3.defaults, sequence: 33630 };
  const traffic = memoryTraffic(h3, p);
  const sw = traffic.rows.find(r => r.id === 'sw')!;
  expect(sw.forward).toBe(6*p.sequence*p.intermediate);
  expect(sw.backward).toBe(10*p.sequence*p.intermediate);
  expect(traffic.forwardMemory).toBe(11811932160);
  expect(traffic.forwardMemory / 1344 / 1e6).toBeCloseTo(8.78864, 5);
  const time = computeTimes(computeFlops(h3,p),{...referenceRates,memory:1344},none,traffic)!;
  expect(time.forward).toBe(time.ms.forwardGemm + time.ms.forwardFa + 8.78864);
});
it('excludes communication, top-k backward and retained top-k replay', () => {
  const traffic = memoryTraffic(lingbot, lingbot.defaults);
  expect(traffic.rows.some(r => ['dispatch','combine'].includes(r.id))).toBe(false);
  const topk = traffic.rows.find(r => r.id === 'topk')!;
  expect(topk.forward).toBe(4096*128*4 + 4096*8*8);
  expect(topk.backward).toBe(0);
  expect(topk.replay).toBe(0);
  const cast = traffic.rows.find(r => r.id === 'castRouter')!;
  expect(cast.forward).toBe(4096*2048*6);
  expect(cast.backward).toBe(cast.forward);
});
it('scales traffic linearly and time inversely with bandwidth', () => {
  const p=h3.defaults, traffic=memoryTraffic(h3,p);
  expect(memoryTraffic(h3,{...p,sequence:p.sequence*2}).forwardMemory).toBe(traffic.forwardMemory*2);
  const flops=computeFlops(h3,p);
  const a=computeTimes(flops,referenceRates,none,traffic)!;
  const b=computeTimes(flops,{...referenceRates,memory:referenceRates.memory*2},none,traffic)!;
  expect(b.ms.forwardMemory).toBe(a.ms.forwardMemory/2);
  expect(b.ms.backwardMemory).toBe(a.ms.backwardMemory/2);
  expect(b.ms.forwardGemm).toBe(a.ms.forwardGemm);
});
