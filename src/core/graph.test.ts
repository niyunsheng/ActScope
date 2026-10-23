import { describe, expect, it } from 'vitest';
import { logicalBytes, parseGraph, savedStorageBytes } from './graph';
import example from '../data/qwen3-layer.json';

describe('activation accounting', () => {
  it('computes BF16 bytes, scalars and empty tensors', () => {
    expect(logicalBytes({ shape: [7456, 4096], dtype: 'bf16' })).toBe(58.25 * 2 ** 20);
    expect(logicalBytes({ shape: [], dtype: 'fp32' })).toBe(4);
    expect(logicalBytes({ shape: [0, 4096], dtype: 'bf16' })).toBe(0);
  });
  it('counts the full shared allocation once across aliases and multiple savers', () => {
    const g = parseGraph(example);
    g.tensors.forEach(t => { t.savedBy = []; });
    g.tensors.find(t => t.id === 'v')!.savedBy = ['qkv_bwd', 'proj_bwd'];
    g.tensors.find(t => t.id === 'qkv_out')!.savedBy = ['qkv_bwd'];
    expect(savedStorageBytes(parseGraph(g))).toBe(2048 * 9216 * 2);
    g.tensors.find(t => t.id === 'qkv_out')!.savedBy = [];
    expect(savedStorageBytes(g)).toBe(2048 * 9216 * 2);
    g.tensors.find(t => t.id === 'v')!.savedBy = [];
    expect(savedStorageBytes(g)).toBe(0);
  });
  it('rejects dangling references and wrong-phase savers', () => {
    const g = parseGraph(example);
    g.tensors[0].savedBy = ['missing'];
    expect(() => parseGraph(g)).toThrow();
    g.tensors[0].savedBy = ['norm1'];
    expect(() => parseGraph(g)).toThrow();
  });
  it('rejects duplicate IDs, impossible shapes and unsupported versions', () => {
    const g = parseGraph(example);
    expect(() => parseGraph({ ...g, schemaVersion: 3 })).toThrow();
    g.tensors[0].shape = [-1];
    expect(() => parseGraph(g)).toThrow();
    g.tensors[0].shape = [Number.MAX_SAFE_INTEGER, 10];
    expect(() => parseGraph(g)).toThrow();
    expect(() => parseGraph({ ...example, tensors: [...example.tensors, example.tensors[0]] })).toThrow();
  });
  it('round trips domain data and layout losslessly', () => {
    const g = parseGraph(example);
    expect(parseGraph(JSON.parse(JSON.stringify(g)))).toEqual(g);
  });
  it('accounts for the observed expanded MoE index view without materializing it', () => {
    const g = parseGraph(example);
    g.schemaVersion = 2;
    g.tensors[0] = { ...g.tensors[0], shape: [256, 256], dtype: 'int64', stride: [1, 0], storageOffset: 0 };
    g.storages.find(s => s.id === g.tensors[0].storageId)!.bytes = 2048;
    expect(logicalBytes(parseGraph(g).tensors[0])).toBe(524288);
    g.tensors[0].stride = [256, 1];
    expect(() => parseGraph(g)).toThrow();
    g.tensors[0].stride = [1, 0];
    g.tensors[0].storageOffset = 1;
    expect(() => parseGraph(g)).toThrow();
    g.tensors[0].storageOffset = 0;
    g.schemaVersion = 1;
    expect(() => parseGraph(g)).toThrow();
  });
});
