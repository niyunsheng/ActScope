import { describe, expect, it } from 'vitest';
import { createGrid } from './layout';
import { models } from './model';

describe('graph grid', () => {
  it('positions sections using the configured row spacing', () => {
    const grid = createGrid({ main: 250 }, { attention: 0, ffn: 16 }, 60, 40);
    expect(grid.point('main', 2, 'ffn')).toEqual({ x: 250, y: 1120 });
    expect(grid.size(520, 20)).toEqual({ width: 520, height: 1300 });
  });

  for (const model of models.flatMap(model => model.variants ?? [model])) {
    it(`${model.id} routes upward inputs through explicit lanes and keeps nodes inside the canvas`, () => {
      const tensors = new Map(model.tensors.map(t => [t.id, t]));
      for (const op of model.operators) {
        for (const id of op.inputs) {
          if (op.y <= tensors.get(id)!.y) expect(op.inputLanes?.[id]).toBeDefined();
          else expect(op.y - tensors.get(id)!.y).toBeGreaterThanOrEqual(50);
        }
        for (const id of op.outputs) expect(tensors.get(id)!.y - op.y).toBe(50);
      }
      for (const node of [...model.operators, ...model.tensors]) {
        expect(node.y).toBeGreaterThanOrEqual(35);
        expect(node.y + 30).toBeLessThan(model.layout!.height);
      }
    });
  }
});
