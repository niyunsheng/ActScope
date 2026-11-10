import type { Model } from './model';

export function drawConnections(canvas: HTMLCanvasElement, model: Model, selectedOp: string | null, selectedTensor: string | null, scale: number) {
    const el = canvas; const dpr = (window.devicePixelRatio || 1) * Math.max(1, scale);
    el.width = (model.layout?.width ?? 520) * dpr; el.height = (model.layout?.height ?? 1490) * dpr;
    const ctx = el.getContext('2d')!; ctx.scale(dpr, dpr);
    const ts = new Map(model.tensors.map(t => [t.id, t]));
    const branchY = new Map(model.tensors.map(t => {
      const consumers = model.operators.filter(op => op.inputs.includes(t.id));
      const downward = consumers.filter(op => op.y > t.y);
      const firstInputY = downward.length ? Math.min(...downward.map(op => op.y - 26)) : t.y + 30;
      return [t.id, consumers.length > 1 ? Math.min(t.y + 30, firstInputY) : firstInputY];
    }));
    function arrow(points: [number, number][]) {
      ctx.beginPath(); ctx.setLineDash([]);
      ctx.strokeStyle = '#596678'; ctx.lineWidth = 1.3;
      ctx.moveTo(...points[0]); for (const point of points.slice(1)) ctx.lineTo(...point); ctx.stroke();
      const a = points.at(-2)!, b = points.at(-1)!; const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
      ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(...b);
      ctx.lineTo(b[0] - 6 * Math.cos(angle - .45), b[1] - 6 * Math.sin(angle - .45));
      ctx.lineTo(b[0] - 6 * Math.cos(angle + .45), b[1] - 6 * Math.sin(angle + .45)); ctx.closePath(); ctx.fillStyle = ctx.strokeStyle; ctx.fill();
    }
    for (const op of model.operators) {
      for (const id of op.inputs) {
        const t = ts.get(id)!;
        if (t.metadata) continue;
        const lane = op.inputLanes?.[id];
        const branch = op.inputBranches?.[id];
        if (branch !== undefined) {
          const y = branchY.get(id)!;
          const mergeY = op.y - 40;
          arrow([[t.x, t.y + 10], [t.x, y], [branch, y], [branch, mergeY], [op.x, mergeY], [op.x, op.y - 16]]);
        } else if (lane !== undefined) {
          const y = branchY.get(id)!;
          arrow([[t.x, t.y + 10], [t.x, y], [lane, y], [lane, op.y], [op.x + (lane < op.x ? -65 : 65), op.y]]);
        } else {
          const y = branchY.get(id)!;
          arrow([[t.x, t.y + 10], [t.x, y], [op.x, y], [op.x, op.y - 16]]);
        }
      }
      for (const id of op.outputs) {
        const t = ts.get(id)!;
        if (t.metadata) continue;
        arrow([[op.x, op.y + 16], [op.x, op.y + 25], [t.x, op.y + 25], [t.x, t.y - 10]]);
      }
      for (const id of op.backward) {
        const t = ts.get(id)!;
        if (t.metadata) continue;
        // A smooth dependency arc, distinct from the orthogonal forward path.
        const start = { x: op.x - 62, y: op.y };
        const end = { x: t.x - 10, y: t.y };
        const lane = Math.min(start.x, end.x) - 42;
        const highlighted = op.id === selectedOp || id === selectedTensor;
        ctx.beginPath(); ctx.setLineDash([3, 4]);
        ctx.strokeStyle = highlighted ? '#7954c5' : '#9da8b5';
        ctx.lineWidth = highlighted ? 2.5 : 1;
        ctx.moveTo(start.x, start.y);
        ctx.bezierCurveTo(lane, start.y, lane, end.y, end.x, end.y);
        ctx.stroke(); ctx.setLineDash([]);
        // The curve's final tangent points right into the tensor's left edge.
        ctx.beginPath(); ctx.moveTo(end.x, end.y);
        ctx.lineTo(end.x - 6, end.y - 3);
        ctx.lineTo(end.x - 6, end.y + 3);
        ctx.closePath(); ctx.fillStyle = ctx.strokeStyle; ctx.fill();
      }
    }
}
