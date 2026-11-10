import { useEffect, useRef, useState } from 'react';
import type { Model } from './model';
import type { Analysis } from './engine';
import { formatBytes } from './engine';
import { drawConnections } from './drawConnections';
import { useGraphViewport } from './useGraphViewport';
import NodeDetails from './NodeDetails';
export default function Graph({ model, result }: { model: Model; result: Analysis }) {
  const [hoveredTensor, setHoveredTensor] = useState<string | null>(null);
  const [hoveredOp, setHoveredOp] = useState<string | null>(null);
  const clearHover = () => { setHoveredOp(null); setHoveredTensor(null); };
  const { viewport, view, dragging, fit, reset, zoomCenter, panHandlers } = useGraphViewport(clearHover, model.layout?.width, model.layout?.height);
  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') { clearHover();  } };
    window.addEventListener('resize', clearHover); window.addEventListener('keydown', key);
    return () => { window.removeEventListener('resize', clearHover); window.removeEventListener('keydown', key); };
  }, []);
  const dependencies = new Set(model.operators.find(op => op.id === hoveredOp)?.backward ?? []);
  const consumers = new Set(model.operators.filter(op => hoveredTensor && op.backward.includes(hoveredTensor)).map(op => op.id));
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    drawConnections(canvas.current!, model, hoveredOp, hoveredTensor, view.scale);
  }, [model, hoveredOp, hoveredTensor, view.scale]);
  return <>
    <div className="ag-zoom-toolbar"><NodeDetails operator={model.operators.find(op => op.id === hoveredOp)} tensor={result.rows.find(t => t.id === hoveredTensor)} rows={result.rows} operators={model.operators} /><div>
      <button aria-label="Zoom out" onClick={() => zoomCenter(1 / 1.2)}>−</button>
      <output aria-label="Graph zoom">{Math.round(view.scale * 100)}%</output>
      <button aria-label="Zoom in" onClick={() => zoomCenter(1.2)}>＋</button>
      <button onClick={fit}>Fit to view</button>
      <button onClick={reset}>Reset 100%</button>
    </div></div>
    <div ref={viewport} className={`ag-scroll ag-zoom-viewport ${dragging ? 'dragging' : ''}`} aria-label="Zoomable graph"
      {...panHandlers}

    ><div className="ag-diagram" aria-label={`${model.name} single-layer graph`} style={{width: model.layout?.width, height: model.layout?.height, transform:`translate(${view.x}px, ${view.y}px) scale(${view.scale})`}}>
    {model.regions?.map(region => <div key={region.label} className="ag-graph-region" style={{left: region.x, top: region.y, width: region.width, height: region.height}}><span>{region.label}</span></div>)}
    <canvas ref={canvas} aria-hidden="true" style={{width: model.layout?.width, height: model.layout?.height}} />
    {model.operators.map(op => <button type="button" key={op.id} data-operator={op.id} data-backward-consumer={consumers.has(op.id)} aria-label={`Operator ${op.label} (${op.id})${op.forwardOnly ? ' · Forward only' : ''}`} onMouseEnter={() => { if (!dragging) setHoveredOp(op.id); }} onMouseLeave={() => setHoveredOp(null)} onFocus={() => setHoveredOp(op.id)} onBlur={() => setHoveredOp(null)} className={`ag-operator ${op.category === 'memory' ? 'memory-bound' : 'compute-bound'} ${hoveredOp === op.id || consumers.has(op.id) ? 'inspected' : ''}`} style={{ left: op.x, top: op.y }}>{op.label}{op.forwardOnly && <span className="ag-forward-only">Fwd only</span>}</button>)}
    {result.rows.filter(t => !t.metadata).map(t => <button key={t.id} data-tensor={t.id} onMouseEnter={() => { if (!dragging) setHoveredTensor(t.id); }} onMouseLeave={() => setHoveredTensor(null)} onFocus={() => setHoveredTensor(t.id)} onBlur={() => setHoveredTensor(null)} data-backward-dependency={dependencies.has(t.id)} className={`ag-tensor ${t.keep ? 'saved' : 'released'} ${dependencies.has(t.id) || hoveredTensor === t.id ? 'dependency' : ''}`} style={{ left: t.x, top: t.y }} aria-label={`${t.label}: ${t.keep ? 'Saved' : 'Not saved'}, ${formatBytes(t.bytes)}`}><span className="ag-dot" /><span className="ag-tensor-label">{t.label}</span></button>)}
  </div>

  </div>
  </>;
}
