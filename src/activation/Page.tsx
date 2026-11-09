import { useMemo, useState } from 'react';
import { models } from './model';
import type { Model } from './model';
import { analyze, formatBytes } from './engine';
import Graph from './Graph';
import './activation.css';
const levels = [
  { label: 'No recomputation', detail: 'Retain activations needed for backward', policy: { memory: false, gemm: false, fa: false } },
  { label: 'Recompute memory-bound', detail: 'Norm, modulate, RoPE, SwiGLU, residual', policy: { memory: true, gemm: false, fa: false } },
  { label: 'Recompute memory-bound + GEMM', detail: 'Memory-bound + GEMM', policy: { memory: true, gemm: true, fa: false } },
  { label: 'Full recompute (memory-bound + GEMM + FA)', detail: 'Memory-bound + GEMM + FA', policy: { memory: true, gemm: true, fa: true } },
];
export default function ActivationPage({ model }: { model: Model }) {
  const [sequence, setSequence] = useState(model.defaults.sequence);
  const params = useMemo(() => ({ ...model.defaults, sequence }), [model, sequence]);
  const [level, setLevel] = useState(0);
  const data = useMemo(() => {
    try { return { results: levels.map(item => analyze(model, params, item.policy)), error: null }; }
    catch (e) { return { results: null, error: (e as Error).message }; }
  }, [model, params]);
  const { results, error } = data;
  const result = results?.[level];
  return <main className="activation-page ag-single-layer">
    <header className="ag-header"><div className="ag-brand">ActScope</div><label className="ag-model-inline">Model<select aria-label="Select model" value={model.id} onChange={e => { location.hash = `/models/${e.target.value}`; }}>{models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select></label></header>
    <div className="ag-main">
      <section className="ag-parameter-block" aria-label="Model and single-layer parameters">
        <div className="ag-fixed-parameters">
          <label className="ag-sequence-field">Sequence length · S<input aria-label="Sequence length · S" type="number" min={1} step={1} value={sequence || ''} onChange={e => setSequence(Number(e.target.value))} /></label>
          <div><span>Hidden size</span><strong>{params.hidden}</strong></div>
          <div><span>FFN intermediate</span><strong>{params.intermediate}</strong></div>
          <div><span>Attention heads</span><strong>{params.heads}</strong></div>
          <div><span>Head dimension</span><strong>{params.headDim}</strong></div>
          <div><span>Q / K / V width</span><strong>{params.heads * params.headDim}</strong></div>
          <div><span>Activation dtype</span><strong>BF16</strong></div>
          <div><span>Microbatch size</span><strong>1</strong></div>
        </div>
      </section>
      {error && <div className="ag-error" role="alert">{error}</div>}
      <fieldset className="ag-strategies"><legend>Saved activations per layer · Select a strategy to update the graph</legend>{levels.map((item,index) => {
        const summary = results?.[index];
        return <label className={`ag-strategy ${level === index ? 'active' : ''}`} key={item.label}>
          <input type="radio" name="recompute-level" aria-label={item.label} value={index} checked={level === index} onChange={() => setLevel(index)} />
          <span className="ag-strategy-heading"><span className="ag-level-number">{index}</span><strong>{item.label}</strong><span className="ag-choice-mark" aria-hidden="true">{level === index ? '●' : '○'}</span></span>
          <span className="ag-strategy-description">{index === 0 ? 'No recomputation' : item.detail}</span>
          <strong className="ag-strategy-bytes" data-testid={index === level ? 'current-memory' : undefined}>{summary ? formatBytes(summary.total) : '—'}</strong>
          <span className="ag-strategy-footer">{summary ? `${summary.savedCount} tensors saved` : 'Check parameters'}<span>{summary && results ? `Saved ${((1-summary.total/results[0].total)*100).toFixed(1)}%` : ''}</span></span>
        </label>;
      })}</fieldset>
      <section className="ag-graph-panel"><div className="ag-panel-heading ag-graph-heading"><h2>Single-layer graph</h2><div className="ag-legend"><span><i className="compute-swatch" />Compute-bound · GEMM / FA</span><span><i className="memory-swatch" />Memory-bound</span><span><i className="yellow" />Saved</span><span><i />Not saved</span><span>Solid: forward</span><span>Dashed: backward dependencies</span><span><i className="purple" />Highlighted dependencies</span></div></div>
        {result ? <Graph model={model} result={result} /> : <p className="ag-empty">Enter a valid configuration to display the graph.</p>}
      </section>

    </div>
  </main>;
}
