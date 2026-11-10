import KernelCosts from './KernelCosts';
import { computeFlops, computeTimes, referenceRates } from './compute';
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
  const [rates, setRates] = useState(referenceRates);
  const data = useMemo(() => {
    try { return { results: levels.map(item => analyze(model, params, item.policy)), flops: computeFlops(model, params), error: null }; }
    catch (e) { return { results: null, flops: undefined, error: (e as Error).message }; }
  }, [model, params]);
  const { results, error, flops } = data;
  const costs = levels.map(item => flops ? computeTimes(flops, rates, item.policy) : null);
  const result = results?.[level];
  return <main className="activation-page ag-single-layer">
    <header className="ag-header">
      <div className="ag-header-brand">
        <div className="ag-brand">ActScope</div>
        <select className="ag-model-select" aria-label="Select model" value={model.id} onChange={e => { location.hash = `/models/${e.target.value}`; }}>
          {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
        </select>
      </div>
      <a className="ag-github-corner" href="https://github.com/niyunsheng/ActScope" target="_blank" rel="noopener noreferrer" aria-label="Star on GitHub" title="Star on GitHub">
        <svg viewBox="0 0 100 100" aria-hidden="true">
          <path className="ag-corner-background" d="M0 0H100V100Z" />
          <path className="ag-corner-logo" transform="translate(57 10) rotate(45 12 12) scale(1.3)" d="M12 .297C5.37.297 0 5.67 0 12.297c0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.043-1.61-4.043-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.09-.745.083-.73.083-.73 1.205.085 1.838 1.237 1.838 1.237 1.07 1.835 2.807 1.305 3.492.998.108-.776.42-1.305.763-1.605-2.665-.305-5.467-1.335-5.467-5.93 0-1.31.468-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23a11.5 11.5 0 0 1 3.003-.404c1.02.005 2.045.138 3.003.404 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.607-2.805 5.625-5.475 5.922.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
        </svg>
      </a>
    </header>
    <div className="ag-main">
      <section className="ag-parameter-block" aria-label="Model and single-layer parameters">
        <div className="ag-fixed-parameters">
          <label className="ag-sequence-field">Token count · T<input aria-label="Token count · T" type="number" min={1} step={1} value={sequence || ''} onChange={e => setSequence(Number(e.target.value))} /></label>
          <div><span>Hidden size</span><strong>{params.hidden}</strong></div>
          <div><span>{model.intermediateLabel ?? 'FFN intermediate'}</span><strong>{params.intermediate}</strong></div>
          <div><span>Attention heads</span><strong>{params.heads}</strong></div>
          <div><span>Head dimension</span><strong>{params.headDim}</strong></div>
          <div><span>Q / K / V width</span><strong>{params.heads * params.headDim}</strong></div>
          <div><span>Activation dtype</span><strong>{model.activationDtype ?? 'BF16'}</strong></div>
          {model.facts?.map(fact => <div key={fact.label}><span>{fact.label}</span><strong>{fact.value}</strong></div>)}
        </div>
      </section>
      {model.note && <p className="ag-model-note">{model.note}</p>}
      {error && <div className="ag-error" role="alert">{error}</div>}
      <KernelCosts flops={flops} rates={rates} times={costs[0]?.ms} onChange={setRates} />
      <fieldset className="ag-strategies"><legend>Saved activations per layer · Select a strategy to update the graph</legend>{levels.map((item,index) => {
        const summary = results?.[index];
        const cost = costs[index];
        return <label className={`ag-strategy ${level === index ? 'active' : ''}`} key={item.label}>
          <input type="radio" name="recompute-level" aria-label={item.label} value={index} checked={level === index} onChange={() => setLevel(index)} />
          <span className="ag-strategy-heading"><span className="ag-level-number">{index}</span><strong>{item.label}</strong><span className="ag-choice-mark" aria-hidden="true">{level === index ? '●' : '○'}</span></span>
          <span className="ag-strategy-description">{index === 0 ? 'No recomputation' : index === 1 ? model.memoryDescription ?? item.detail : item.detail}</span>
          <strong className="ag-strategy-bytes" data-testid={index === level ? 'current-memory' : undefined}>{summary ? formatBytes(summary.total) : '—'}</strong>
          <span className="ag-strategy-time">
            <span>Forward <b data-cost="forward">{cost ? `${cost.forward.toFixed(1)} ms` : '—'}</b></span>
            <span>Backward <b data-cost="backward">{cost ? `${cost.backward.toFixed(1)} ms` : '—'}</b></span>
            <small data-cost="recompute">{cost ? `Includes +${cost.recompute.toFixed(1)} ms recompute` : 'Timing unavailable'}</small>
            <span>Total <b data-cost="total">{cost ? `${cost.total.toFixed(1)} ms` : '—'}</b></span>
            <span>Relative time <b data-cost="relative">{cost ? `${cost.relative.toFixed(1)}%` : '—'}</b></span>
          </span>
          <span className="ag-strategy-footer">{summary ? `${summary.savedCount} tensors saved` : 'Check parameters'}<span>{summary && results ? `Saved ${((1-summary.total/results[0].total)*100).toFixed(1)}%` : ''}</span></span>
        </label>;
      })}</fieldset>
      <section className="ag-graph-panel"><div className="ag-panel-heading ag-graph-heading"><h2>Single-layer graph</h2><div className="ag-legend"><span><i className="compute-swatch" />Compute-bound · GEMM / FA</span><span><i className="memory-swatch" />Memory-bound</span><span><i className="yellow" />Saved</span><span><i />Not saved</span><span>Solid: forward</span><span>Dashed: backward dependencies</span><span><i className="purple" />Highlighted dependencies</span></div></div>
        {result ? <Graph model={model} result={result} /> : <p className="ag-empty">Enter a valid configuration to display the graph.</p>}
      </section>

    </div>
  </main>;
}
