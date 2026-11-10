import { kernels, type KernelRates, type KernelValues, type computeTimes } from './compute';
import type { memoryTraffic } from './memoryTraffic';
import { positive, type Hardware } from './hardware';

export default function KernelCosts({ flops, traffic, rates, times, hardware, onChange }: {
  flops?: KernelValues; traffic?: ReturnType<typeof memoryTraffic>; rates: KernelRates;
  times?: NonNullable<ReturnType<typeof computeTimes>>['ms']; hardware: Hardware;
  onChange: (rates: KernelRates) => void;
}) {
  const invalid = Object.values(rates).some(rate => !positive(rate));
  return <section className="ag-kernel-cost" aria-label="Kernel cost">
    <h2>Kernel throughput</h2>
    <table>
      <thead><tr><th>Kernel</th><th>Throughput</th><th>Unit</th><th>Effective / peak</th></tr></thead>
      <tbody>{kernels.map(k => {
        const peak = k.id === 'memory' ? hardware.bandwidth : hardware.peak;
        return <tr key={k.id} data-rate={k.id}>
          <th scope="row">{k.label}</th>
          <td>{(k.id === 'gemm' ? ['gemm', 'backwardGemm'] as const : [k.id]).map(id => <label className="ag-rate-input" key={id}>
            {k.id === 'gemm' && <span>{id === 'gemm' ? 'Fwd' : 'Bwd'}</span>}
            <input aria-label={`${id === 'backwardGemm' ? 'GEMM Backward' : k.label} kernel ${k.unit}`} type="number" min="0.01" step="any" value={rates[id] || ''} aria-invalid={!positive(rates[id])} onChange={e => onChange({ ...rates, [id]: Number(e.target.value) })} />
          </label>)}</td>
          <td>{k.unit}</td><td data-peak-ratio>{positive(peak) && positive(rates[k.id]) ? `${(100*rates[k.id]/peak).toFixed(1)}%` : '—'}{k.id === 'gemm' && <><br />{positive(peak) && positive(rates.backwardGemm) ? `${(100*rates.backwardGemm/peak).toFixed(1)}%` : '—'}</>}</td>
        </tr>;
      })}</tbody>
    </table>
    {invalid && <p role="alert">Enter a positive throughput for every kernel.</p>}
    <p>Compute defaults use your RTX PRO 5000 measurements. Memory defaults to 1,344 GB/s peak: an optimistic estimate, not measured bandwidth. Edit for your workload.</p>
    <details className="ag-work-details"><summary>Work and time per layer</summary>
      <table><thead><tr><th>Computation</th><th>Work</th><th>Unit</th><th>Time (ms)</th></tr></thead><tbody>
        {([
          ['forwardGemm', 'Forward GEMM', flops?.forwardGemm, 1e12, 'TFLOPs'],
          ['forwardFa', 'Forward FA', flops?.forwardFa, 1e12, 'TFLOPs'],
          ['backwardGemm', 'Backward GEMM', flops?.backwardGemm, 1e12, 'TFLOPs'],
          ['backwardFa', 'Backward FA', flops?.backwardFa, 1e12, 'TFLOPs'],
          ['forwardMemory', 'Forward memory', traffic?.forwardMemory, 1e9, 'GB'],
          ['backwardMemory', 'Backward memory', traffic?.backwardMemory, 1e9, 'GB'],
        ] as const).map(([id, label, work, divisor, unit]) => <tr key={id} data-kernel={id}><th scope="row">{label}</th><td>{work === undefined ? '—' : (work/divisor).toFixed(2)}</td><td>{unit}</td><td>{times ? times[id].toFixed(1) : '—'}</td></tr>)}
      </tbody></table>
      <p>One fully connected sequence · Multiply-add = 2 FLOPs · FA backward uses effective FLOPs. Selected operations replay once; retained top-k selection is not replayed.</p>
      <p>Memory time = logical main-tensor reads + writes / effective GB/s. Materialized kernels and gradients; excludes communication, launch latency, caches, cross-kernel fusion, auxiliary buffers and conditioning. The bandwidth ratio is not measured MBU.</p>
    </details>
  </section>;
}
