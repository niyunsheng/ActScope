import { kernels, type KernelValues } from './compute';

export default function KernelCosts({ flops, rates, times, onChange }: {
  flops?: KernelValues; rates: KernelValues; times?: KernelValues;
  onChange: (rates: KernelValues) => void;
}) {
  const invalid = kernels.some(k => !Number.isFinite(rates[k.id]) || rates[k.id] <= 0);
  return <section className="ag-kernel-cost" aria-label="Kernel cost">
    <h2>Kernel cost · Per layer</h2>
    <table>
      <thead><tr><th>Computation</th><th>TFLOPs</th><th>Kernel TFLOP/s</th><th>Time (ms)</th></tr></thead>
      <tbody>{kernels.map(k => <tr key={k.id} data-kernel={k.id}>
        <th scope="row">{k.label}</th>
        <td>{flops ? (flops[k.id] / 1e12).toFixed(2) : '—'}</td>
        <td><input aria-label={`${k.label} kernel TFLOP/s`} type="number" min="0.01" step="any" value={rates[k.id] || ''} aria-invalid={rates[k.id] <= 0} onChange={e => onChange({ ...rates, [k.id]: Number(e.target.value) })} /></td>
        <td>{times ? times[k.id].toFixed(1) : '—'}</td>
      </tr>)}</tbody>
    </table>
    {invalid && <p role="alert">Enter a positive throughput for every kernel.</p>}
    <p>One fully connected sequence · Multiply-add = 2 FLOPs · FA backward uses effective FLOPs.</p>
    <p>Initial rates are user-supplied H3 reference measurements; edit for your hardware and workload. Memory-bound and communication time excluded. Selected GEMM / FA operations replay once during backward.</p>
  </section>;
}
