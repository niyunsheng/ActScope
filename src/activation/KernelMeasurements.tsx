import { measuredGroups, measuredRates, measurementWorkload, measurementMethod, measurementEnvironment } from './calibration';
import { referenceRates, type KernelRates } from './compute';

export default function KernelMeasurements({ modelId, sequence, rates, onChange }: {
  modelId: string; sequence: number; rates: KernelRates; onChange: (rates: KernelRates) => void;
}) {
  const matches = modelId === measurementWorkload.model && sequence === measurementWorkload.tokens;
  const selected = Object.entries(measuredRates).every(([key, value]) => rates[key as keyof KernelRates] === value);
  return <details className="ag-work-details ag-measurements">
    <summary>RTX PRO 5000 lab measurements · H3 / 33,630 tokens</summary>
    <p>BF16 · Customer FA4 · {measurementMethod.samples} samples after warmup · CUDA event timing. Lab device reports “NVIDIA Graphics Device”, 110 SMs, SM 12.0, about 72 GiB. Hardware peak values above are a separate specification reference.</p>
    <p data-measurement-source>Customer image: {measurementEnvironment.container.image_ref}<br />FA4 build: {measurementEnvironment.fa4.version}</p>
    <table aria-label="Measured kernel throughput">
      <thead><tr><th>Probe</th><th>Forward</th><th>Backward</th><th>Unit</th></tr></thead>
      <tbody>{measuredGroups.map(group => <tr key={group.id}>
        <th scope="row">{group.label}</th><td>{group.forward.rate.toFixed(1)}</td><td>{group.backward.rate.toFixed(1)}</td><td>{group.unit}</td>
      </tr>)}</tbody>
    </table>
    <p>Memory probes use fixed conditioning and compiled operators; logical GB/s is not measured DRAM bandwidth. Shared GEMM and memory inputs pool forward/backward work over time. Layer and recomputation times remain estimates; activation sizes remain logical estimates.</p>
    <div className="ag-measurement-actions">
      <button type="button" disabled={!matches} onClick={() => onChange({ ...measuredRates })}>Use measured H3 rates</button>
      <button type="button" onClick={() => onChange({ ...referenceRates })}>Reset reference rates</button>
    </div>
    <p role="status" aria-label="Calibration status">{!matches
      ? 'Measurements cover H3 at 33,630 tokens only. Any use of these rates at another workload is an extrapolation.'
      : selected ? 'Measured probe rates selected. Strategy timings are estimates, not measured layer runtimes.'
      : 'Reference or edited rates are active. Apply the measured probe rates to compare estimates.'}</p>
    <p><a href="https://github.com/niyunsheng/ActScope/tree/codex/kernel-cost-estimates/benchmarks" target="_blank" rel="noopener noreferrer">Raw samples, reproduction steps and limitations</a></p>
  </details>;
}
