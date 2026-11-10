import { gpus, positive, type Hardware } from './hardware';

export default function HardwareTable({ hardware, onChange }: { hardware: Hardware; onChange: (h: Hardware) => void }) {
  const preset = gpus.find(g => g.id === hardware.id);
  return <section className="ag-kernel-cost" aria-label="GPU hardware">
    <h2>GPU hardware · Per GPU</h2>
    <table><tbody>
      <tr><th scope="row">GPU</th><td><select aria-label="GPU" value={hardware.id} onChange={e => {
        const gpu = gpus.find(g => g.id === e.target.value);
        onChange(gpu ?? { ...hardware, id: 'custom' });
      }}>{gpus.map(g => <option key={g.id} value={g.id}>{g.name}</option>)}<option value="custom">Custom</option></select></td></tr>
      {([{ key: 'peak', label: 'BF16 dense peak (TFLOP/s)' }, { key: 'bandwidth', label: 'Memory bandwidth (GB/s)' }] as const).map(field => <tr key={field.key}>
        <th scope="row">{field.label}</th><td><input aria-label={field.label} type="number" min="0.01" step="any" value={hardware[field.key] || ''} aria-invalid={!positive(hardware[field.key])} onChange={e => onChange({ ...hardware, id: 'custom', [field.key]: Number(e.target.value) })} /></td>
      </tr>)}
    </tbody></table>
    {(!positive(hardware.peak) || !positive(hardware.bandwidth)) && <p role="alert">Enter positive GPU peak values.</p>}
    <p>{preset && <><a href={preset.source} target="_blank" rel="noreferrer">NVIDIA specifications</a> · </>}Dense BF16 ≈ 258 TFLOP/s derived from sparse FP4 / 8; GDDR7 bandwidth. Peak references only.</p>
  </section>;
}
