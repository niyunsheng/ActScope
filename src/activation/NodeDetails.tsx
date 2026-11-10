import type { Analysis } from './engine';
import type { Operator } from './model';

type Row = Analysis['rows'][number];
const shape = (dimensions: number[]) => `[${dimensions.join(', ')}]`;
const features = (tensor: Row) => tensor.dimensions.slice(1).reduce((a, b) => a * b, 1);

export default function NodeDetails({ operator, tensor, rows, operators }: {
  operator?: Operator; tensor?: Row; rows: Row[]; operators: Operator[];
}) {
  const inputs = operator?.inputs.map(id => rows.find(row => row.id === id)!) ?? [];
  const outputs = operator?.outputs.map(id => rows.find(row => row.id === id)!) ?? [];
  return <span className="ag-node-summary" aria-label="Node information">
    {operator ? <>
      <strong>{operator.label}</strong>
      {operator.category === 'gemm' && <> · W {operator.weightShape?.length === 3 ? '[experts, out, in]' : '[out, in]'} <code data-testid="weight-shape">{shape(operator.weightShape ?? [outputs.reduce((sum, row) => sum + features(row), 0), features(inputs[0])])}</code></>}
      {operator.forwardOnly ? <span> · Discrete index selection · No backward</span> : <span> · Bwd needs: {operator.backward.map(id => rows.find(row => row.id === id)!).filter(row => !row.metadata).map(row => row.label).join(', ') || 'None (activations)'}</span>}
    </> : tensor ? <>
      <strong>{tensor.label}</strong> · <code data-testid="tensor-shape">{shape(tensor.dimensions)}</code>
      {' · '}{tensor.dtype && tensor.dtype !== 'BF16' && `${tensor.dtype} · `}<strong data-testid="tensor-size">{(tensor.bytes / 1024 ** 2).toFixed(2)} MiB</strong>
      {' · '}{tensor.keep ? 'Saved' : 'Not retained'}
      <span> · Bwd used by: {operators.filter(op => op.backward.includes(tensor.id)).map(op => op.label).join(', ') || 'None'}</span>
    </> : 'Scroll to zoom · Drag graph or nodes to pan'}
  </span>;
}
