import type { Operator, Parameters, RuleKey, Tensor } from './model';

type Position = { x: number; y: number };
type OperatorOptions = Pick<Operator, 'weightShape' | 'inputLanes' | 'inputBranches' | 'forwardOnly'>;
export const hidden = (p: Parameters) => [p.sequence, p.hidden];

export const tensorFactory = (defaultDtype?: Tensor['dtype']) =>
  (id: string, label: string, position: Position, shape = hidden, dtype = defaultDtype): Tensor =>
    ({ id, label, ...position, shape, dtype });

export const operator = (id: string, label: string, rule: RuleKey, inputs: string[], outputs: string[], backward: string[], position: Position, options: OperatorOptions = {}): Operator =>
  ({ id, label, rule, category: rule === 'linear' ? 'gemm' : (rule === 'attention' || rule === 'kda') ? 'fa' : 'memory', inputs, outputs, backward, ...position, ...options });
