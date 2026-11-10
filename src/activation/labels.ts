// Shared display names keep equivalent modules consistent across model graphs.
export const operatorLabels = {
  n1: 'rmsnorm', n2: 'rmsnorm', m1: 'modulate', m2: 'modulate',
  qp: 'q_proj', kp: 'k_proj', vp: 'v_proj', o: 'o_proj',
  qn: 'q_rmsnorm', kn: 'k_rmsnorm', qr: 'q_rope', kr: 'k_rope',
  fa: 'flash_attn', residual: 'gated_residual',
} as const;

export const tensorLabels = {
  input: 'Block input', output: 'Block output',
  norm1: 'Norm output', mod1: 'Modulate output',
  q: 'Q', k: 'K', v: 'V', qn: 'Q norm', kn: 'K norm',
  qr: 'Q RoPE', kr: 'K RoPE', attention: 'FA output', out: 'O projection',
  residual: 'Attention residual', norm2: 'FFN norm', mod2: 'FFN modulate',
} as const;
