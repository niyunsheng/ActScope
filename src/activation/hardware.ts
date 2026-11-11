// RTX PRO 5000 Blackwell: NVIDIA datasheet lists 2,064 sparse FP4 TOPS and
// 1,344 GB/s GDDR7 bandwidth. Approximate dense BF16 = 2064 / 8 = 258 TFLOP/s,
// derived from the GB202 dense BF16 : sparse FP4 ratio (architecture appendix A).
// https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf
// Checked 2026-11-10. This is a boost-clock peak reference, not measured throughput.
export const gpus = [
  { id: 'rtx-pro-5000-blackwell', name: 'RTX PRO 5000 Blackwell', peak: 258, bandwidth: 1344, source: 'https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-gtc25-spring-nvidia-3658700.pdf' },
];
export type Hardware = { id: string; peak: number; bandwidth: number };
export const positive = (n: number) => Number.isFinite(n) && n > 0;
