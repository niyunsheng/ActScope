# H3 kernel measurements on the RTX PRO 5000 lab system

These use the **customer image's installed FA4 build** for isolated CUDA kernel
probes for ActScope's H3 workload, not a measured
training step or recomputation strategy. The page keeps the original editable
reference rates and offers **Use measured H3 rates** for H3 at 33,630 tokens.
Applying the rates changes timing estimates; activation accounting stays logical.

## Device and software

Measured on 2026-11-10 Pacific time (2026-11-11 UTC), using one GPU on the lab's
pro5k system. The driver reports
`NVIDIA Graphics Device`, PCI device `0x2BB310DE`, SM 12.0, 110 SMs, and 73,415 MiB
of framebuffer memory; PyTorch reports 76,368,642,048 usable bytes. This is a lab
device, not a verified retail SKU. The page's NVIDIA specification reference
(258 dense BF16 TFLOP/s and 1,344 GB/s) is separate from these observations.
No clocks, power limits, or other users' processes were changed.

Customer image: `oppenheimli/aigc:share.h3.ofld.v3`, registry digest
`sha256:8594ffb9da1e05a1ef598270b45b28dba78f426468cbdb8be57ee5efc10df793`.
The installed FA4 is `0.0.0+h3fl2va.bwdv2.gb011817.pkg1`.
No packages or customer kernel sources were installed, replaced or patched.
The benchmark refuses an unexpected FA4 package version and records the loaded
module path, interface source hash and a hash of the complete Python
source tree. It verifies that CUDA and the occupancy monitor select the same GPU
by UUID. Customer source code is not copied into this repository.

Environment: NVIDIA driver 580.95.05, PyTorch `2.10.0+cu128`, CUDA runtime
12.8, Python 3.10.0, CuTe DSL 4.8.1, and Triton
3.6.0.


## Results

| Probe group | Forward ms | Forward rate | Backward ms | Backward rate |
|---|---:|---:|---:|---:|
| gemm | 115.552 | 224.3 | 236.034 | 219.6 |
| fa | 145.167 | 223.4 | 372.921 | 173.9 |
| memory | 10.635 | 1110.7 | 13.130 | 1285.2 |

GEMM/FA rates are effective TFLOP/s; memory rates are **logical GB/s**.
The device-copy probe measured 1105.8 GB/s
(read + write bytes / CUDA-event time), at 1.744 ms.

The [customer-image run](data/h3-33630.json) uses 10 warmups and 9 samples of
10 iterations; the maximum within-row (max/min − 1) spread was 2.50%.
[Telemetry](data/telemetry.json) contains 36 polls with no foreign compute
process observed. Polls are about two seconds apart and do not prove exclusive
ownership between polls. All GEMM, FA4, memory and copy rows were collected in
the same customer environment.


The previous upstream PyPI b38 runs are retained only as historical comparisons:
[initial](data/upstream-b38-initial.json),
[monitored](data/upstream-b38-monitored.json),
[telemetry](data/upstream-b38-telemetry.json). They use a different image,
PyTorch/CUDA stack and GPU on the lab system; differences cannot be attributed
solely to the FA4 implementation. They are not used by the current preset.

The application pools GEMM forward, input-gradient and weight-gradient FLOPs over
their summed times to obtain one editable GEMM rate. It pools forward/backward
logical bytes over their summed probe times for the shared memory input. Rates
are rounded to one decimal. Pooling reproduces the sum of probe medians (within
rounding) at this workload, but changes the split between forward and backward.
Recomputation time is still modeled as a single replay; it was not measured.

## Workload and measurement method

- One noncausal sequence, BF16, T=33,630, hidden=5,376, 56 heads × 128, FFN=14,336.
  Random inputs with seed 20261110; no model weights are downloaded.
- GEMMs: Q/K/V projections ×3, output projection ×1, gate/up projections ×2,
  down projection ×1. Separate `torch.mm` calls with preallocated outputs for
  forward, dX and dW; no bias, optimizer or gradient accumulation. These logical
  projections do not benchmark a packed QKV or gate/up implementation.
- Customer FA4: THD variable-length API with one document and LSE, plus its private
  backward entry point. Forward FLOPs = `4*T*T*56*128`; effective backward FLOPs =
  twice forward, excluding FA's internal QK replay. No causal mask or dropout.
- Memory probes: compiled RMSNorm (hidden and per-head), modulation, gated
  residual, SwiGLU and 96-channel partial RoPE. Each operator is compiled
  separately with `torch.compile(fullgraph=True)`; no cross-operator fusion or
  CUDA graph capture. Norm affine and conditioning gradients are excluded.
  Modulation and residual gates use fixed scalar conditioning. Backward reuses
  one saved forward graph with AOTAutograd buffer donation disabled.
- Each row records all CUDA-event samples, minimum, median and maximum. Each
  sample times repeated calls, synchronizing its end event. Compilation,
  correctness checks and warmup occur before sampling. Allocation for memory/FA
  outputs is part of those calls; GEMM output allocation is outside timing.
- The reported group time is the sum of per-shape medians weighted by operator
  multiplicity. It is not a median of end-to-end layer runs.
- FA output/dQ/dK/dV are checked against FP32 SDPA on a small sequence. Compiled
  memory outputs/gradients are checked against eager PyTorch on small slices.
  These are numerical smoke checks, not full-model correctness validation.

**Logical GB/s is not measured DRAM bandwidth or MBU.** Its numerator is the
frontend's main-tensor traffic formula, not hardware counter data. In particular,
fixed-conditioning backward can eliminate reads, return an aliased gradient, or
omit conditioning reductions that the model's logical traffic includes. Some
per-operator logical rates can therefore exceed the hardware bandwidth reference.
The copy probe is reported separately and is not used to calibrate all operators.
No GPU-memory allocation result is presented as activation memory.

Excluded: conditioning/affine gradients, communication, optimizer, full-layer
scheduling, checkpoint machinery, training data and cross-operator fusion.
Results at other shapes, on LingBot MoE, on another GPU, or with another kernel
implementation are extrapolations. The preset button is disabled outside the
measured model and token count; changing the workload after applying rates shows
an extrapolation notice. Hardware table edits remain specification inputs and do
not recalibrate the measured throughput.

## Reproduce

The GPU host needs access to the customer image, Docker with NVIDIA Container
Toolkit, and an idle GPU. Use a
new container and directory; the monitor refuses an already occupied GPU and
stops only this benchmark if another compute process appears during polling.
This is a contention check, not a scheduler reservation.

```bash
gpu_index=2
bench_dir=$(mktemp -d /tmp/actscope-kernels-XXXXXX)
container_name=$(basename "$bench_dir")
cp scripts/benchmark-kernels.py scripts/monitor-kernel-benchmark.py "$bench_dir/"
docker run --rm -d --name "$container_name" --gpus "device=$gpu_index" \
  --shm-size=2g -v "$bench_dir:/bench" --entrypoint sleep \
  oppenheimli/aigc@sha256:8594ffb9da1e05a1ef598270b45b28dba78f426468cbdb8be57ee5efc10df793 infinity
python3 scripts/monitor-kernel-benchmark.py --container "$container_name" \
  --gpu "$gpu_index" --directory "$bench_dir"
docker stop "$container_name"
# Results and telemetry remain in "$bench_dir".
```

The measured host had a separate faulty GPU that prevented the NVIDIA runtime's
global device discovery from starting new GPU containers. This run used a new
`runc` container with only the healthy GPU's device nodes and read-only host
driver-library mounts. The customer image's libraries and packages were left
intact. CUDA selection and host monitoring were both pinned to the healthy GPU's
UUID; no GPU reset or driver reconfiguration was performed.

For a direct run in an already isolated customer-image CUDA environment:

```bash
python3 scripts/benchmark-kernels.py --warmup 10 --samples 9 --iterations 10 \
  --image-ref oppenheimli/aigc:share.h3.ofld.v3 \
  --output h3-33630.json
```

The JSON includes the benchmark script's SHA-256, versions, device observations,
workload, sample counts and completion time. An incomplete JSON without
`finished_at_utc` must not be used as a preset. Frontend tests check that benchmark
FLOPs and logical traffic agree with the current model and that all samples exist.

## Sources

- [H3 config supplied for this project](../src/activation/h3-config.ts) and
  [model/operator assumptions](../src/activation/model.ts).
- Customer FA4: the pinned image digest and source hashes in
  [the raw environment record](data/h3-33630.json). The upstream
  [FlashAttention implementation](https://github.com/Dao-AILab/flash-attention)
  supplies API context, not the measured customer build.
- [PyTorch CUDA event timing](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html).
- [NVIDIA RTX PRO 5000 specification reference](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-gtc25-spring-nvidia-3658700.pdf)
  and [Blackwell PRO architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf).
