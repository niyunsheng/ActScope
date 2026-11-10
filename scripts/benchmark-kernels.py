#!/usr/bin/env python3
"""H3-shaped CUDA microbenchmarks; see benchmarks/README.md for scope and units."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import torch._functorch.config


def rms(x):
    y = x.float()
    return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-5)).to(x.dtype)


def modulate(x):
    # Fixed scalar conditioning; no conditioning/affine parameter gradients.
    return x * 1.125 + 0.125


def residual(x, branch):
    return x + branch * 0.75


def swiglu(gate, up):
    return F.silu(gate) * up


def rope(x, cos, sin):
    a, b = x[..., :96].chunk(2, dim=-1)
    rotated = torch.cat((-b, a), dim=-1)
    return torch.cat((x[..., :96] * cos + rotated * sin, x[..., 96:]), dim=-1)


def summary(samples):
    return {"samples_ms": samples, "median_ms": statistics.median(samples),
            "min_ms": min(samples), "max_ms": max(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=33630)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if min(args.tokens, args.warmup, args.samples, args.iterations) < 1:
        parser.error("counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; no CPU fallback")
    torch.manual_seed(20261110)
    torch.set_num_threads(4)
    # Reuse one saved forward graph while timing backward alone. Donating its
    # buffers would invalidate retain_graph=True after the first iteration.
    torch._functorch.config.donated_buffer = False
    device = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "packages": {p: importlib.metadata.version(p) for p in
                         ("flash-attn-4", "triton", "nvidia-cutlass-dsl")},
            "gpu_name": device.name, "compute_capability": [device.major, device.minor],
            "sm_count": device.multi_processor_count, "total_memory_bytes": device.total_memory,
            "nvidia_smi": subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,pci.device_id,driver_version,memory.total,power.limit",
                "--format=csv"], text=True).strip(),
        },
        "workload": {"model": "minimax-h3-dit", "tokens": args.tokens, "hidden": 5376,
                     "heads": 56, "head_dim": 128, "intermediate": 14336,
                     "dtype": "bfloat16", "causal": False, "documents": 1},
        "method": {"timer": "CUDA events, synchronized per sample", "warmup": args.warmup,
                   "samples": args.samples, "iterations_per_sample": args.iterations,
                   "seed": 20261110, "cuda_graphs": False,
                   "aot_autograd_donated_buffer": False,
                   "memory_kernels": "torch.compile fullgraph, default mode; fixed conditioning",
                   "fa_backward_flops": "2x forward (effective); excludes internal QK replay"},
        "rows": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def tensor(shape, grad=False):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=grad)

    def measure(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.samples):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.iterations):
                fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / args.iterations)
        return summary(samples)

    def record(row):
        report["rows"].append(row)
        save()
        print(json.dumps(row), flush=True)

    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()

    T, H, A, I = args.tokens, 5376, 7168, 14336

    def gemm(name, k, n, count):
        x, w, dy = tensor((T, k)), tensor((n, k)), tensor((T, n))
        y, dx, dw = torch.empty_like(dy), torch.empty_like(x), torch.empty_like(w)
        flops = 2 * T * k * n
        for direction, fn in [
            ("forward", lambda: torch.mm(x, w.T, out=y)),
            ("backward_input", lambda: torch.mm(dy, w, out=dx)),
            ("backward_weight", lambda: torch.mm(dy.T, x, out=dw)),
        ]:
            timing = measure(fn)
            record({"group": "gemm", "name": name, "direction": direction,
                    "shape_mkn": [T, k, n], "count": count, "flops": flops,
                    **timing, "tflops": flops / timing["median_ms"] / 1e9})

    for spec in [("qkv_projection", H, A, 3), ("output_projection", A, H, 1),
                 ("gate_up_projection", H, I, 2), ("down_projection", I, H, 1)]:
        gemm(*spec)
        cleanup()

    def attention():
        from flash_attn.cute import flash_attn_varlen_func
        from flash_attn.cute.interface import _flash_attn_bwd

        def fa(q, k, v, bounds, length):
            return flash_attn_varlen_func(q, k, v, cu_seqlens_q=bounds, cu_seqlens_k=bounds,
                                          max_seqlen_q=length, max_seqlen_k=length,
                                          causal=False, return_lse=True)

        def bwd(q, k, v, output, grad, lse, bounds, length):
            return _flash_attn_bwd(q, k, v, output, grad, lse,
                                  cu_seqlens_q=bounds, cu_seqlens_k=bounds,
                                  max_seqlen_q=length, max_seqlen_k=length, causal=False)

        # Check output and all three gradients against FP32 SDPA before timing.
        small = [tensor((128, 2, 128), True) for _ in range(3)]
        bounds = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
        actual, lse = fa(*small, bounds, 128)
        reference = F.scaled_dot_product_attention(
            *[x.float().transpose(0, 1).unsqueeze(0) for x in small]
        ).squeeze(0).transpose(0, 1)
        torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
        grad = torch.randn_like(actual)
        actual_grads = bwd(*small, actual, grad, lse, bounds, 128)
        ref_grads = torch.autograd.grad(reference, small, grad.float())
        for a, b in zip(actual_grads, ref_grads):
            torch.testing.assert_close(a, b, atol=0.02, rtol=0.03)
        report["fa_correctness"] = "BF16 output/dQ/dK/dV passed FP32 SDPA comparison at T=128"
        q, k, v = [tensor((T, 56, 128)) for _ in range(3)]
        bounds = torch.tensor([0, T], device="cuda", dtype=torch.int32)
        fn = lambda: fa(q, k, v, bounds, T)
        output, lse = fn()
        grad = torch.randn_like(output)
        backward = lambda: bwd(q, k, v, output, grad, lse, bounds, T)
        for direction, call, factor in [("forward", fn, 1), ("backward", backward, 2)]:
            timing = measure(call)
            work = 4 * T * T * A * factor
            record({"group": "fa", "name": "fa4_varlen", "direction": direction,
                    "count": 1, "flops": work, **timing,
                    "tflops": work / timing["median_ms"] / 1e9})

    attention()
    cleanup()

    def memory(name, fn, shapes, count, forward_elements, backward_elements, extras=()):
        inputs = [tensor(shape, True) for shape in shapes]
        compiled = torch.compile(fn, fullgraph=True)
        output = compiled(*inputs, *extras)
        grad = torch.randn_like(output)
        backward = lambda: torch.autograd.grad(output, inputs, grad, retain_graph=True)
        # Small slices keep correctness validation bounded; use the same compiled function.
        small = [x[:2].detach().requires_grad_() for x in inputs]
        extra_small = tuple(x[:2] for x in extras)
        actual = compiled(*small, *extra_small)
        ref = fn(*small, *extra_small)
        torch.testing.assert_close(actual, ref, atol=0.04, rtol=0.04)
        small_grad = torch.randn_like(actual)
        for a, b in zip(torch.autograd.grad(actual, small, small_grad),
                        torch.autograd.grad(ref, small, small_grad)):
            torch.testing.assert_close(a, b, atol=0.04, rtol=0.04)
        for direction, call, elements in [
            ("forward", lambda: compiled(*inputs, *extras), forward_elements),
            ("backward", backward, backward_elements),
        ]:
            timing = measure(call)
            record({"group": "memory", "name": name, "direction": direction,
                    "count": count, "shapes": shapes, "logical_bytes": elements * 2,
                    **timing, "logical_gbps": elements * 2 / timing["median_ms"] / 1e6})

    specs = [
        ("hidden_rmsnorm", rms, [(T, H)], 2, 2*T*H, 3*T*H),
        ("qk_rmsnorm", rms, [(T, 56, 128)], 2, 2*T*A, 3*T*A),
        ("modulate", modulate, [(T, H)], 2, 2*T*H, 3*T*H),
        ("residual", residual, [(T, H), (T, H)], 2, 3*T*H, 4*T*H),
        ("swiglu", swiglu, [(T, I), (T, I)], 1, 3*T*I, 5*T*I),
    ]
    for spec in specs:
        memory(*spec)
        cleanup()
    angles = tensor((T, 1, 48)).float()
    angles = torch.cat((angles, angles), dim=-1)
    memory("rope", rope, [(T, 56, 128)], 2, 2*T*A, 2*T*A,
           (angles.cos().bfloat16(), angles.sin().bfloat16()))
    cleanup()
    report["memory_correctness"] = "Compiled forward/backward passed eager comparison on two token rows"

    # A separate device-copy probe is not a calibration of every memory kernel.
    src, dst = tensor((T, I)), torch.empty((T, I), device="cuda", dtype=torch.bfloat16)
    timing = measure(lambda: dst.copy_(src))
    record({"group": "copy", "name": "device_copy", "direction": "forward", "count": 1,
            "logical_bytes": src.numel()*src.element_size()*2, **timing,
            "logical_gbps": src.numel()*src.element_size()*2 / timing["median_ms"] / 1e6})
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == "__main__":
    main()
