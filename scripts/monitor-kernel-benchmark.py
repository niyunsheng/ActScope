#!/usr/bin/env python3
"""Run in an existing dedicated Docker container; abort our benchmark on GPU contention.

Run on the GPU host, after mounting a private output directory at /bench in the
container. Saves telemetry beside the local log; does not modify GPU settings.
"""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# The host runner does not import torch; the child also enforces this version.
CUSTOMER_FA_VERSION = "0.0.0+h3fl2va.bwdv2.gb011817.pkg1"


def output(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--expected-fa-version", default=CUSTOMER_FA_VERSION)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)

    def processes():
        raw = output("nvidia-smi", "-i", args.gpu,
                     "--query-compute-apps=pid", "--format=csv,noheader,nounits")
        return {int(line) for line in raw.splitlines() if line.strip()}

    if processes():
        raise RuntimeError("Selected GPU already has compute processes; benchmark not started")
    gpu_uuid = output("nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader")
    image_ref = output("docker", "inspect", args.container, "--format", "{{.Config.Image}}")
    image_id = output("docker", "inspect", args.container, "--format", "{{.Image}}")
    digests = json.loads(output("docker", "image", "inspect", image_id, "--format", "{{json .RepoDigests}}"))
    image_args = ["--image-ref", image_ref, "--image-id", image_id, "--gpu-query-id", gpu_uuid,
                  "--expected-fa-version", args.expected_fa_version, "--expected-gpu-uuid", gpu_uuid]
    if digests:
        image_args += ["--image-digest", digests[0]]
    telemetry = []
    with (args.directory / "run.log").open("w") as log:
        child = subprocess.Popen([
            "docker", "exec", "-e", f"CUDA_VISIBLE_DEVICES={gpu_uuid}", args.container,
            "python3", "-u", "/bench/benchmark-kernels.py",
            "--warmup", "10", "--samples", "9", "--iterations", "10",
            "--output", "/bench/h3-33630.json",
            *image_args,
        ], stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                # Read active GPU PIDs before Docker's list, so a just-started
                # benchmark process is already visible in the latter.
                active = processes()
                owned = {int(line.split()[0]) for line in output(
                    "docker", "top", args.container, "-eo", "pid").splitlines()[1:]}
                foreign = active - owned
                sample = {
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "gpu_compute_process_count": len(active),
                    "foreign_compute_process_count": len(foreign),
                    "gpu": output("nvidia-smi", "-i", args.gpu,
                                  "--query-gpu=utilization.gpu,memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw",
                                  "--format=csv,noheader,nounits"),
                }
                telemetry.append(sample)
                if foreign:
                    raise RuntimeError("Another GPU compute process appeared; discarding this run")
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"Benchmark exited with {child.returncode}; see run.log")
        finally:
            if child.poll() is None:
                # Only our benchmark inside our dedicated container is stopped.
                subprocess.run(["docker", "exec", args.container, "pkill", "-f",
                                "^python3 -u /bench/benchmark-kernels.py"], check=False)
                child.wait()
            (args.directory / "telemetry.json").write_text(json.dumps({
                "columns": ["gpu_util_percent", "memory_used_mib", "sm_mhz", "memory_mhz", "temperature_c", "power_w"],
                "samples": telemetry,
            }, indent=2) + "\n")


if __name__ == "__main__":
    main()
