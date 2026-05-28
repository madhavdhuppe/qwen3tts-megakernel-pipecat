"""Verify the rented RTX 5090 environment before running real TTS."""

from __future__ import annotations

import platform
import subprocess
import sys


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"FAILED: {exc}"


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print("nvidia-smi:")
    print(run(["nvidia-smi"]))

    try:
        import torch
    except ImportError:
        print("torch=missing")
        return 1

    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return 1

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"gpu_name={name}")
    print(f"compute_capability={capability[0]}.{capability[1]}")

    if "5090" not in name:
        print("warning=GPU name does not look like RTX 5090")
    if capability[0] < 12:
        print("error=Blackwell sm_120+ capability is required for this kernel")
        return 1

    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    print(f"bf16_matmul_ok={tuple(y.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
