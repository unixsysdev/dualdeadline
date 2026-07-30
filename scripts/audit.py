#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
import transformers

from specstream.io import atomic_json, git_revision, utc_now


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def source_hashes(root: Path) -> dict[str, str]:
    paths = [
        *root.glob("scripts/*"),
        *root.glob("src/**/*.py"),
        root / "requirements.txt",
        root / "pyproject.toml",
        root / "protocol" / "pilot.yaml",
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    config_path = args.model / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    report = {
        "captured_at_utc": utc_now(),
        "git_revision": git_revision(args.root),
        "git_status_short": command(
            "git", "-C", str(args.root), "status", "--short"
        ),
        "source_sha256": source_hashes(args.root),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            {
                "name": gpu.name,
                "total_memory_bytes": gpu.total_memory,
                "compute_capability": [gpu.major, gpu.minor],
                "multi_processor_count": gpu.multi_processor_count,
            }
            if gpu
            else None
        ),
        "nvidia_smi": command(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,pstate,pcie.link.gen.current,"
            "pcie.link.width.current,memory.total",
            "--format=csv,noheader",
        ),
        "model_path": str(args.model.resolve()),
        "model_config": config,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
