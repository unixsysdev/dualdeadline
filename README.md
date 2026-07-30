# DualDeadline

[![Build paper](https://github.com/unixsysdev/dualdeadline/actions/workflows/paper.yml/badge.svg)](https://github.com/unixsysdev/dualdeadline/actions/workflows/paper.yml)

DualDeadline is a reproducible study of component-staged expert prefetching for
exact, offloaded mixture-of-experts inference. It asks whether a gated expert
must have one transfer deadline.

The predictor speculates only on the gate and up projections. After the
unchanged native router reveals the true top-k experts, their down projections
move to the GPU concurrently with gate/up computation. Prediction is advisory:
misses are fetched on demand, so routing and model outputs remain exact.

The current paper is [output/pdf/main.pdf](output/pdf/main.pdf), and the
machine-readable preregistration and amendments are in
[protocol/pilot.yaml](protocol/pilot.yaml).
GitHub Actions recompiles the manuscript for paper-related pushes and pull
requests and retains the resulting `dualdeadline-paper` artifact for 30 days.

## Current evidence

On OLMoE-1B-7B-Instruct and one PCIe Gen5 x16 NVIDIA H200:

- a 120-prompt held-out pilot reaches 45.23% expert recall with four
  candidates, versus 26.74% for training-set popularity;
- a prompt-disjoint 300-prompt replication reaches 46.63%;
- cacheless trace replay at equal speculative bytes estimates 1.223 ms exposed
  stall for monolithic prefetch and 1.106 ms for component staging;
- a real pinned-memory, two-CUDA-stream benchmark reduces an eight-expert block
  from 2.193 ms to 1.939 ms with zero numerical error; and
- the measured benefit persists across all 16 layers and under prompt-cold
  per-layer LRU capacities from 8 to 64 experts; and
- a custom two-kernel Triton predictor cuts p50 from 0.0989 ms to 0.0635 ms
  while preserving top-2/4/8 expert sets on all 32,520 held-out pairs.

The prespecified Qwen3.6-35B-A3B confirmation uses 88,920 held-out target
pairs:

- learned R@4 is 31.28%, versus 13.40% for training-set popularity;
- cacheless replay at budget four estimates 0.666 ms monolithic stall and
  0.505 ms component-staged stall;
- real CUDA-stream checks on layers 0, 20, and 39 reduce block latency by
  28.8%, 24.1%, and 16.2%, respectively, with zero numerical error; and
- the custom Triton predictor cuts p50 from 0.0968 ms to 0.0654 ms while
  preserving held-out top-2/4/8 sets.

These are held-out trace results, trace-driven simulations, and isolated
microbenchmarks—not an end-to-end serving-speedup claim. The Qwen model used
the Transformers/PyTorch fallback consistently because its optional fast path
was unavailable in this environment.

## Reproduce

Python 3.10, CUDA 12.8, PyTorch 2.9.1, Transformers 5.14.1, and Triton 3.5.1
were used for the reported H200 runs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

Model checkpoints, detached traces, and learned `.pt` checkpoints are excluded
from Git because of size. JSON/CSV reports, corpus manifests, figures, the
paper, and hardware timing records are versioned. The core stages are:

```bash
python scripts/collect_decode_traces.py --help
python scripts/benchmark_model.py --help
python scripts/benchmark_staged_overlap.py --help
python scripts/train_predictor.py --help
python scripts/evaluate_predictor.py --help
python scripts/simulate_prefetch.py --help
python scripts/simulate_cache.py --help
```

`scripts/run_qwen_confirmation.sh` is the unattended H200 pipeline used by the
study, including the checkpoint-level kernel-equivalence postcheck. It assumes
the VM layout `/root/dataDisk/specstream`; individual Python commands are
path-independent. `scripts/merge_cache_results.py` validates and merges
capacity-sharded cache jobs; this parallel execution is semantically equivalent
to the serial multi-capacity command in the canonical pipeline.

## Repository map

- `src/specstream/`: predictor, trace loading, metrics, and atomic I/O
- `scripts/`: collection, training, evaluation, simulation, and H200 benchmarks
- `artifacts/results/`: held-out reports and prompt-bootstrap intervals
- `artifacts/timing/`: pinned-copy, layer-window, predictor, and overlap timing
- `artifacts/corpus/`: prompt-grouped pilot and disjoint-replication manifests
- `paper/`: LaTeX source, bibliography, and generated figures
- `protocol/`: frozen protocol plus timestamped amendments

The base models are never fine-tuned. Only the small post-hoc route predictor
is trained.
