# SpecStream

SpecStream is a reproducible feasibility study of deadline-aware, staged expert
prefetching for exact mixture-of-experts inference. The first target is
`Qwen/Qwen3.6-35B-A3B` on one NVIDIA H200.

The project tests a narrower claim than a complete serving system:

> Given real router traces and measured H200 transfer/compute timings, can a
> predictor trained after the base model reduce exposed expert-fetch stall at a
> fixed memory/byte budget without changing model outputs?

All prediction is advisory. The original router remains authoritative, and a
miss triggers the exact expert-weight fetch. Therefore the proposed mechanism
does not approximate expert selection or alter logits.

## Study stages

1. Validate the pinned runtime and model.
2. Collect prompt-grouped router/hidden-state traces.
3. Fit a frozen lightweight predictor on training prompts only.
4. Compare random, popularity, LRU, router-only, learned, and oracle policies.
5. Measure host-to-device transfer and layer timing on the target H200.
6. Replay held-out traces in a discrete-event simulator.
7. Report paired bootstrap confidence intervals and all negative results.

The preregistered pilot design is in
[`protocol/pilot.yaml`](protocol/pilot.yaml). Large models and caches belong on
the VM data disk and are intentionally excluded from version control.

