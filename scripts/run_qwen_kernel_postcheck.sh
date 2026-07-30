#!/usr/bin/env bash
set -euo pipefail

cd /root/dataDisk/specstream
source scripts/remote_env.sh

traces="artifacts/traces/qwen36_pilot_v4"
checkpoint="artifacts/checkpoints/qwen36_next_preregistered_e32_w256.pt"
completion_result="artifacts/results/qwen36_next_preregistered_e32_cache.json"

# The cache replay is the last command in the main Qwen confirmation pipeline.
# Waiting for its atomic output keeps these latency measurements isolated.
while [[ ! -f "${completion_result}" ]]; do
    sleep 5
done
while pgrep -f \
    'simulate_cache.py|benchmark_staged_overlap.py|simulate_prefetch.py|evaluate_predictor.py|train_predictor.py' \
    >/dev/null; do
    sleep 2
done

python scripts/benchmark_precomputed_layer_predictor.py \
    --output artifacts/timing/qwen36_precomputed_layer_predictor_e32_checkpoint_h200.json \
    --checkpoint "${checkpoint}" \
    --traces "${traces}" \
    --hidden-size 2048 \
    --num-layers 40 \
    --num-experts 256 \
    --width 256 \
    --layer-embedding-width 32 \
    --warmup 200 \
    --iterations 2000 \
    --batch-size 8192

python scripts/benchmark_fused_layer_predictor.py \
    --output artifacts/timing/qwen36_fused_layer_predictor_e32_heldout_h200.json \
    --checkpoint "${checkpoint}" \
    --traces "${traces}" \
    --validation-pairs 0 \
    --hidden-size 2048 \
    --num-layers 40 \
    --num-experts 256 \
    --width 256 \
    --layer-embedding-width 32 \
    --warmup 200 \
    --iterations 4000
