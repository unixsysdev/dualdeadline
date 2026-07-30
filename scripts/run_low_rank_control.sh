#!/usr/bin/env bash
set -euo pipefail

cd /root/dataDisk/specstream
while [[ ! -f artifacts/traces/olmoe_decode_v4/manifest.json ]]; do
    sleep 2
done

.venv/bin/python scripts/benchmark_predictor.py \
    --output artifacts/timing/olmoe_low_rank_widths_h200.json \
    --hidden-size 2048 --num-layers 16 --num-experts 64 \
    --architecture low_rank --widths 64 128 256 \
    --warmup 100 --iterations 1000

predictor_latency_ms=$(
    .venv/bin/python -c \
        'import json; d=json.load(open("artifacts/timing/olmoe_low_rank_widths_h200.json")); print(next(x["p50_ms"] for x in d["results"] if x["width"] == 256))'
)
.venv/bin/python scripts/build_deadline_profile.py \
    --model models/OLMoE-1B-7B-0924-Instruct \
    --transfer-timing artifacts/timing/h2d_h200.json \
    --model-timing artifacts/timing/olmoe_model_h200.json \
    --output artifacts/profiles/olmoe_next_low_rank_budget8.json \
    --feature-key router_features --target-horizon 1 \
    --byte-budget-expert-equivalents 8 \
    --predictor-latency-ms "$predictor_latency_ms"

for loss in kl bce; do
    .venv/bin/python scripts/train_predictor.py \
        --traces artifacts/traces/olmoe_decode_v4 \
        --output "artifacts/checkpoints/olmoe_next_low_rank_${loss}_w256.pt" \
        --deadline-profile artifacts/profiles/olmoe_next_low_rank_budget8.json \
        --feature-key router_features --target-horizon 1 \
        --architecture low_rank --loss "$loss" \
        --epochs 8 --batch-size 4096 --width 256
    .venv/bin/python scripts/evaluate_predictor.py \
        --traces artifacts/traces/olmoe_decode_v4 \
        --checkpoint "artifacts/checkpoints/olmoe_next_low_rank_${loss}_w256.pt" \
        --output "artifacts/results/olmoe_next_low_rank_${loss}_w256_predictor.json" \
        --budgets 2 4 8 12 16 --bootstrap-resamples 2000 \
        > "logs/eval_low_rank_${loss}.log"
done

.venv/bin/python scripts/simulate_prefetch.py \
    --traces artifacts/traces/olmoe_decode_v4 \
    --checkpoint artifacts/checkpoints/olmoe_next_low_rank_kl_w256.pt \
    --model models/OLMoE-1B-7B-0924-Instruct \
    --transfer-timing artifacts/timing/h2d_h200.json \
    --model-timing artifacts/timing/olmoe_model_h200.json \
    --output artifacts/results/olmoe_next_low_rank_kl_stall.json \
    > logs/sim_low_rank_kl.log
