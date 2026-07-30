#!/usr/bin/env bash
set -euo pipefail

cd /root/dataDisk/specstream
while [[ ! -f artifacts/results/olmoe_next_low_rank_kl_stall.json ]]; do
    sleep 2
done

for seed in 260724788 260724789; do
    .venv/bin/python scripts/train_predictor.py \
        --traces artifacts/traces/olmoe_decode_v4 \
        --output "artifacts/checkpoints/olmoe_next_uniform_w256_seed${seed}.pt" \
        --deadline-profile artifacts/profiles/olmoe_next_layer_budget8.json \
        --feature-key router_features --target-horizon 1 \
        --architecture layer_aware --loss bce \
        --epochs 8 --batch-size 4096 --width 256 --seed "$seed"
    .venv/bin/python scripts/evaluate_predictor.py \
        --traces artifacts/traces/olmoe_decode_v4 \
        --checkpoint "artifacts/checkpoints/olmoe_next_uniform_w256_seed${seed}.pt" \
        --output "artifacts/results/olmoe_next_uniform_w256_seed${seed}.json" \
        --budgets 2 4 8 12 16 --bootstrap-resamples 2000 \
        --seed 260724787 \
        > "logs/eval_seed${seed}.log"
done
