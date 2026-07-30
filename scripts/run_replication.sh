#!/usr/bin/env bash
set -euo pipefail

cd /root/dataDisk/specstream
while [[ ! -f artifacts/traces/olmoe_replication_v4/manifest.json ]]; do
    sleep 2
done

.venv/bin/python scripts/train_predictor.py \
    --traces artifacts/traces/olmoe_replication_v4 \
    --output artifacts/checkpoints/olmoe_replication_next_uniform_w256.pt \
    --deadline-profile artifacts/profiles/olmoe_next_layer_budget8.json \
    --feature-key router_features --target-horizon 1 \
    --architecture layer_aware --loss bce \
    --epochs 8 --batch-size 4096 --width 256 --seed 260724787

.venv/bin/python scripts/evaluate_predictor.py \
    --traces artifacts/traces/olmoe_replication_v4 \
    --checkpoint artifacts/checkpoints/olmoe_replication_next_uniform_w256.pt \
    --output artifacts/results/olmoe_replication_next_uniform_predictor.json \
    --budgets 2 4 8 12 16 --bootstrap-resamples 2000 \
    --seed 260724787 \
    > logs/eval_replication.log

.venv/bin/python scripts/simulate_prefetch.py \
    --traces artifacts/traces/olmoe_replication_v4 \
    --checkpoint artifacts/checkpoints/olmoe_replication_next_uniform_w256.pt \
    --model models/OLMoE-1B-7B-0924-Instruct \
    --transfer-timing artifacts/timing/h2d_h200.json \
    --model-timing artifacts/timing/olmoe_model_h200.json \
    --output artifacts/results/olmoe_replication_next_uniform_stall.json \
    > logs/sim_replication.log
