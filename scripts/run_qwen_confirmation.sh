#!/usr/bin/env bash
set -euo pipefail

cd /root/dataDisk/specstream
source scripts/remote_env.sh

model="models/Qwen3.6-35B-A3B"
traces="artifacts/traces/qwen36_pilot_v4"
checkpoint="artifacts/checkpoints/qwen36_next_preregistered_e32_w256.pt"

# ModelScope exposes partially allocated shard files while downloading.  Start
# only after every expected shard has its final name.
while true; do
    final_shards="$(find "${model}" -maxdepth 1 \
        -name 'model-*-of-00026.safetensors' | wc -l)"
    partial_shards="$(find "${model}" -maxdepth 1 \
        -name '*.incomplete' | wc -l)"
    if [[ "${final_shards}" -eq 26 && "${partial_shards}" -eq 0 ]]; then
        break
    fi
    sleep 5
done

# Keep timing measurements isolated from the OLMoE robustness queue.
while pgrep -f \
    'train_predictor.py|evaluate_predictor.py|benchmark_predictor.py|benchmark_compiled_predictor.py' \
    >/dev/null; do
    sleep 2
done

python scripts/audit.py \
    --model "${model}" \
    --output artifacts/audit/qwen36_h200.json

python scripts/benchmark_model.py \
    --model "${model}" \
    --output artifacts/timing/qwen36_model_h200.json \
    --warmup-tokens 8 \
    --measurement-tokens 32

python scripts/collect_decode_traces.py \
    --model "${model}" \
    --corpus artifacts/corpus/pilot.jsonl \
    --output-dir "${traces}" \
    --maximum-prompt-tokens 1024 \
    --new-tokens 96 \
    --seed 260724787 \
    --limit 120 \
    --resume \
    --empty-cache-every 8

python scripts/benchmark_predictor.py \
    --output artifacts/timing/qwen36_predictor_e32_h200.json \
    --hidden-size 2048 \
    --num-layers 40 \
    --num-experts 256 \
    --widths 256 \
    --architecture layer_aware \
    --layer-embedding-width 32 \
    --model-dtype float32 \
    --source-dtype bfloat16 \
    --warmup 200 \
    --iterations 2000

predictor_latency_ms="$(
    python -c 'import json; print(json.load(open(
        "artifacts/timing/qwen36_predictor_e32_h200.json"
    ))["results"][0]["p50_ms"])'
)"

python scripts/build_deadline_profile.py \
    --model "${model}" \
    --transfer-timing artifacts/timing/h2d_h200.json \
    --model-timing artifacts/timing/qwen36_model_h200.json \
    --output artifacts/profiles/qwen36_next_layer_budget8_e32.json \
    --feature-key router_features \
    --target-horizon 1 \
    --byte-budget-expert-equivalents 8 \
    --predictor-latency-ms "${predictor_latency_ms}"

python scripts/train_predictor.py \
    --traces "${traces}" \
    --output "${checkpoint}" \
    --epochs 8 \
    --batch-size 4096 \
    --width 256 \
    --architecture layer_aware \
    --layer-embedding-width 32 \
    --loss bce \
    --learning-rate 0.002 \
    --maximum-train-pairs-per-prompt 4096 \
    --feature-key router_features \
    --target-horizon 1 \
    --deadline-profile artifacts/profiles/qwen36_next_layer_budget8_e32.json \
    --seed 260724787

python scripts/evaluate_predictor.py \
    --traces "${traces}" \
    --checkpoint "${checkpoint}" \
    --output artifacts/results/qwen36_next_preregistered_e32_predictor.json \
    --budgets 2 4 8 12 16 \
    --batch-size 8192 \
    --bootstrap-resamples 2000 \
    --seed 260724787

python scripts/simulate_prefetch.py \
    --traces "${traces}" \
    --checkpoint "${checkpoint}" \
    --model "${model}" \
    --transfer-timing artifacts/timing/h2d_h200.json \
    --model-timing artifacts/timing/qwen36_model_h200.json \
    --output artifacts/results/qwen36_next_preregistered_e32_stall.json \
    --byte-budget-expert-equivalents 0 2 4 8 12 16 \
    --bootstrap-resamples 2000 \
    --seed 260724787

for layer in 0 20 39; do
    python scripts/benchmark_staged_overlap.py \
        --model "${model}" \
        --output "artifacts/timing/qwen36_staged_overlap_layer${layer}_h200.json" \
        --layer "${layer}" \
        --experts 8 \
        --warmup 50 \
        --iterations 500
done

python scripts/simulate_cache.py \
    --traces "${traces}" \
    --checkpoint "${checkpoint}" \
    --model "${model}" \
    --transfer-timing artifacts/timing/h2d_h200.json \
    --model-timing artifacts/timing/qwen36_model_h200.json \
    --output artifacts/results/qwen36_next_preregistered_e32_cache.json \
    --byte-budget-expert-equivalents 0 2 4 8 12 16 \
    --cache-experts-per-layer 8 16 32 64 \
    --bootstrap-resamples 2000 \
    --seed 260724787

bash scripts/run_qwen_kernel_postcheck.sh
