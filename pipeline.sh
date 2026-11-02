#!/bin/sh
# Every stage skips itself when its result exists, so after a dead session or a sleeping Mac this just runs again:
#   nohup caffeinate -i sh pipeline.sh > logs/pipeline.log 2>&1 &
set -e
mkdir -p logs
stage() { echo "=== $1 $(date +%T)"; }

stage data
uv run python -u generate_data.py > logs/generate_data3.log 2>&1

stage baselines
[ -f results_eval_zero_shot.json ] || uv run python -u eval_ollama.py zero_shot > logs/eval_zero_shot.log 2>&1
[ -f results_eval_few_shot.json ] || uv run python -u eval_ollama.py few_shot --shots 3 > logs/eval_few_shot.log 2>&1

stage lora
[ -f results_train_sft.json ] || uv run python -u train_sft.py > logs/train_sft.log 2>&1
[ -f results_eval_lora.json ] || uv run python -u eval_mlx.py lora --adapter adapters/sft > logs/eval_lora.log 2>&1

stage dpo
[ -f results_train_dpo.json ] || uv run python -u train_dpo.py > logs/train_dpo.log 2>&1
[ -f results_eval_lora_dpo.json ] || uv run python -u eval_mlx.py lora_dpo --adapter adapters/dpo > logs/eval_lora_dpo.log 2>&1

stage done
