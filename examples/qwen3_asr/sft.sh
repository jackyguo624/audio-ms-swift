#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    cat <<'USAGE'
Usage: bash sft.sh [model_id_or_path] [train.jsonl] [extra Swift arguments...]
Or set MODEL_DIR and DATASET; MODEL_DIR defaults to Qwen/Qwen3-ASR-1.7B.
Optional: VAL_DATASET, OUTPUT_DIR, CUDA_VISIBLE_DEVICES, NPROC_PER_NODE,
TRAIN_BS, EVAL_BS, GRAD_ACC, LEARNING_RATE, NUM_TRAIN_EPOCHS, MAX_LENGTH,
TORCH_DTYPE, ATTN_IMPL, TUNER_TYPE, SPLIT_DATASET_RATIO, DRY_RUN=1.
See README.md for data preparation and a four-step smoke run.
USAGE
    exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
model=${MODEL_DIR:-Qwen/Qwen3-ASR-1.7B}
dataset=${DATASET:-}
if [[ $# -gt 0 && $1 != -* ]]; then model=$1; shift; fi
if [[ $# -gt 0 && $1 != -* ]]; then dataset=$1; shift; fi
if [[ -z $dataset ]]; then
    printf '%s\n' 'Set DATASET or pass model_id_or_path and train.jsonl. Use --help for usage.' >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}

command=(sft)
if [[ ${DRY_RUN:-0} == 1 ]]; then command+=(--dry-run); fi
validation=(--split_dataset_ratio "${SPLIT_DATASET_RATIO:-0.01}")
if [[ -n ${VAL_DATASET:-} ]]; then
    validation=(--val_dataset "$VAL_DATASET" --split_dataset_ratio 0)
fi

exec uv run --project "$REPO_ROOT" --locked --extra qwen3-asr audio-swift "${command[@]}" \
    --model "$model" \
    --model_type qwen3_asr \
    --template qwen3_asr \
    --dataset "$dataset" \
    "${validation[@]}" \
    --output_dir "${OUTPUT_DIR:-${REPO_ROOT}/output/qwen3_asr}" \
    --tuner_type "${TUNER_TYPE:-lora}" \
    --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
    --attn_impl "${ATTN_IMPL:-sdpa}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "${TRAIN_BS:-1}" \
    --per_device_eval_batch_size "${EVAL_BS:-1}" \
    --gradient_accumulation_steps "${GRAD_ACC:-16}" \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --lora_rank "${LORA_RANK:-8}" \
    --lora_alpha "${LORA_ALPHA:-32}" \
    --target_modules all-linear \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --vit_gradient_checkpointing false \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_length "${MAX_LENGTH:-2048}" \
    --truncation_strategy delete \
    --warmup_ratio 0.05 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --dataset_num_proc 1 \
    --dataloader_num_workers 0 \
    --report_to none \
    "$@"
