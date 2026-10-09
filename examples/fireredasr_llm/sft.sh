#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    cat <<'USAGE'
Usage: bash sft.sh [local_model_dir] [train.jsonl] [extra Swift arguments...]
Or set MODEL_DIR and DATASET. MODEL_DIR must be a local FireRedASR-LLM-L directory.
Set FIRERED_ASR_REPO to the prepared FireRedASR source tree.
Optional: VAL_DATASET, OUTPUT_DIR, CUDA_VISIBLE_DEVICES, NPROC_PER_NODE,
TRAIN_BS, EVAL_BS, GRAD_ACC, LEARNING_RATE, NUM_TRAIN_EPOCHS, MAX_LENGTH,
TORCH_DTYPE, ATTN_IMPL, TUNER_TYPE, SPLIT_DATASET_RATIO, PLUGIN_PATH, DRY_RUN=1.
See README.md for source/model preparation and a four-step smoke run.
USAGE
    exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
model=${MODEL_DIR:-}
dataset=${DATASET:-}
if [[ $# -gt 0 && $1 != -* ]]; then model=$1; shift; fi
if [[ $# -gt 0 && $1 != -* ]]; then dataset=$1; shift; fi
if [[ -z $model || -z $dataset ]]; then
    printf '%s\n' 'Set MODEL_DIR and DATASET, or pass local_model_dir and train.jsonl. Use --help for usage.' >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export FIRERED_ASR_DATASET_PATH=$dataset
export FIRERED_ASR_VAL_DATASET_PATH=${VAL_DATASET:-}

command=(sft)
if [[ ${DRY_RUN:-0} == 1 ]]; then command+=(--dry-run); fi
validation=(--split_dataset_ratio "${SPLIT_DATASET_RATIO:-0.01}")
if [[ -n ${VAL_DATASET:-} ]]; then
    validation=(--val_dataset firered_asr_jsonl_val --split_dataset_ratio 0)
fi

exec uv run --project "$REPO_ROOT" --locked audio-swift "${command[@]}" \
    --external_plugins "${PLUGIN_PATH:-${SCRIPT_DIR}/plugin.py}" \
    --model "$model" \
    --model_type fireredasr_llm \
    --template fireredasr_llm \
    --dataset firered_asr_jsonl \
    "${validation[@]}" \
    --output_dir "${OUTPUT_DIR:-${REPO_ROOT}/output/fireredasr_llm}" \
    --tuner_type "${TUNER_TYPE:-lora_llm}" \
    --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
    --attn_impl "${ATTN_IMPL:-eager}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "${TRAIN_BS:-1}" \
    --per_device_eval_batch_size "${EVAL_BS:-1}" \
    --gradient_accumulation_steps "${GRAD_ACC:-16}" \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --vit_lr "${VIT_LR:-1e-5}" \
    --aligner_lr "${ALIGNER_LR:-1e-5}" \
    --lora_rank "${LORA_RANK:-8}" \
    --lora_alpha "${LORA_ALPHA:-32}" \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_length "${MAX_LENGTH:-2048}" \
    --truncation_strategy delete \
    --padding_free false \
    --packing false \
    --lazy_tokenize true \
    --warmup_ratio 0.05 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --dataset_num_proc 1 \
    --dataloader_num_workers 0 \
    --dataloader_persistent_workers false \
    --report_to none \
    "$@"
