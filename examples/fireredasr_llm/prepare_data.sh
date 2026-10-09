#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    cat <<'USAGE'
Usage: bash prepare_data.sh [raw.jsonl] [prepared.jsonl] [extra prepare-data arguments...]
Or set INPUT and OUTPUT. Optional: LIMIT.
This example configures FireRedASR-LLM's transcription prompt with no response prefix.
USAGE
    exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
input=${INPUT:-}
output=${OUTPUT:-}
if [[ $# -gt 0 && $1 != -* ]]; then input=$1; shift; fi
if [[ $# -gt 0 && $1 != -* ]]; then output=$1; shift; fi
if [[ -z $input || -z $output ]]; then
    printf '%s\n' 'Set INPUT and OUTPUT, or pass raw.jsonl and prepared.jsonl. Use --help for usage.' >&2
    exit 2
fi
limit=()
if [[ -n ${LIMIT:-} ]]; then limit=(--limit "$LIMIT"); fi

exec uv run --project "$REPO_ROOT" --locked audio-swift prepare-data \
    --input "$input" --output "$output" \
    --prompt '<audio>请转写音频为文字' \
    "${limit[@]}" "$@"
