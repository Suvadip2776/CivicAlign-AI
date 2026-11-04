#!/bin/bash
# Run every capability benchmark for one Qwen configuration.
#   ./run_config.sh <name> <inspect model> <constitution true|false> <max-connections>
# Same seeded sample (--sample-shuffle 42 + --limit) in every configuration, so all
# four see identical questions. Graders: GPT-4o via OpenRouter (never see the
# constitution; it is a solver step on the evaluated model only).
NAME=$1; MODEL=$2; CONST=$3; CONN=$4
cd "$(dirname "$0")"
export OPENROUTER_API_KEY=$(grep '^OPENROUTER_API_KEY=' ../.env | cut -d= -f2-)
export HF_TOKEN=${HF_TOKEN:-$(hf auth token 2>/dev/null | tail -1)}
: "${NYCCNT_BASE_URL:?set NYCCNT_BASE_URL to the vLLM endpoint serving the OCT model (.../v1)}"
export NYCCNT_API_KEY=${NYCCNT_API_KEY:-$HF_TOKEN}
LOG=logs/$NAME; mkdir -p $LOG
# benchmark  limit  extra-args
BENCH=(
  "mmlu_pro 300"
  "gpqa_diamond 198 --epochs 1"
  "math 300"
  "simpleqa 300"
  "mask 300"
  "sycophancy 300"
  "truthfulqa 300"
  "bbq 300"
  "legalbench_subset 320"
  "confaide_tier2a 97"
  "confaide_tier3 270"
  "xstest_safe 250"
  "xstest_unsafe 200"
)
for b in "${BENCH[@]}"; do
  set -- $b; T=$1; L=$2; shift 2
  [ -n "$(ls $LOG/*_${T//_/-}_*.eval 2>/dev/null)" ] && { echo "$(date +%T) $NAME $T already done"; continue; }
  echo "$(date +%T) $NAME $T start"
  .venv/bin/inspect eval evals.py@$T -T constitution=$CONST --model "$MODEL" \
      --model-role grader=openrouter/openai/gpt-4o --limit $L --sample-shuffle 42 \
      --max-tokens 8192 --max-connections $CONN --log-dir $LOG --display none \
      --no-fail-on-error --retry-on-error 2 "$@" > $LOG/$T.out 2>&1
  echo "$(date +%T) $NAME $T exit $?"
done
echo "$(date +%T) $NAME ALL DONE"
