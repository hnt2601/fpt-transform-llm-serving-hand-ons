#!/usr/bin/env bash
set -euo pipefail

LAB="${1:?can ten lab, vd 01-g12-agg}"
BASE="${2:?can url, vd http://vllm-g4-agg:8000}"
CLIST="${3:-1 2 4 8 16 24 32 48 64 96 128}"

MODEL=gemma4-12b
TOKENIZER=/models/RedHatAI/gemma-4-12B-it-FP8-Dynamic
DATASET_DIR=/datasets/speed-bench
DATASET_SUBSET=throughput_8k

nprompts_for() {
  local c=$1
  local n=$(( c * 2 ))
  (( n < 8 ))   && n=8
  (( n > 256 )) && n=256
  echo $n
}

if ps -eo pid,etime,args | grep -q "[b]ench serve"; then
  echo "!! DANG CO benchmark khac chay - dung lai de tranh nhiem ket qua:"
  ps -eo pid,etime,args | grep "[b]ench serve"
  exit 1
fi

for C in $CLIST; do
  OUT="/results/${LAB}-c${C}.json"
  NPROMPTS=$(nprompts_for "$C")
  echo "=========== ${LAB}  c=${C}  n=${NPROMPTS} ==========="
  LOG="/results/${LAB}-c${C}.log"
  if ! vllm bench serve \
      --backend openai-chat \
      --base-url "$BASE" \
      --endpoint /v1/chat/completions \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --dataset-name speed_bench \
      --dataset-path "$DATASET_DIR" \
      --speed-bench-dataset-subset "$DATASET_SUBSET" \
      --num-prompts "$NPROMPTS" \
      --max-concurrency "$C" \
      --request-rate inf \
      --ignore-eos \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-filename "$OUT" \
      > "$LOG" 2>&1; then
    echo "!! LOI o c=${C}. 40 dong cuoi cua ${LOG}:"
    tail -40 "$LOG"
    exit 1
  fi
  grep -E "Successful|Output token throughput|Total Token|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|Median ITL|Median E2EL|P99" "$LOG" || true
done
echo "=== xong: /results/${LAB}-c*.json ==="
