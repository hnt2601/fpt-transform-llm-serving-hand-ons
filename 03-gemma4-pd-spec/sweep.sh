#!/usr/bin/env bash
set -euo pipefail

LAB="${1:?can ten lab, vd 01-g12-agg}"
BASE="${2:?can url, vd http://vllm-g4-agg:8000}"
CLIST="${3:-1 2 4 8 16 24 32 48 64 96 128}"

MODEL=gemma4-12b
DATASET=/datasets/speed-bench/throughput_8k.jsonl

nprompts_for() {
  local c=$1
  local n=$(( c * 4 ))
  (( n < 32 ))  && n=32
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
  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --dataset-name speed_bench \
    --dataset-path "$DATASET" \
    --num-prompts "$NPROMPTS" \
    --max-concurrency "$C" \
    --request-rate inf \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-filename "$OUT" \
    2>&1 | grep -E "Successful|Output token throughput|Total Token|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|Median ITL|Median E2EL|P99"
done
echo "=== xong: /results/${LAB}-c*.json ==="
