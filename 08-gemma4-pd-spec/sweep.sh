#!/usr/bin/env bash
# Quet duong cong throughput theo concurrency - PHEP DO QUYET DINH cua Phan 3.
#
# Muc tieu KHONG phai tim throughput cao nhat, ma tim DIEM GAY cua vung
# phang trong duong cong MoE. Doc ket qua nhu sau:
#
#   tok/s tang gan TUYEN TINH tu c1        -> khong co vung phang
#                                          -> gia thuyet SAI, dung tai day,
#                                             khong chay bai 07/08
#   tok/s gan NHU KHONG DOI toi mot nguong -> co vung phang
#                                          -> nguong do la diem can quet
#                                             quanh no o bai 08
#
# Dung: ./sweep.sh <ten-lab> <url-base> [danh-sach-concurrency]
#   ./sweep.sh 06-g4-agg http://vllm-g4-agg:8000
set -euo pipefail

LAB="${1:?can ten lab, vd 06-g4-agg}"
BASE="${2:?can url, vd http://vllm-g4-agg:8000}"
CLIST="${3:-1 2 4 8 16 24 32 48 64 96 128}"

MODEL=gemma4-26b-a4b
DATASET=/datasets/speed-bench/throughput_8k.jsonl

# So prompt PHAI ti le voi concurrency, khong the la hang so:
# - c=1 voi 256 prompt thi mat hang gio
# - c=128 voi 32 prompt thi chua kip dat trang thai on dinh da xong
# Quy tac: 4 lan concurrency, kep trong [32, 256].
nprompts_for() {
  # KHONG gop thanh "local c=$1 n=$((c*4))": trong cung mot lenh local,
  # $c chua duoc gan khi $((c*4)) duoc tinh -> voi set -u se bao
  # "c: unbound variable".
  local c=$1
  local n=$(( c * 4 ))
  (( n < 32 ))  && n=32
  (( n > 256 )) && n=256
  echo $n
}

# CANH BAO: hai tien trinh benchmark chay chong nhau tren cung mot server
# da tung cho ra 315,90 va 495,11 tok/s cho CUNG MOT cau hinh (lech 57%).
# Luon kiem tra truoc khi do.
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
