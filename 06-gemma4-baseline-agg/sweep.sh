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
# --tokenizer BAT BUOC phai tro toi thu muc CUC BO. Neu bo trong, vllm bench
# lay --model lam ten tokenizer va di tim tren HuggingFace:
#   OSError: gemma4-26b-a4b is not a local folder and is not a valid model
#   identifier listed on 'https://huggingface.co/models'
# ("gemma4-26b-a4b" chi la --served-model-name, khong phai repo id.)
TOKENIZER=/models/google/gemma-4-26B-A4B-it-fp8-dynamic
# SpeedBench doi --dataset-path la THU MUC, khong phai file .jsonl; subset
# chon rieng bang --dataset-subset (mac dinh la "qualitative").
# Tro thang vao file se bao:
#   ValueError: dataset_path ... is not a directory
DATASET_DIR=/datasets/speed-bench
DATASET_SUBSET=throughput_8k

# So prompt PHAI ti le voi concurrency, khong the la hang so:
# - c=1 voi 256 prompt thi mat hang gio
# - c=128 voi 32 prompt thi chua kip dat trang thai on dinh da xong
# Quy tac: 2 lan concurrency, kep trong [8, 256].
#
# He so 2 (khong phai 4) vi --speed-bench-output-len mac dinh la 4096
# token MOI REQUEST. O c=1 voi TPOT ~8ms, mot request mat ~33 giay; 32
# prompt se ngon 17 phut chi cho mot diem do. 8 request x 4096 token van
# cho 32 nghin mau decode - thua du de uoc luong throughput on dinh.
nprompts_for() {
  # KHONG gop thanh "local c=$1 n=$((c*4))": trong cung mot lenh local,
  # $c chua duoc gan khi $((c*4)) duoc tinh -> voi set -u se bao
  # "c: unbound variable".
  local c=$1
  local n=$(( c * 2 ))
  (( n < 8 ))   && n=8
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
  # Ghi TOAN BO output ra file roi moi loc de hien thi. KHONG duoc
  # `... | grep ...` truc tiep: khi lenh loi, grep nuot sach thong bao va
  # Job chi chet lang le voi dung mot dong tieu de - rat kho chan doan.
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
