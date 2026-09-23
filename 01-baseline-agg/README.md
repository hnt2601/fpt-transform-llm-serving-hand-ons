# Bài 01 — Baseline: vLLM Aggregated Mode

## Giới thiệu

"Aggregated mode" (agg mode) là cách vLLM chạy mặc định: **một tiến trình duy nhất làm cả prefill lẫn decode**, dùng chung một KV cache pool và một scheduler.

Đây là baseline của chúng ta. Mọi con số ở bài 02, 03, 04 đều so với bài này. Vì vậy bài này có hai việc quan trọng ngang nhau:

1. Deploy cho chạy được.
2. **Đo cho đúng** — và hiểu con số đang nói gì.

## Mục lục

1. [Kiến trúc agg mode](#1-kiến-trúc-agg-mode)
2. [Ngân sách bộ nhớ trên H100 80GB](#2-ngân-sách-bộ-nhớ-trên-h100-80gb)
3. [Bước 1: Deploy](#bước-1-deploy)
4. [Bước 2: Kiểm tra](#bước-2-kiểm-tra)
5. [Bước 3: Benchmark](#bước-3-benchmark)
6. [Bước 4: Đọc kết quả](#bước-4-đọc-kết-quả)
7. [Bước 5: Quan sát điểm đau](#bước-5-quan-sát-điểm-đau)
8. [Dọn dẹp](#dọn-dẹp)

## 1. Kiến trúc agg mode

```
                 ┌──────────────────────────────────────┐
  request  ───▶  │  vLLM engine (1 process, 1 GPU)      │
                 │                                       │
                 │   scheduler                           │
                 │      ├── prefill batch  (compute-bound)│
                 │      └── decode  batch  (memory-bound) │
                 │              ↑ cùng tranh 1 GPU ↑      │
                 │                                       │
                 │   KV cache pool (paged, dùng chung)   │
                 └──────────────────────────────────────┘
```

**Ưu điểm:** đơn giản, KV cache không phải đi đâu cả, prefix caching hoạt động tự nhiên.

**Nhược điểm — và đây là lý do tồn tại của bài 04:** prefill và decode có bản chất trái ngược nhau nhưng bị nhốt chung một chỗ.

| | Prefill | Decode |
|---|---|---|
| Bị chặn bởi | Compute (FLOPs) | Memory bandwidth |
| Batch lý tưởng | Nhỏ, token dài | Lớn, 1 token/seq |
| Khi chạy chung | Một prefill 100k token sẽ **chiếm GPU và làm nghẽn toàn bộ decode đang chạy** |

Hiện tượng đó gọi là **decode interference**. Chunked prefill (bật mặc định) giảm nhẹ nó bằng cách cắt prefill thành từng mẩu, nhưng không loại bỏ được.

## 2. Ngân sách bộ nhớ trên H100 80GB

Đây là phép tính bạn phải làm được trước mọi lần deploy:

```
Tổng HBM                                  80.0 GB
- CUDA context + activation + graphs   ≈   4.0 GB
- Trọng số Qwen3.8-27B-FP8             ≈  27.5 GB
                                         ─────────
  Còn lại cho KV cache                 ≈  48.5 GB
```

Với `--gpu-memory-utilization 0.90` → vLLM được dùng 72 GB → KV cache ≈ **40 GB**.

### Vì sao 128k context lại khả thi trên 1 GPU

Chuỗi bài đặt `--max-model-len 131072` (128k), dù model hỗ trợ native tới 262k. Con số này vừa với 1× H100 nhờ **kiến trúc lai** của Qwen3.8:

```
Hidden Layout: 16 × ( 3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN) )
                     └── linear attention ──┘      └── attention thường ──┘
                         state CỐ ĐỊNH               KV cache TĂNG theo độ dài
```

Chỉ **16 trong 64 lớp** là Gated Attention có KV cache tăng theo độ dài. 48 lớp còn lại là Gated DeltaNet — linear attention với state kích thước cố định, **không phình theo context**. Ước lượng thô cho mỗi token:

```
16 lớp × 2 (K và V) × 4 KV head × 256 head_dim × 1 byte (fp8)  ≈  32 KB/token
```

Với ~40 GB KV cache: **≈ 1.3 triệu token**, tức khoảng **10 session ở full 128k**, hoặc **hàng trăm session** ở độ dài thực tế của agentic coding (~10k token/lượt).

> **Đây là lý do một model 27B lại phục vụ được 128k context trên một GPU duy nhất.** Với một dense transformer 27B thông thường (toàn bộ 64 lớp đều có attention KV), cùng ngân sách 40 GB chỉ chứa được khoảng 1/4 số token đó. Kiến trúc lai là thứ khiến bài toán này khả thi.

**Con số ước lượng ở trên chỉ để bạn hiểu độ lớn — hãy lấy số thật từ log của vLLM** (Bước 2). Nó quyết định bạn phục vụ song song được bao nhiêu session:

```
số session đồng thời ≈ (GPU KV cache size) / (độ dài trung bình mỗi session)
```

### Vì sao TP1

Toàn bộ chuỗi bài dùng `--tensor-parallel-size 1`: **một engine = một GPU**.

| | TP1 (bài này dùng) | TP2+ |
|---|---|---|
| Trọng số | Nạp đủ trên 1 GPU | Chia đôi qua 2 GPU |
| Giao tiếp mỗi lớp | Không có | All-reduce qua NVLink/PCIe |
| Khi nào cần | Model **vừa** 1 GPU | Model **không vừa**, hoặc cần giảm TPOT bằng mọi giá |

Qwen3.8-27B ở FP8 chỉ tốn 27.5 GB — **thừa sức vừa một H100 80GB**, nên TP2 chỉ thêm chi phí all-reduce mà không giải quyết vấn đề gì. Quan trọng hơn cho mục đích học tập: giữ TP1 xuyên suốt nghĩa là **mọi so sánh giữa 4 bài đều trên cùng một đơn vị phần cứng**. Bài 04 dùng 2 GPU là vì nó chạy **2 engine**, không phải vì tăng TP — đó là hai cách dùng GPU thứ hai hoàn toàn khác nhau, và phân biệt được chúng chính là một phần của bài học.

## Bước 1: Deploy

```bash
kubectl apply -f deployment.yaml
```

Theo dõi khởi động (lần đầu mất 3–8 phút do torch.compile):

```bash
kubectl logs -f deploy/vllm-agg -n token-factory
```

### Giải thích các cờ quan trọng trong `deployment.yaml`

| Cờ | Giá trị | Vì sao |
|---|---|---|
| `--served-model-name` | `qwen3.8-27b` | Tên ngắn, cố định qua cả 4 bài → script benchmark không phải sửa |
| `--max-model-len` | `131072` | 128k — đủ cho phiên agentic coding dài; cố định để so sánh công bằng giữa 4 bài |
| `--tensor-parallel-size` | `1` | Model vừa 1 GPU → TP2 chỉ thêm chi phí all-reduce |
| `--kv-cache-dtype fp8` | | Gấp đôi số token KV chứa được — quan trọng vì agentic coding có context dài |
| `--enable-prefix-caching` | | Agent gửi lại cùng system prompt + tool schema mỗi lượt → tiết kiệm prefill rất lớn |
| `--max-num-batched-tokens 8192` | | Chunked prefill: cắt prefill dài thành mẩu 8k để decode không bị treo quá lâu |
| `--max-num-seqs 64` | | Trần concurrency; khớp với mức cao nhất trong sweep benchmark |
| `--gpu-memory-utilization 0.90` | | Để lại ~8 GB cho CUDA context và phân mảnh |
| `--reasoning-parser qwen3` | | Qwen3.8 bật thinking mặc định → tách `reasoning_content` khỏi `content` |
| `--tool-call-parser qwen3_coder` | | Bắt buộc cho agentic coding: parse tool call thành JSON chuẩn OpenAI |
| `--enable-auto-tool-choice` | | Cho phép model tự quyết định gọi tool |

> **Vì sao dùng FP8 thay vì BF16?** Qwen3.8-27B ở BF16 tốn ~54 GB — vẫn nằm vừa H100 80GB, nhưng chỉ còn ~20 GB cho KV cache, và quan trọng hơn: decode là memory-bound, đọc 54 GB trọng số mỗi bước thay vì 27.5 GB nghĩa là **chậm gần gấp đôi**. FP8 trên H100 có Tensor Core hỗ trợ native, chất lượng gần như không đổi.

## Bước 2: Kiểm tra

```bash
kubectl wait --for=condition=ready pod -l app=vllm-agg -n token-factory --timeout=900s
kubectl get svc vllm-agg -n token-factory
```

Test một request thật:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- \
  curl -s http://vllm-agg:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role":"user","content":"Viết hàm Python đảo ngược linked list."}],
    "max_tokens": 200
  }' | python3 -m json.tool
```

Lấy ngân sách KV cache thực tế mà vLLM đã cấp phát:

```bash
kubectl logs deploy/vllm-agg -n token-factory | grep -i "KV cache size"
```

Kết quả mong đợi có dạng:

```
GPU KV cache size: 1,2xx,xxx tokens
Maximum concurrency for 131,072 tokens per request: xx.xx
```

Dòng thứ hai là số session **ở full 128k** — với agentic coding thực tế (~10k token/lượt) con số phục vụ được sẽ cao hơn nhiều. Ghi lại cả hai dòng.

## Bước 3: Benchmark

Vào bench client:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- bash
```

Chạy sweep 4 mức concurrency:

```bash
for C in 1 8 32 64; do
  vllm bench serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --base-url http://vllm-agg:8000 \
    --model qwen3.8-27b \
    --tokenizer /models/Qwen/Qwen3.8-27B-FP8 \
    --dataset-name random \
    --random-prefix-len 2048 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --random-range-ratio 0.2 \
    --num-prompts $(( C * 8 )) \
    --max-concurrency ${C} \
    --request-rate inf \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --seed 42 \
    --save-result \
    --result-dir /results \
    --result-filename "01-agg-c${C}.json" \
    --label "01-agg-c${C}"
done
```

### Giải thích lựa chọn tham số benchmark

| Tham số | Vì sao chọn như vậy |
|---|---|
| `--random-prefix-len 2048` | Mô phỏng system prompt + tool schema **dùng chung** giữa các request → kích hoạt prefix caching, đúng như agent thật |
| `--random-input-len 8000` | Context file + lịch sử hội thoại của một phiên agentic coding điển hình |
| `--random-output-len 1000` | Agent sinh code + reasoning, không phải chatbot trả lời 1 câu |
| `--random-range-ratio 0.2` | Cho độ dài dao động ±20% → tránh batch "đều tăm tắp" giả tạo |
| `--ignore-eos` | **Bắt buộc để so sánh công bằng.** Ép mọi request sinh đủ 1000 token, nếu không mỗi cấu hình sẽ dừng ở độ dài khác nhau và throughput không so được |
| `--max-concurrency` + `--request-rate inf` | Đo ở trạng thái bão hoà có kiểm soát: luôn có đúng C request trong hệ thống |
| `--num-prompts = C * 8` | Đủ request để mỗi luồng chạy ~8 lượt, đủ ổn định thống kê mà không quá lâu |
| `--seed 42` | Cố định qua cả 4 bài → cùng một bộ prompt |

> **Bẫy thường gặp:** nếu bỏ `--ignore-eos`, bài 02/03 (speculative decoding) có thể trông "nhanh hơn" chỉ vì output ngắn hơn. Luôn giữ cờ này khi so sánh.

## Bước 4: Đọc kết quả

Output của `vllm bench serve` có dạng:

```
============ Serving Benchmark Result ============
Successful requests:                     256
Benchmark duration (s):                  ...
Total input tokens:                      ...
Total generated tokens:                  ...
Request throughput (req/s):              ...
Output token throughput (tok/s):         ...
Total Token throughput (tok/s):          ...
---------------Time to First Token----------------
Mean TTFT (ms):                          ...
Median TTFT (ms):                        ...
P99 TTFT (ms):                           ...
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          ...
Median TPOT (ms):                        ...
P99 TPOT (ms):                           ...
---------------Inter-token Latency----------------
Mean ITL (ms):                           ...
Median ITL (ms):                         ...
P99 ITL (ms):                            ...
==================================================
```

Điền vào bảng baseline (copy vào [99-compare-results/](../99-compare-results/)):

| Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | Output tok/s |
|---|---|---|---|---|---|
| 1 | | | | | |
| 8 | | | | | |
| 32 | | | | | |
| 64 | | | | | |

### Ba điều phải nhận ra từ bảng này

1. **Ở concurrency 1, TPOT p50 chính là "tốc độ gõ code" mà người dùng cảm nhận.** Trên H100, model 27B FP8 decode tuần tự thường cho TPOT khoảng 9–12 ms → ~85–110 token/s. Đây là **trần vật lý của decode tuần tự**, và không có cách nào vượt qua bằng cách chỉnh tham số batch. Chỉ speculative decoding mới phá được trần này — đó là bài 02 và 03.

2. **TPOT tăng dần theo concurrency.** Batch lớn hơn → mỗi bước decode phải xử lý nhiều seq hơn → mỗi token chậm hơn, nhưng tổng throughput cao hơn. Đây là đánh đổi cơ bản latency ⇄ throughput.

3. **TTFT p99 tăng vọt nhanh hơn TTFT p50.** Đó là dấu vết của decode interference: một request không may rơi đúng lúc hệ thống đang prefill 8k token của request khác.

## Bước 5: Quan sát điểm đau

Chạy một test riêng để **nhìn thấy** interference. Ở một terminal, sinh tải prefill nặng:

```bash
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://vllm-agg:8000 --model qwen3.8-27b \
  --tokenizer /models/Qwen/Qwen3.8-27B-FP8 \
  --dataset-name random --random-input-len 30000 --random-output-len 50 \
  --num-prompts 64 --max-concurrency 16 --request-rate inf --ignore-eos
```

Ở terminal thứ hai (mở thêm một `kubectl exec`), đo trải nghiệm single-user cùng lúc:

```bash
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://vllm-agg:8000 --model qwen3.8-27b \
  --tokenizer /models/Qwen/Qwen3.8-27B-FP8 \
  --dataset-name random --random-input-len 1000 --random-output-len 500 \
  --num-prompts 20 --max-concurrency 1 --request-rate inf --ignore-eos \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99
```

So sánh TPOT/ITL p99 của lần chạy thứ hai với lần chạy concurrency=1 ở Bước 3. Con số phình ra chính là **cái giá của việc nhốt prefill và decode chung một GPU**. Ghi lại — bài 04 sẽ quay lại đúng phép đo này.

## Dọn dẹp

Chúng ta chỉ có 1 GPU, nên **phải xoá trước khi sang bài 02**:

```bash
kubectl delete -f deployment.yaml
kubectl wait --for=delete pod -l app=vllm-agg -n token-factory --timeout=300s
```

Kiểm tra GPU đã được giải phóng:

```bash
kubectl get pods -n token-factory
```

---

**Tiếp theo:** [Bài 02 — Speculative decoding với MTP](../02-spec-decode-mtp/)
