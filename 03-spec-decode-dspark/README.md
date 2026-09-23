# Bài 03 — Tối ưu 2: Speculative Decoding với DSpark

## Giới thiệu

MTP ở bài 02 cho bạn acceptance length khoảng 1.5–2.0 với công sức gần như bằng 0. Bài này đi xa hơn: **DSpark**, một drafter chuyên dụng đạt acceptance length **3.4–4.0 trên workload code**, cộng thêm **adaptive verification** — cơ chế tự điều chỉnh độ sâu speculation theo tải, đúng thứ mà bài 02 đã chỉ ra là còn thiếu.

## Mục lục

1. [DSpark khác MTP ở đâu](#1-dspark-khác-mtp-ở-đâu)
2. [Adaptive verification](#2-adaptive-verification)
3. [Checkpoint speculator](#3-checkpoint-speculator)
4. [Bước 1: Deploy](#bước-1-deploy)
5. [Bước 2: Kiểm tra acceptance](#bước-2-kiểm-tra-acceptance)
6. [Bước 3: Benchmark](#bước-3-benchmark)
7. [Bước 4: Đọc kết quả](#bước-4-đọc-kết-quả)
8. [Bước 5: Đo riêng workload code](#bước-5-đo-riêng-workload-code)
9. [Bước 6: Kiểm chứng ở context dài](#bước-6-kiểm-chứng-ở-context-dài)
10. [Xử lý sự cố](#xử-lý-sự-cố)
11. [Dọn dẹp](#dọn-dẹp)

## 1. DSpark khác MTP ở đâu

Cả EAGLE và MTP đều **draft tuần tự**: muốn có 4 token ứng viên thì phải chạy drafter 4 lần liên tiếp, lần sau phụ thuộc lần trước. Drafter tuy nhỏ nhưng 4 lần chạy tuần tự vẫn tốn thời gian thật — và thời gian đó nằm trên đường găng của mỗi bước decode.

**DSpark draft song song.** Toàn bộ block 7 token được đề xuất trong **một lần chạy duy nhất**:

```
EAGLE / MTP (tuần tự):
  [draft t+1] → [draft t+2] → [draft t+3] → [draft t+4] → [target verify]
  ◀─────────── 4 lượt tuần tự, mỗi lượt phụ thuộc lượt trước ──────────▶

DSpark (song song):
  [draft cả block 7 token trong 1 lượt] → [target verify 8 token]
  ◀──────── 1 lượt ────────▶
```

Cơ chế bên trong, đọc trực tiếp từ `config.json` của checkpoint `RedHatAI/Qwen3.8-27B-speculator.dspark-preview`:

| Thành phần | Chi tiết |
|---|---|
| Architecture | `DSparkDraftModel`, định dạng thư viện [speculators](https://github.com/vllm-project/speculators) — **native vLLM** |
| Backbone | 5 lớp `sliding_attention` kiểu Qwen3, hidden size 5120, 20 query head / 4 KV head, head_dim 256, sliding window 2048. Trọng số BF16 ~4.0 GB |
| Đầu vào | Hidden state phụ từ **8 lớp** của target: 4, 12, 20, 28, 36, 44, 52, 60 — drafter "nhìn" trực tiếp vào trạng thái ẩn của target thay vì đoán mù |
| `block_size` | **8** — đề xuất tối đa 8 token mỗi bước decode |
| Markov head | `vanilla`, rank 256 — hiệu chỉnh mỗi vị trí theo token ngay trước nó, bù cho việc draft song song không có quan hệ nhân quả |
| **Confidence head** | **Bật**, có Markov feature — dự đoán **xác suất được chấp nhận của từng vị trí** trong block |
| Verifier | `Qwen/Qwen3.8-27B` (`Qwen3_5ForConditionalGeneration`) |
| Proposal method mặc định | `greedy`, `speculative_tokens: 8` |

> **Confidence head là chi tiết quan trọng nhất trong bảng trên.** DSpark không chỉ đề xuất 8 token — nó còn **tự ước lượng token nào có khả năng được chấp nhận**. Đây chính là cơ sở kỹ thuật của adaptive verification ở phần sau: engine biết được lúc nào nên verify cả block và lúc nào nên dừng sớm. DFlash (tiền thân của DSpark) không có phần này.
>
> Số liệu validation của chính checkpoint cho thấy confidence head hoạt động tốt: sai số tuyệt đối khi dự đoán acceptance chỉ **0.168**, độ lệch tích luỹ **−0.013** (gần như không thiên lệch).

### Acceptance length công bố (model card `RedHatAI/Qwen3.8-27B-speculator.dspark-preview`)

Đo bằng **vLLM 0.29.0** với verifier `Qwen/Qwen3.8-27B`, 8 speculative token:

| Dataset | Acceptance length | Pos 0 | Pos 3 | Pos 7 |
|---|---:|---:|---:|---:|
| math_reasoning | **5.70** | 89.7% | 61.1% | 32.5% |
| rag | 4.68 | 83.4% | 45.5% | 21.7% |
| translation | 4.28 | 81.3% | 38.9% | 17.0% |
| **HumanEval (code)** | **4.20** | 80.8% | 37.8% | 16.8% |
| qa | 3.98 | 77.3% | 35.4% | 13.9% |
| summarization | 3.70 | 79.6% | 31.6% | 7.4% |
| writing | 3.61 | 76.5% | 29.0% | 9.9% |
| question | 3.59 | 76.7% | 28.8% | 9.6% |
| **tool_call** | **3.57** | 76.5% | 28.4% | 9.1% |

Hãy đọc bảng này theo **hai chiều**, đây là bài học chính của phần 1:

**Chiều dọc — workload nào dễ đoán.** Sinh code (HumanEval 4.20) cao hơn viết văn (3.61) và trả lời câu hỏi (3.59), đúng như lập luận ở bài 02. Nhưng chú ý điều bất ngờ: **`tool_call` thấp nhất bảng (3.57)**.

> **Điều chỉnh lại nhận định ở bài 02.** "Code dễ đoán" là đúng với *thân hàm* — cú pháp lặp, indent, tên biến quen thuộc. Nhưng **tool call thì không**: đó là JSON chứa tham số cụ thể (đường dẫn file, tên hàm, chuỗi tìm kiếm) mà drafter không thể suy ra từ ngữ cảnh. Agentic coding là **hỗn hợp của cả hai**: phần sinh code được lợi nhiều, phần sinh tool call được lợi ít.
>
> Hệ quả thực tiễn cho Token Factory: acceptance length bạn đo được sẽ nằm đâu đó **giữa 3.6 và 4.2**, tuỳ agent của bạn gọi tool nhiều hay ít. Một agent kiểu "đọc file → sửa file → chạy test" (nhiều tool call, ít code) sẽ được lợi ít hơn một agent kiểu "sinh nguyên module".

**Chiều ngang — acceptance suy giảm theo vị trí.** Với HumanEval: vị trí 0 đạt 80.8% nhưng vị trí 7 chỉ còn 16.8%. Token thứ k chỉ được chấp nhận nếu **cả k−1 token trước đó** đều được chấp nhận, nên xác suất giảm theo cấp số nhân.

Đây chính là lý do `block_size = 8` chứ không phải 30: các vị trí sau đóng góp quá ít để bù chi phí verify. Và cũng là lý do adaptive verification tồn tại — khi tải cao, verify tới vị trí 7 với 16.8% cơ hội là lãng phí compute.

### Acceptance ở context dài — điểm mạnh riêng cho agentic coding

Đo trên MRCR 8-Needle, 224 request:

| Độ dài prompt | Mean acceptance length |
|---|---:|
| 4K–8K | 4.75 |
| 8K–16K | 4.04 |
| 16K–32K | 4.63 |
| 32K–65K | 4.48 |
| 65K–131K | 4.35 |
| 131K–262K | 4.84 |
| 1M+ | 4.30 |
| **Tổng thể** | **4.49** |

**Acceptance length gần như không suy giảm khi context dài ra** — từ 8K tới hơn 1 triệu token vẫn dao động quanh 4.3–4.8. Với agentic coding, nơi context phình lên theo từng lượt hội thoại, đây là tính chất rất đáng giá: lợi ích của speculative decoding **không bốc hơi** khi phiên làm việc kéo dài.

## 2. Adaptive verification

Thay vì chọn cứng một độ sâu speculation cho cả deployment, adaptive verification để engine **quyết định theo từng bước** sẽ verify bao nhiêu phần của block draft — và nó làm được điều đó nhờ **confidence head** đã nói ở phần 1: drafter tự báo xác suất chấp nhận của từng vị trí.

```
Tải thấp,  drafter tự tin  → verify cả 7 token   → tăng tốc tối đa
Tải cao,   drafter do dự   → verify 2-3 token    → không lãng phí compute
```

Nhờ vậy lợi ích được giữ trên dải concurrency rộng mà **không cần tự tay tinh chỉnh cho từng mức tải** — khác hẳn việc dò `num_speculative_tokens` bằng tay ở bài 02 Bước 5. Đây là tính năng khiến DSpark dùng được trong production thật, nơi tải thay đổi theo giờ.

Bật bằng khoá `"enable_adaptive_verification": true`.

## 3. Checkpoint speculator

Bài 00 đã tải sẵn về PVC. Xác nhận:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- \
  ls -la /models/Qwen3.8-27B-speculator.dspark
```

Phải thấy đúng **3 file**:

```
config.json          2.1 KB
config.py            2.3 KB
model.safetensors    3.98 GB
```

| File | Vai trò |
|---|---|
| `model.safetensors` | Trọng số drafter, BF16 |
| `config.json` | `architectures: ["DSparkDraftModel"]`, `block_size: 8`, `aux_hidden_state_layer_ids`, cấu hình Markov + confidence head |
| `config.py` | Định nghĩa `DSparkSpeculatorConfig`, được nạp qua `auto_map` → **bắt buộc phải có `--trust-remote-code`** |

> Nếu bạn thấy thêm `optimizer_state_dict.pt` (4.23 GB), nghĩa là job ở bài 00 đã tải cả repo. File đó là state của Muon optimizer dùng để **tiếp tục huấn luyện**, vLLM không bao giờ đọc tới. Xoá được an toàn.

**Ngân sách bộ nhớ cập nhật:**

```
Tổng HBM                                  80.0 GB
- CUDA context + activation + graphs   ≈   4.5 GB
- Trọng số target FP8                  ≈  27.5 GB
- Trọng số speculator (BF16, 5 lớp)    ≈   4.0 GB
                                         ─────────
  Còn lại cho KV cache                 ≈  44.0 GB
```

Speculator tốn thêm 4.0 GB — chấp nhận được, nhưng ta hạ `--gpu-memory-utilization` xuống `0.86` để chừa chỗ cho CUDA graph của cả hai nhánh. KV cache còn ~33 GB (≈1.05 triệu token), vẫn thừa sức cho `--max-model-len 131072`.

Speculator chạy cùng GPU với target ở **TP1** — nó chỉ 5 lớp nên không có lý do gì phải chia qua nhiều GPU.

## Bước 1: Deploy

```bash
kubectl get pods -n token-factory     # xác nhận bài 02 đã xoá
kubectl apply -f deployment.yaml
kubectl logs -f deploy/vllm-dspark -n token-factory
```

### Khối `--speculative-config` của bài này

Model card của checkpoint đưa ra lệnh tối giản sau — ta dùng đúng nó làm cấu hình cơ sở:

```json
{
  "method": "dspark",
  "model": "/models/Qwen3.8-27B-speculator.dspark",
  "num_speculative_tokens": 8
}
```

| Khoá | Giá trị | Ý nghĩa |
|---|---|---|
| `method` | `dspark` | Chọn thuật toán draft song song DSpark |
| `model` | đường dẫn PVC | **Bắt buộc** — khác MTP, DSpark cần checkpoint speculator rời |
| `num_speculative_tokens` | `8` | Khớp đúng `block_size: 8` trong `config.json` của checkpoint |

> **`num_speculative_tokens` phải khớp `block_size`.** DSpark sinh cả block trong một lần chạy; đặt giá trị nhỏ hơn 8 là vứt bỏ phần draft đã tính xong mà vẫn trả tiền compute cho nó. Nếu bạn đổi sang checkpoint DSpark khác, **đọc `block_size` trong `config.json` của nó** thay vì đoán.

### Tuỳ chọn: bật adaptive verification

Model card không bật sẵn. Sau khi đã xác nhận cấu hình cơ sở chạy được ở Bước 2, hãy thử thêm:

```json
{
  "method": "dspark",
  "model": "/models/Qwen3.8-27B-speculator.dspark",
  "num_speculative_tokens": 8,
  "enable_adaptive_verification": true
}
```

Rồi đo lại và so sánh — **đặc biệt ở concurrency 32 và 64**, nơi ta đã dự đoán ở bài 02 rằng lợi ích sẽ teo lại. Đây là một thí nghiệm có đối chứng đúng nghĩa: cùng checkpoint, cùng mọi thứ, chỉ khác một cờ.

`deployment.yaml` để cấu hình cơ sở ở dạng đang dùng và cấu hình adaptive ở dạng comment ngay bên dưới — đổi bằng cách bỏ comment.

> **Thứ tự này là cố ý.** Đừng bật cả hai thứ mới cùng lúc rồi không biết cái nào có tác dụng. Chạy cơ sở → ghi số → thêm một biến → đo lại. Đây là cách duy nhất để kết luận có giá trị.

## Bước 2: Kiểm tra acceptance

Giống bài 02 — nhưng lần này kỳ vọng cao hơn hẳn:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- \
  curl -s http://vllm-dspark:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b",
       "messages":[{"role":"user","content":"Viết một HTTP server bất đồng bộ bằng Python asyncio, có xử lý lỗi và graceful shutdown."}],
       "max_tokens":1200}' > /dev/null

kubectl exec -it deploy/bench-client -n token-factory -- \
  curl -s http://vllm-dspark:8000/metrics | grep -E "spec_decode"
```

```
Acceptance length = 1 + (num_accepted_tokens_total / num_drafts_total)
```

| Acceptance length | Kết luận |
|---|---|
| < 2.0 | Có vấn đề — xem [Xử lý sự cố](#xử-lý-sự-cố) |
| 2.0 – 3.0 | Chạy đúng nhưng thấp hơn kỳ vọng. Thường do target FP8 (xem ghi chú bên dưới) |
| **3.5 – 4.2** | Đúng kỳ vọng: model card công bố 4.20 trên HumanEval, 3.57 trên tool_call |
| > 4.5 | Rất tốt — nhiều khả năng workload của bạn thiên về sinh code hơn là gọi tool |

> **Ghi chú quan trọng về target FP8.** Checkpoint này được huấn luyện và đánh giá với verifier **`Qwen/Qwen3.8-27B` bản BF16 gốc**, trong khi chuỗi bài của chúng ta phục vụ bản **FP8**. Speculator đọc hidden state từ 8 lớp của target; lượng tử hoá FP8 làm các hidden state đó lệch đi một chút, nên acceptance length của bạn **có thể thấp hơn con số công bố**.
>
> Đây là một đánh đổi thật mà bạn nên đo, không nên đoán. Nếu acceptance length rơi xuống dưới 3.0, hãy thử chạy lại với target BF16 (`Qwen/Qwen3.8-27B`) để tách bạch nguyên nhân — nhưng nhớ rằng BF16 tốn ~54 GB trọng số, nên phải hạ `--max-model-len` đáng kể. Kết quả của phép thử này chính là câu trả lời cho câu hỏi: *FP8 tiết kiệm băng thông, nhưng có làm speculative decoding kém đi đủ nhiều để mất lợi không?*

So sánh trực tiếp với con số MTP bạn ghi ở bài 02. Chênh lệch này là toàn bộ lý do tồn tại của một drafter chuyên dụng.

## Bước 3: Benchmark

Vẫn **cùng lệnh, cùng seed**:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- bash

for C in 1 8 32 64; do
  vllm bench serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --base-url http://vllm-dspark:8000 \
    --model qwen3.8-27b \
    --tokenizer /models/Qwen3.8-27B-FP8 \
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
    --result-filename "03-dspark-c${C}.json" \
    --label "03-dspark-c${C}"
done
```

## Bước 4: Đọc kết quả

| Concurrency | TPOT p50: 01 agg | 02 MTP | 03 DSpark | Tăng tốc DSpark vs baseline |
|---|---|---|---|---|
| 1 | | | | |
| 8 | | | | |
| 32 | | | | |
| 64 | | | | |

Và quan trọng không kém — bảng acceptance:

| Cấu hình | Acceptance length | Tăng tốc TPOT @c1 |
|---|---|---|
| 02 MTP (`num_spec=2`) | | |
| 03 DSpark (`num_spec=7`, adaptive) | | |

### Câu hỏi phải tự trả lời được

1. **Tỉ lệ tăng tốc TPOT có bám sát acceptance length không?** Nếu acceptance length là 3.5 nhưng TPOT chỉ nhanh 2.2× thì phần chênh lệch đi đâu? (Gợi ý: chi phí chạy backbone 5 lớp của speculator, và chi phí verify 8 token thay vì 1.)

2. **Ở concurrency 64, DSpark có còn thắng MTP không?** Bảng per-position ở phần 1 dự báo điều gì sẽ xảy ra: vị trí 7 chỉ có 16.8% cơ hội được chấp nhận, nên ở batch lớn phần lớn draft token cuối block là compute vứt đi. Nếu DSpark tụt xuống gần hoặc dưới baseline, đó chính là lúc **adaptive verification** (mục tuỳ chọn ở Bước 1) cần được bật — hãy chạy lại với nó và so.

3. **TTFT có thay đổi không?** Về lý thuyết là **không** — speculative decoding chỉ tác động lên giai đoạn decode. Nếu TTFT của bạn xấu đi rõ rệt, nguyên nhân thường là KV cache bị co lại (do speculator chiếm ~4 GB VRAM) khiến preemption tăng. Kiểm tra log tìm `preempted`.

## Bước 5: Đo riêng workload code

Dataset `random` dùng token ngẫu nhiên — **không có cấu trúc để drafter đoán**. Nghĩa là acceptance length bạn đo ở trên là **cận dưới bi quan**. Workload thật sẽ tốt hơn.

Chạy thêm một phép đo với `spec_bench` (dataset chuyên dùng để đánh giá speculative decoding, có sẵn trong `vllm bench serve`):

```bash
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://vllm-dspark:8000 \
  --model qwen3.8-27b \
  --tokenizer /models/Qwen3.8-27B-FP8 \
  --dataset-name spec_bench \
  --num-prompts 200 \
  --max-concurrency 8 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --seed 42 \
  --save-result --result-dir /results \
  --result-filename "03-dspark-specbench.json" --label "03-dspark-specbench"
```

Đọc lại `/metrics` và so acceptance length với lần chạy `random`. **Khoảng chênh giữa hai con số chính là giá trị mà cấu trúc của văn bản thật mang lại** — và là lý do bạn không nên chỉ tin dataset `random` khi ra quyết định về speculative decoding.

> Để đo sát thực tế Token Factory nhất, hãy thay bằng chính log request của bạn: `--dataset-name custom --dataset-path /results/my-agent-traces.jsonl`.
>
> **Đây là phép đo đáng tin nhất trong cả bài.** Model card cho thấy khoảng cách lớn giữa `HumanEval` (4.20) và `tool_call` (3.57); tỉ lệ pha trộn hai loại này trong agent của bạn quyết định con số thực tế. Không dataset công khai nào thay thế được trace thật của chính bạn.

## Bước 6: Kiểm chứng ở context dài

Bảng ở phần 1 công bố một tính chất bất thường: acceptance length **không suy giảm** khi context dài ra (4.3–4.8 từ 8K tới hơn 1M token). Chuỗi bài đặt `--max-model-len 131072` nên bạn kiểm chứng được điều này trực tiếp.

Chạy 3 mức độ dài prompt, giữ nguyên mọi thứ khác:

```bash
for IN in 8000 32000 100000; do
  vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --base-url http://vllm-dspark:8000 --model qwen3.8-27b \
    --tokenizer /models/Qwen3.8-27B-FP8 \
    --dataset-name random \
    --random-prefix-len 2048 \
    --random-input-len ${IN} \
    --random-output-len 500 \
    --random-range-ratio 0.1 \
    --num-prompts 32 --max-concurrency 4 --request-rate inf \
    --ignore-eos --percentile-metrics ttft,tpot,itl \
    --metric-percentiles 50,95,99 --seed 42 \
    --save-result --result-dir /results \
    --result-filename "03-dspark-len${IN}.json" --label "03-dspark-len${IN}"

  echo "=== acceptance sau lần chạy input=${IN} ==="
  curl -s http://vllm-dspark:8000/metrics | grep -E "spec_decode"
done
```

> **Lưu ý khi đọc:** các counter `spec_decode_*` là **cộng dồn** từ lúc engine khởi động, không reset giữa các lần chạy. Muốn acceptance length riêng của từng mức, hãy lấy **hiệu số** giữa hai lần đọc liên tiếp:
>
> ```
> acceptance length của đoạn = 1 + (Δ num_accepted_tokens / Δ num_drafts)
> ```

Điền bảng:

| Input length | TTFT p50 (ms) | TPOT p50 (ms) | Acceptance length (tính theo Δ) |
|---|---|---|---|
| 8k | | | |
| 32k | | | |
| 100k | | | |

**Hai thứ phải tách bạch được trong bảng này:**

1. **TTFT tăng gần tuyến tính theo độ dài input** — đó là prefill, và speculative decoding không giúp gì cho nó. Đây chính là vấn đề mà bài 04 (PD disaggregation) nhắm tới.

2. **Acceptance length gần như không đổi** — nếu đúng như model card, lợi ích của DSpark **không bốc hơi** khi phiên agent kéo dài. Đây là tính chất quyết định với agentic coding: context phình lên theo từng lượt hội thoại, và bạn cần biết liệu khoản đầu tư 4 GB VRAM có còn sinh lời ở lượt thứ 50 hay không.

Nếu acceptance length **giảm rõ rệt** ở 100k, nguyên nhân khả dĩ là target FP8 (xem ghi chú ở Bước 2) hoặc prefix caching đang làm nhiễu phép đo — hãy ghi lại đúng những gì bạn quan sát được.

## Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `unknown speculative method: dspark` | Image vLLM quá cũ | Cập nhật image. Model card đánh giá bằng vLLM `0.29.0` — hãy dùng bản này trở lên |
| OOM khi khởi động | Speculator + target vượt ngân sách | Hạ `--gpu-memory-utilization` xuống 0.84, hoặc hạ `--max-model-len` |
| `Unrecognized configuration class` / lỗi nạp config | Thiếu `--trust-remote-code` | `config.json` dùng `auto_map` trỏ tới `config.py`; cờ này là bắt buộc |
| Không tìm thấy trọng số | Job bài 00 chỉ tải 3 file — kiểm tra `model.safetensors` có mặt | `ls -la /models/Qwen3.8-27B-speculator.dspark` |
| Acceptance length < 2.0 | Speculator không khớp target | Checkpoint này chỉ dành cho **Qwen3.8-27B**. Ngoài ra nó được đánh giá với verifier **BF16**, ta đang dùng **FP8** — xem ghi chú ở Bước 2 |
| Lỗi liên quan CUDA graph / attention backend | Adaptive verification yêu cầu `AttentionCGSupport.ALWAYS`; tài liệu vLLM nêu các backend sparse-MLA/sparse-SWA trên SM100 | Trên H100 (SM90): thử bỏ `enable_adaptive_verification`, xác nhận DSpark cơ bản chạy trước, rồi bật lại. Nếu vẫn lỗi, thêm `--enforce-eager` để chẩn đoán (chậm hơn nhưng loại trừ được nguyên nhân CUDA graph) |
| Throughput thấp hơn baseline ở concurrency cao | Đúng như dự đoán khi adaptive verification chưa bật | Kiểm tra lại `enable_adaptive_verification: true` thực sự có hiệu lực trong log |

> **Lưu ý về phần cứng và phiên bản.** Model card đánh giá bằng **vLLM 0.29.0** và validation chạy trên **1× H100** — nghĩa là cấu hình cơ sở của bài này đã được kiểm chứng đúng trên loại GPU bạn đang dùng. Riêng `enable_adaptive_verification` là phần mở rộng: tài liệu vLLM tham chiếu các backend trên SM100, còn H100 là SM90. Vì vậy bài này để nó ở mục **tuỳ chọn**, chạy sau khi cấu hình cơ sở đã xác nhận hoạt động. Nếu adaptive verification chưa chạy trên image của bạn, DSpark cơ bản vẫn là một cải thiện đáng kể so với MTP — hãy ghi lại đúng những gì bạn đo được.

## Dọn dẹp

```bash
kubectl delete -f deployment.yaml
kubectl wait --for=delete pod -l app=vllm-dspark -n token-factory --timeout=300s
```

---

**Tiếp theo:** [Bài 04 — PD Disaggregation + DSpark](../04-pd-disagg-dspark/)
