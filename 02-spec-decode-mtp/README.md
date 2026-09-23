# Bài 02 — Tối ưu 1: Speculative Decoding với MTP

## Giới thiệu

Ở bài 01 bạn đã chạm trần: TPOT ~10 ms ở concurrency 1, và không tham số scheduler nào phá được. Bài này phá trần đó bằng **speculative decoding**, dùng **MTP head có sẵn trong Qwen3.8**.

Thay đổi so với bài 01 chỉ là **một cờ duy nhất**. Nhưng để chỉnh nó đúng, bạn cần hiểu cơ chế.

## Mục lục

1. [Vì sao speculative decoding lại hiệu quả](#1-vì-sao-speculative-decoding-lại-hiệu-quả)
2. [MTP là gì](#2-mtp-là-gì)
3. [Bước 1: Deploy](#bước-1-deploy)
4. [Bước 2: Kiểm tra acceptance](#bước-2-kiểm-tra-acceptance)
5. [Bước 3: Benchmark](#bước-3-benchmark)
6. [Bước 4: Đọc kết quả](#bước-4-đọc-kết-quả)
7. [Bước 5: Tinh chỉnh num_speculative_tokens](#bước-5-tinh-chỉnh-num_speculative_tokens)
8. [Dọn dẹp](#dọn-dẹp)

## 1. Vì sao speculative decoding lại hiệu quả

Quay lại con số vật lý:

```
Trọng số Qwen3.8-27B-FP8       ≈ 27.5 GB
Băng thông HBM của H100        ≈ 3.35 TB/s
                                 ─────────────
Thời gian tối thiểu đọc hết
trọng số 1 lần                 ≈ 8.2 ms
```

Mỗi bước decode tuần tự phải đọc **toàn bộ** 27.5 GB đó từ HBM, chỉ để sinh ra **1 token duy nhất**. GPU gần như rảnh rỗi về mặt compute — nó chỉ đang chờ bộ nhớ.

Bây giờ chú ý: nếu bạn đưa vào **4 token ứng viên cùng lúc** và cho model kiểm tra tất cả trong một lần forward, bạn vẫn chỉ đọc 27.5 GB đó **một lần**. Chi phí compute tăng gấp 4 nhưng compute vốn đang thừa. Đây là bữa trưa gần như miễn phí.

```
Decode tuần tự:   [đọc 27.5GB] → 1 token   [đọc 27.5GB] → 1 token   [đọc 27.5GB] → 1 token
                  ◀────8.2ms───▶           ◀────8.2ms───▶           ◀────8.2ms───▶

Speculative:      [draft rẻ] [đọc 27.5GB, verify 4 ứng viên] → chấp nhận 3 token
                  ◀─0.5ms─▶  ◀──────────8.5ms──────────────▶
                  → 3 token trong 9ms thay vì 24.6ms
```

**Chỉ số quan trọng nhất: acceptance length (τ)** — trung bình số token được chấp nhận mỗi bước. Tăng tốc lý thuyết ≈ τ, trừ đi chi phí draft.

**Và đây là lý do agentic coding là workload đáng để tối ưu:** phần sinh code rất dễ đoán. Sau `for i in ` thì `range(` gần như chắc chắn. Sau `def foo(self,` thì newline + 8 dấu cách. Theo model card của speculator DSpark cho chính Qwen3.8-27B, acceptance length trên HumanEval đạt **4.20** so với **3.61** khi viết văn xuôi.

> Bài 03 sẽ chỉ ra một sắc thái quan trọng mà nhận định này bỏ sót: **tool call lại là loại khó đoán nhất** (3.57, thấp nhất bảng). Agentic coding là hỗn hợp của cả hai.

## 2. MTP là gì

**MTP (Multi-Token Prediction)** là các head phụ được huấn luyện **ngay trong quá trình pretrain model gốc**, dự đoán token thứ t+2, t+3, ... song song với head chính dự đoán t+1.

Model card của Qwen3.8-27B ghi rõ:

> `MTP (Multi-Token Prediction): trained with multiple steps`

Hệ quả thực tiễn rất quan trọng:

| | MTP | Draft model rời (EAGLE/DSpark) |
|---|---|---|
| Cần checkpoint phụ? | **Không** — nằm sẵn trong trọng số target | Có, phải tải thêm ~4 GB |
| Tốn thêm VRAM? | Không đáng kể | ~4 GB |
| Cần huấn luyện? | Không — Qwen đã làm khi pretrain | Có (hoặc dùng checkpoint cộng đồng) |
| Acceptance length điển hình | ~2.0–2.5 | ~3.6–4.2 (xem bài 03) |
| Công sức triển khai | **1 cờ** | Tải checkpoint + tinh chỉnh |

MTP là **lựa chọn có tỉ lệ lợi ích/công sức cao nhất**. Luôn thử MTP trước khi nghĩ đến drafter rời.

vLLM tự nhận diện: với `model_type` là `qwen3_5` (kiến trúc của Qwen3.8), `SpeculativeConfig` tự dựng một draft ModelConfig trỏ về **chính đường dẫn model target** và đổi architecture thành `Qwen3_5MTP`. Vì vậy bạn **không cần khai báo `"model"`** trong `--speculative-config`.

> **Chạy mọi lệnh từ thư mục `02-spec-decode-mtp/`** — các manifest được tham chiếu bằng đường dẫn tương đối:
>
> ```bash
> cd 02-spec-decode-mtp
> ```

## Bước 1: Deploy

Đảm bảo bài 01 đã được xoá:

```bash
kubectl get pods -n token-factory    # không còn pod vllm-agg
kubectl apply -f deployment.yaml
# Chờ container khởi động RỒI mới theo dõi log.
# `kubectl logs -f` chạy ngay sau `apply` sẽ báo:
#   Error from server (BadRequest): container "vllm" ...
#   is waiting to start: ContainerCreating
kubectl wait --for=jsonpath='{.status.phase}'=Running \
  pod -l app=vllm-mtp -n token-factory --timeout=300s
kubectl logs -f deploy/vllm-mtp -n token-factory
```

### Điểm khác biệt duy nhất so với bài 01

```yaml
- --speculative-config={"method":"mtp","num_speculative_tokens":2}
```

Giải thích từng khoá:

| Khoá | Giá trị | Ý nghĩa |
|---|---|---|
| `method` | `mtp` | Dùng MTP head tích hợp sẵn. Không cần `"model"` — vLLM tự trỏ về target |
| `num_speculative_tokens` | `2` | Đề xuất 2 token mỗi bước. Bắt đầu từ 2, bài này sẽ hướng dẫn tinh chỉnh |

Ngoài ra `--gpu-memory-utilization` giảm nhẹ xuống `0.88`: bộ verify cần thêm bộ đệm cho draft token và CUDA graph của nhánh speculative.

> **Nếu gặp lỗi.** Với một số bản vLLM, `num_speculative_tokens > 1` trên dòng Qwen3.5/3.8 có thể báo lỗi tuỳ số MTP step mà checkpoint có. Nếu vậy hãy hạ về `1`, xác nhận chạy được, rồi mới tăng dần. Bạn cũng có thể thấy warning về multimodal processor — Qwen3.8 là vision-language model và draft MTP kế thừa đường multimodal của target; warning này vô hại với workload text thuần.

## Bước 2: Kiểm tra acceptance

Đây là bước **quan trọng nhất** của bài. Trước khi benchmark, phải xác nhận speculative decoding thực sự đang hoạt động — chứ không phải đang bật nhưng bị từ chối 100%.

Sinh một ít tải cho có số liệu:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- \
  curl -s http://vllm-mtp:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b",
       "messages":[{"role":"user","content":"Viết một class LRUCache đầy đủ trong Python, có docstring."}],
       "max_tokens":800}' > /dev/null
```

Đọc metric speculative:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- \
  curl -s http://vllm-mtp:8000/metrics | grep -E "spec_decode"
```

Bạn sẽ thấy các counter dạng:

```
vllm:spec_decode_num_drafts_total          ...
vllm:spec_decode_num_draft_tokens_total    ...
vllm:spec_decode_num_accepted_tokens_total ...
```

Tính hai chỉ số:

```
Acceptance rate   = num_accepted_tokens_total / num_draft_tokens_total
Acceptance length = 1 + (num_accepted_tokens_total / num_drafts_total)
```

(Cộng 1 vì mỗi bước verify luôn sinh chắc chắn ít nhất 1 token "bonus" từ chính target.)

| Acceptance length | Kết luận |
|---|---|
| < 1.3 | Hỏng. Kiểm tra lại cờ, hoặc `num_speculative_tokens` quá lớn |
| 1.5 – 2.0 | Bình thường cho MTP với `num_speculative_tokens=2` |
| > 2.0 | Rất tốt — workload code đang phát huy tác dụng |

> **Đây là thói quen phải hình thành:** với speculative decoding, benchmark mà không nhìn acceptance length thì bạn đang mò trong bóng tối. Throughput giảm có thể do draft quá tham (acceptance thấp) hoặc do draft quá đắt — hai nguyên nhân khác nhau, hai cách sửa khác nhau.

## Bước 3: Benchmark

**Cùng một lệnh, cùng seed, cùng dataset như bài 01** — chỉ đổi URL và tên file kết quả:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- bash

for C in 1 8 32 64; do
  vllm bench serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --base-url http://vllm-mtp:8000 \
    --model qwen3.8-27b \
    --tokenizer /models/Qwen/Qwen3.8-27B-FP8 \
    --dataset-name speed_bench \
    --dataset-path /datasets/speed-bench \
    --speed-bench-dataset-subset throughput_8k \
    --speed-bench-output-len 1000 \
    --num-prompts $(( C * 8 )) \
    --max-concurrency ${C} \
    --request-rate inf \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --seed 42 \
    --save-result \
    --result-dir /results \
    --result-filename "02-mtp-c${C}.json" \
    --label "02-mtp-c${C}"
done
```

## Bước 4: Đọc kết quả

| Concurrency | TPOT p50 bài 01 | TPOT p50 bài 02 | Tăng tốc | Output tok/s bài 01 | Output tok/s bài 02 |
|---|---|---|---|---|---|
| 1 | | | | | |
| 8 | | | | | |
| 32 | | | | | |
| 64 | | | | | |

### Quy luật bạn sẽ quan sát được

**Tăng tốc lớn nhất ở concurrency thấp, giảm dần khi concurrency tăng.** Đây không phải lỗi cấu hình — đây là bản chất:

- Ở **concurrency 1**, GPU cực kỳ dư compute. Verify 3 token thay vì 1 gần như miễn phí → tăng tốc gần bằng acceptance length.
- Ở **concurrency 64**, batch đã lớn, GPU không còn dư compute. Mỗi draft token bị từ chối giờ là **lãng phí compute thật**. Tăng tốc co lại, và ở batch đủ lớn speculative decoding thậm chí có thể **chậm hơn** baseline.

Quy luật này áp dụng cho **mọi** phương pháp speculative decoding, kể cả DSpark ở bài 03 — nên đừng kỳ vọng con số tăng tốc ở concurrency 1 lặp lại ở concurrency 64.

**Bài học vận hành:** speculative decoding không phải cờ "bật rồi quên". Ở giờ cao điểm của Token Factory (concurrency cao) lợi ích thu hẹp. Bài 03 sẽ giới thiệu **adaptive verification** — cơ chế tự điều chỉnh theo tải, chính là để xử lý vấn đề này.

## Bước 5: Tinh chỉnh `num_speculative_tokens`

Thử 3 giá trị và đo cả acceptance lẫn throughput:

```bash
# Sửa giá trị trong deployment rồi rollout lại
kubectl set env deploy/vllm-mtp -n token-factory --list   # xem cấu hình hiện tại
```

Cách nhanh hơn: sửa `deployment.yaml`, đổi `num_speculative_tokens`, rồi:

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deploy/vllm-mtp -n token-factory
```

Ghi lại bảng này (chỉ cần chạy ở concurrency 1 và 32):

| `num_speculative_tokens` | Acceptance length | TPOT p50 @c1 | Output tok/s @c32 |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

**Đánh đổi bạn đang đo:**

- Tăng `num_speculative_tokens` → mỗi bước có thể chấp nhận nhiều token hơn → **acceptance length tăng**.
- Nhưng token thứ k chỉ được chấp nhận nếu cả k−1 token trước đó đều được chấp nhận → xác suất giảm theo cấp số nhân → **acceptance rate giảm**.
- Và mỗi draft token bị từ chối vẫn tốn compute verify.

Có một điểm tối ưu, thường là 2–3 cho MTP. **Điểm tối ưu này phụ thuộc workload và mức concurrency của bạn** — đó là lý do bài này bắt bạn tự đo thay vì cho sẵn con số.

## Dọn dẹp

```bash
kubectl delete -f deployment.yaml
kubectl wait --for=delete pod -l app=vllm-mtp -n token-factory --timeout=300s
```

---

**Tiếp theo:** [Bài 03 — Speculative decoding với DSpark](../03-spec-decode-dspark/)
