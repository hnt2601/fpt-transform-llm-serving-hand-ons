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

### Cái giá bằng KV cache

So số đo thật với bài 01:

| | Bài 01 (baseline) | Bài 02 (MTP) | Chênh |
|---|---:|---:|---:|
| `--gpu-memory-utilization` | 0.90 | 0.88 | |
| KV cache | 39.47 GiB | **36.25 GiB** | −3.2 GiB |
| Số token | 1.200.036 | **973.279** | −19% |
| Session @128k | 9.16 | **7.43** | −19% |

MTP không tốn thêm trọng số (nó nằm sẵn trong checkpoint), nhưng vẫn **mất 19% KV cache** cho bộ đệm draft và CUDA graph của nhánh speculative. Đây là chi phí thật cần ghi nhận: bạn đổi **số session phục vụ song song** lấy **tốc độ mỗi session**.

> Nếu Token Factory đang bị giới hạn bởi số session đồng thời chứ không phải tốc độ gõ, MTP có thể là đánh đổi sai. Bảng ở Bước 4 sẽ cho bạn dữ liệu để quyết định.

### Khởi động nhanh hơn hẳn bài 01

```
Bài 01: 7 phút 07 giây   (compile từ đầu)
Bài 02: 3 phút 31 giây   (dùng lại cache)
```

Chênh lệch này đến từ PVC `vllm-cache` dựng ở [bài 00](../00-prerequisites/): kết quả `torch.compile` và FlashInfer autotune được giữ giữa các lần deploy. Phần phải compile lại chỉ là những gì thực sự đổi — nhánh speculative.

### Ba dòng log xác nhận MTP đang chạy

```
Resolved architecture: Qwen3_5MTP
speculative_config=SpeculativeConfig(method='mtp',
                                     model='/models/Qwen/Qwen3.8-27B-FP8',
                                     num_spec_tokens=2)
Overriding draft model max model len from 262144 to 131072
```

Chú ý dòng giữa: `model` trỏ về **chính đường dẫn model target**. Đây là bằng chứng cụ thể cho điều đã nói ở phần 2 — MTP không cần checkpoint rời, vLLM tự dựng draft model từ trọng số đã có (`mtp.safetensors` bên trong thư mục target).

### Một warning bạn SẼ thấy, và nó quan trọng

```
WARNING [speculative.py:1318] Enabling num_speculative_tokens > 1 will run
multiple times of forward on same MTP layer, which may result in lower
acceptance rate
```

> **Đọc kỹ dòng này — nó nói về giới hạn kiến trúc, không phải lỗi cấu hình.**
>
> Qwen3.8 chỉ có **một** lớp MTP. Khi bạn đặt `num_speculative_tokens = 2`, vLLM **chạy đi chạy lại cùng một lớp đó** để sinh token thứ hai, thay vì dùng một lớp riêng được huấn luyện cho vị trí t+2.
>
> Hệ quả: token thứ hai được đoán bởi một lớp **không hề được huấn luyện cho vị trí đó**, nên xác suất chấp nhận thấp hơn hẳn token thứ nhất. Đây chính là lý do Bước 5 bắt bạn đo cả `num_speculative_tokens = 1` — rất có thể giá trị 1 lại cho throughput tốt hơn 2, vì token thứ hai gần như luôn bị từ chối mà vẫn tốn compute verify.
>
> Đối chiếu với bài 03: DSpark có backbone riêng 5 lớp được huấn luyện để dự đoán **cả block 8 token**, nên không gặp giới hạn này. Đó là phần lớn lý do acceptance length của nó cao gấp đôi.

Bạn cũng có thể thấy warning về multimodal processor — Qwen3.8 là vision-language model và draft MTP kế thừa đường multimodal của target; warning này vô hại với workload text thuần.

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
- Ở **concurrency 64**, batch đã lớn, GPU không còn dư compute. Mỗi draft token bị từ chối giờ là **lãng phí compute thật**.

Quy luật này áp dụng cho **mọi** phương pháp speculative decoding, kể cả DSpark ở bài 03 — nên đừng kỳ vọng con số tăng tốc ở concurrency 1 lặp lại ở concurrency 64.

### Điều bất ngờ nhất trong số liệu: acceptance length KHÔNG đổi

Đây là chỗ nhiều người hiểu sai. Hãy nhìn hai cột này cạnh nhau (số đo thật):

| Concurrency | Acceptance length | Tăng tốc TPOT thực đo |
|---|---:|---:|
| 1 | 2.29 | **1.74×** |
| 8 | 2.18 | **1.36×** |
| 32 | 2.20 | **1.22×** |

**Acceptance length gần như là hằng số.** Nó phải như vậy: acceptance đo xem drafter đoán có trúng không — mà điều đó phụ thuộc vào **model và dữ liệu**, hoàn toàn không phụ thuộc bạn đang chạy 1 hay 32 request song song.

Vậy tại sao tăng tốc lại teo từ 1.74× xuống 1.22×?

> **Vì lợi ích của speculative decoding không nằm ở việc đoán trúng, mà ở việc COMPUTE ĐANG RẢNH.**
>
> Ở concurrency 1, GPU chỉ đang chờ bộ nhớ; verify thêm 2 token là miễn phí. Ở concurrency 32, GPU đã bận thật, và 2 token verify thêm giờ phải **cạnh tranh** với công việc của 31 request khác.
>
> Nói cách khác: bạn không mất khả năng đoán, bạn mất **chỗ trống để tận dụng việc đoán đó**.

Hệ quả vận hành rất cụ thể: **đừng dùng acceptance length để dự đoán tăng tốc ở production.** Acceptance length tốt chỉ nói rằng drafter phù hợp với dữ liệu của bạn. Tăng tốc thực tế còn phụ thuộc mức tải — và phải đo ở đúng mức tải bạn định chạy.

### Ba điều nữa số liệu cho thấy

**1. Tăng tốc thực luôn thấp hơn acceptance length.** Ở c1: acceptance 2.29 nhưng chỉ tăng tốc 1.74× — hiệu suất **76%**. Phần hụt là chi phí chạy MTP head và verify 3 token thay vì 1. Tỉ lệ này là thước đo "drafter có đủ rẻ không".

**2. TTFT cũng cải thiện, dù về lý thuyết không nên.** Số đo ở c8: TTFT p50 từ 2506 xuống **624 ms**. Speculative decoding chỉ tác động lên decode, nhưng decode xong nhanh hơn nghĩa là scheduler có nhiều chỗ cho prefill hơn → hàng đợi ngắn lại. Đây là lợi ích gián tiếp, chỉ xuất hiện khi hệ thống đang có hàng đợi.

**3. ITL p99 xấu đi — mặt trái phải biết.** Ở c8: từ 18.5 lên **268 ms**. Khi draft bị từ chối, bước đó tốn thời gian gấp bội. TPOT trung vị đẹp hơn nhưng **nhịp gõ kém đều hơn**.

> Với agentic coding, đây là đánh đổi thường chấp nhận được: người dùng thấy code hiện ra nhanh hơn rõ rệt, và thỉnh thoảng khựng một nhịp. Nhưng nếu SLA của bạn ràng buộc p99 ITL, hãy cân nhắc.

**Bài học vận hành:** speculative decoding không phải cờ "bật rồi quên". Ở giờ cao điểm của Token Factory lợi ích thu hẹp. Bài 03 sẽ giới thiệu **adaptive verification** — cơ chế tự điều chỉnh theo tải, chính là để xử lý vấn đề này.

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
