# Bài 07 — Speculative decoding với Gemma-4 assistant drafter

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Assistant drafter là gì](#2-assistant-drafter-là-gì)
3. [Điều kiện tiên quyết](#3-điều-kiện-tiên-quyết)
4. [Các bước thực hiện](#4-các-bước-thực-hiện)
5. [Kết quả mong đợi](#5-kết-quả-mong-đợi)
6. [Giải thích](#6-giải-thích)

---

## 1. Giới thiệu

Bài này giữ **nguyên** cấu hình bài 06 và chỉ đổi **hai dòng**:

| | Bài 06 | Bài 07 |
|---|---|---|
| `--gpu-memory-utilization` | 0.90 | **0.89** |
| `--speculative-config` | *(không có)* | **có** |

Mọi thứ khác giữ nguyên: TP1, `--max-model-len=131072`, `--kv-cache-dtype=fp8`,
`--max-num-seqs=128`, `--limit-mm-per-prompt={"image":8,"video":0}`.
Giữ đối xứng cấu hình là bắt buộc — bất đối xứng là nguồn sai lệch dễ bị
bỏ qua nhất khi so sánh.

| Thành phần | Giá trị |
|---|---|
| Target | `/models/google/gemma-4-26B-A4B-it-fp8-dynamic` (28.6 GB) |
| Drafter | `/models/speculators/google/gemma-4-26B-A4B-it-assistant` (839 MB) |
| Phần cứng | 1× H100 80GB, TP=1 |

## 2. Assistant drafter là gì

Đây **không** phải draft model cổ điển (một model nhỏ độc lập chạy song song).
Đọc `vllm/model_executor/models/gemma4_mtp.py`:

> *"The Gemma4 assistant model is a lightweight decoder that **shares KV cache**
> with the target (backbone) model. All assistant decoder layers are KV-shared:
> they only have **Q projections** (no K/V projections or norms), and read K/V
> from the target model's cache at runtime."*

Cấu hình của drafter:

| Thuộc tính | Giá trị | Nghĩa |
|---|---|---|
| `num_hidden_layers` | 4 | rất nông |
| `hidden_size` | 1024 | hẹp (target: 2816) |
| `enable_moe_block` | `false` | **drafter là dense** — chỉ backbone mới MoE |
| `backbone_hidden_size` | 2816 | khớp target, nối qua `pre/post_projection` |
| trọng số | 839 MB bf16 | |

### vLLM nhận diện nó thế nào

Chuỗi biến đổi trong `vllm/config/speculative.py`:

```
dòng 981   model_type "gemma4_assistant" -> "gemma4_mtp"
           n_predict = 1
           num_kv_shared_layers ép về 0   (chia sẻ KV giữa hai model do
                                           proposer dựng SAU khi model
                                           được khởi tạo)
dòng 1090  "gemma4_mtp" ∈ MTPModelTypes  -> rút gọn tiếp thành "mtp"
                                           (đặt "gemma4_mtp" vẫn chạy nhưng
                                            bị cảnh báo deprecated)
```

Nên cấu hình đúng là:

```json
{"method":"mtp",
 "model":"/models/speculators/google/gemma-4-26B-A4B-it-assistant",
 "num_speculative_tokens":2}
```

> **Bẫy:** nếu **bỏ trống** khoá `model`, vLLM sẽ lấy **chính target** làm
> draft (`speculative.py:1106`: `self.model = target_model_config.model`) —
> sai hoàn toàn và không báo lỗi. Với MTP nội sinh như Qwen3.8 (bài 02) thì
> bỏ trống là **đúng**; với assistant rời thì **phải** trỏ tường minh.

### Vì sao drafter này rẻ bất thường

Drafter **không có KV cache riêng** — nó đọc ké KV của target. So sánh với
bài 03 (DSpark) trên Qwen3.8-27B:

| | Checkpoint drafter | KV cache của target |
|---|---|---|
| Bài 02 MTP (nội sinh) | 477 MB | 973.279 token |
| Bài 03 DSpark | 4.0 GB | 550.320 token |
| **Bài 07 assistant** | **839 MB** | *(điền sau khi đo)* |

Đó là lý do chỉ phải lùi `gpu-memory-utilization` từ 0.90 xuống **0.89**,
thay vì xuống 0.86 như DSpark.

## 3. Điều kiện tiên quyết

- **Bài 06 đã chạy xong và đã xoá** (cần lại đúng 1 GPU đó).

```bash
kubectl delete -f ../06-gemma4-baseline-agg/deployment.yaml
```

- Kiểm tra drafter có sẵn:

```bash
kubectl exec -n token-factory deploy/bench-client -- \
  ls -la /models/speculators/google/gemma-4-26B-A4B-it-assistant/
```

## 4. Các bước thực hiện

> `cd 07-gemma4-spec-assistant`

### Bước 1 — Deploy

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deploy/vllm-g4-spec -n token-factory --timeout=15m
```

### Bước 2 — Xác nhận drafter thực sự được nạp

```bash
kubectl logs -n token-factory -l app=vllm-g4-spec --tail=600 \
  | grep -Ei "speculative|draft|Gemma4MTP|KV cache size"
```

Phải thấy kiến trúc `Gemma4MTPModel` và dung lượng KV cache. So dung lượng
này với bài 06 để biết drafter "ăn" mất bao nhiêu.

### Bước 3 — Smoke test

```bash
kubectl exec -n token-factory deploy/bench-client -- curl -s \
  http://vllm-g4-spec:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-26b-a4b","messages":[{"role":"user","content":"Viet ham Python tinh Fibonacci thu n. Chi code."}],"max_tokens":120,"temperature":0}'
```

> **Đọc kỹ output.** Lỗi nguy hiểm nhất của speculative decoding **không
> phải** là crash mà là **HTTP 200 kèm văn bản rác**. Bài 04 từng trả về
> `!ductductduct...` với mã 200. Nếu output vô nghĩa, dừng lại và kiểm tra
> cấu hình drafter.

### Bước 4 — Kiểm tra lại giới hạn 8 ảnh

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp ../06-gemma4-baseline-agg/test-images.py token-factory/$POD:/tmp/test-images.py
kubectl exec -n token-factory $POD -- python3 /tmp/test-images.py http://vllm-g4-spec:8000
```

### Bước 5 — Quét concurrency

Sửa **hai biến** trong `sweep-job.yaml` rồi chạy:

```bash
sed -e 's|value: "06-g4-agg"|value: "07-g4-spec"|' \
    -e 's|http://vllm-g4-agg:8000|http://vllm-g4-spec:8000|' \
    ../06-gemma4-baseline-agg/sweep-job.yaml > sweep-job.yaml

kubectl delete job sweep-bench -n token-factory --ignore-not-found
kubectl delete configmap sweep-script -n token-factory --ignore-not-found
kubectl apply -f sweep-job.yaml
kubectl logs -n token-factory -f job/sweep-bench
```

### Bước 6 — Đo acceptance length

Số quyết định của speculative decoding **không phải** acceptance *rate*
mà là acceptance *length* — trung bình bao nhiêu token được chấp nhận mỗi
bước. Bài 02 đã chứng minh điều này: `nspec=2` thắng `nspec=1` dù tỉ lệ
chấp nhận thấp hơn.

```bash
kubectl logs -n token-factory -l app=vllm-g4-spec --tail=2000 \
  | grep -Ei "acceptance|accepted|draft" | tail -20
```

### Bước 7 — Tinh chỉnh `num_speculative_tokens`

Assistant có `n_predict=1`: mỗi lần forward sinh **đúng 1** draft token, nên
`num_speculative_tokens=N` nghĩa là chạy nối tiếp N lần. Chi phí tăng tuyến
tính theo N, lợi ích thì bão hoà — nên có một giá trị tối ưu.

```bash
# thu N = 1, 2, 3
kubectl set env deploy/vllm-g4-spec -n token-factory --overwrite \
  DUMMY_BUMP=$(date +%s)   # ep rollout lai sau khi sua deployment.yaml
```

Sửa trực tiếp `num_speculative_tokens` trong `deployment.yaml` rồi
`kubectl apply -f deployment.yaml` cho mỗi giá trị.

## 5. Kết quả mong đợi

*(Số liệu thật được điền ở [99-compare-results](../99-compare-results/README.md).)*

Điều cần chú ý — và nó **khác** với model dense:

Speculative decoding làm mỗi bước decode **nặng hơn** (phải verify N+1 token
thay vì 1) nhưng **ít bước hơn**. Với MoE, việc verify nhiều token cùng lúc
cũng có nghĩa là **nhiều token hơn chạm vào expert trong cùng một bước** —
tức nó **đẩy engine về phía vùng bão hoà** của đường cong MoE.

Nói cách khác: với MoE, speculative decoding và batch lớn **cùng giải một
bài toán** (phủ expert). Vì vậy rất có thể lợi ích của chúng **không cộng
dồn** — spec decode giúp nhiều ở concurrency thấp và ít dần khi concurrency
tăng. Hãy kiểm tra giả thuyết này trên đường cong, đừng tin nó sẵn.

## 6. Giải thích

### Vì sao không dùng EAGLE3 speculator

Trong thư viện model còn có `/models/speculators/gemma-4-26B-A4B-it-speculator.eagle3`
(1.86 GB, `speculative_tokens: 3`). Bài này chọn assistant vì:

1. Nhẹ hơn **2,2 lần** (839 MB so với 1.86 GB).
2. Không có KV cache riêng, nên ít ảnh hưởng tới ngân sách KV của target.
3. Nó là drafter **chính chủ** do Google phát hành cùng model.

EAGLE3 là bài tập mở rộng tốt: đổi `--speculative-config` thành
`{"method":"eagle3","model":"/models/speculators/gemma-4-26B-A4B-it-speculator.eagle3","num_speculative_tokens":3}`
rồi so đường cong.

### Dọn dẹp

```bash
kubectl delete -f deployment.yaml
```

---

**Tiếp theo:** [Bài 08 — PD disaggregation + spec decode](../08-gemma4-pd-spec/README.md)
