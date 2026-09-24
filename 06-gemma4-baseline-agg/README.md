# Bài 06 — Baseline MoE: Gemma-4-26B-A4B ở chế độ agg

> **Phần 3 của khoá học.** Phần 1–2 (bài 01–05) đã chứng minh PD **thua** agg
> trên model dense. Phần 3 kiểm tra một giả thuyết cụ thể: kiến trúc **MoE**
> có tạo ra một cơ chế mà PD thắng được không — trong khi vẫn giữ **TP=1**.

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Vì sao đổi sang MoE](#2-vì-sao-đổi-sang-moe)
3. [Điều kiện tiên quyết](#3-điều-kiện-tiên-quyết)
4. [Các bước thực hiện](#4-các-bước-thực-hiện)
5. [Kết quả mong đợi](#5-kết-quả-mong-đợi)
6. [Giải thích](#6-giải-thích)

---

## 1. Giới thiệu

Bài này dựng **baseline** cho Phần 3: Gemma-4-26B-A4B (FP8) chạy aggregated,
**không** speculative decoding, **có** xử lý ảnh (tối đa 8 ảnh/request).

Nhiệm vụ chính của bài không phải "đo throughput cao nhất", mà là **quét
đường cong throughput theo concurrency để tìm điểm gãy của vùng phẳng**.
Kết quả đó quyết định bài 07 và 08 có đáng chạy hay không.

| Thành phần | Giá trị |
|---|---|
| Model | `/models/google/gemma-4-26B-A4B-it-fp8-dynamic` (28.6 GB) |
| Kiến trúc | MoE, **128 expert, top-8**, 30 layer, sliding 1024 : full = 5:1 |
| Tham số | 26B tổng / ~4B active |
| Phần cứng | 1× H100 80GB, TP=1 |
| vLLM | 0.29.0 |
| Multimodal | ảnh ≤ 8/request, video tắt |

## 2. Vì sao đổi sang MoE

Kết luận đo được ở Phần 1–2 với Qwen3.8-27B **dense**: PD thua agg ở mọi
cấu hình, và agg 3 replica thắng 2P1D toàn diện (throughput cao hơn 3,3 lần
ở QPS 3.0).

Lý do gốc nằm ở **roofline của decode**. Mỗi bước decode tốn:

```
thời gian một bước ≈ lượng trọng số phải đọc từ HBM ÷ bandwidth
```

**Dense 27B** — mỗi bước đọc **toàn bộ** ~30 GB, bất kể batch là 1 hay 128:

```
tok/s ≈ B × BW / 30GB      → TUYẾN TÍNH ngay từ B = 1
```

Nên chia tải ra 3 replica (mỗi replica batch `C/3`) gần như không mất gì.
"Thêm replica" mô phỏng được mọi lợi ích của PD, rẻ hơn và mịn hơn.

**MoE 26B-A4B** — mỗi bước chỉ đọc những expert được route tới:

| Vùng | Lượng đọc | Kết quả |
|---|---|---|
| B nhỏ (chạm ≈ `B × 8` expert) | tỉ lệ thuận với B | `tok/s` ≈ **hằng số** |
| B đủ lớn để phủ 128 expert | bão hoà | `tok/s` tuyến tính |

```
tok/s
  │                        ╱  MoE (sau ngưỡng phủ expert)
  │                    ╱
  │              ╱ ╱ dense (tuyến tính từ đầu)
  │        ╱  ╱
  │  ╱  ╱
  │━━━━━  ← MoE PHẲNG: batch nhỏ = lãng phí thuần
  └─────────────────────────────── batch
       ↑ ngưỡng phủ ≈ num_experts / top_k = 128/8 = 16
```

**Hệ quả:** MoE cần decode batch **lớn, liên tục, không bị ngắt** cấp thiết
hơn dense rất nhiều. Đó đúng là thứ PD bán — và lần đầu tiên nó là lợi thế
mà "thêm replica" **không** mô phỏng được, vì thêm replica thì **chia** batch,
đẩy mỗi engine xuống vùng phẳng.

> **Fan-out** `num_experts / top_k` là con số quyết định độ rộng vùng phẳng.
> Gemma-4-26B-A4B: `128/8 = 16`. gpt-oss-120b: `128/4 = 32` (mạnh gấp đôi).
> Mixtral 8x7B: `8/2 = 4` — gần như không có vùng phẳng.

## 3. Điều kiện tiên quyết

- Đã hoàn thành [bài 00](../00-prerequisites/README.md): namespace, PVC,
  bench-client, dataset SPEED-Bench.
- **1 GPU H100 rảnh.**

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
kubectl get pods -n token-factory
```

Kiểm tra trọng số đã có sẵn (workshop):

```bash
kubectl exec -n token-factory deploy/bench-client -- \
  ls -la /models/google/gemma-4-26B-A4B-it-fp8-dynamic/
```

Phải thấy `model.safetensors` ~28.6 GB.

## 4. Các bước thực hiện

> Mọi lệnh chạy từ **trong thư mục bài này**:
> `cd 06-gemma4-baseline-agg`

### Bước 1 — Deploy

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deploy/vllm-g4-agg -n token-factory --timeout=15m
```

### Bước 2 — Đọc dung lượng KV cache

```bash
kubectl logs -n token-factory -l app=vllm-g4-agg --tail=500 | grep "KV cache size"
```

### Bước 3 — Smoke test text

```bash
kubectl exec -n token-factory deploy/bench-client -- curl -s \
  http://vllm-g4-agg:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-26b-a4b","messages":[{"role":"user","content":"Viet ham Python tinh Fibonacci thu n bang quy hoach dong. Chi code."}],"max_tokens":120,"temperature":0}'
```

### Bước 4 — Kiểm tra giới hạn 8 ảnh

Script `test-images.py` gửi lần lượt 8, 9, 10 ảnh màu đặc và in kết quả.

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp test-images.py token-factory/$POD:/tmp/test-images.py
kubectl exec -n token-factory $POD -- python3 /tmp/test-images.py
```

### Bước 5 — Quét đường cong concurrency

**Đây là phép đo quyết định của cả Phần 3.**

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp sweep.sh token-factory/$POD:/tmp/sweep.sh
kubectl exec -n token-factory $POD -- bash -c \
  'chmod +x /tmp/sweep.sh && nohup /tmp/sweep.sh 06-g4-agg http://vllm-g4-agg:8000 > /results/06-g4-agg-sweep.log 2>&1 &'

# theo doi
kubectl exec -n token-factory $POD -- tail -f /results/06-g4-agg-sweep.log
```

Script quét `c ∈ {1,2,4,8,16,24,32,48,64,96,128}`, số prompt = `4×c` kẹp
trong `[32, 256]`.

> **Cảnh báo — nguồn sai lệch nghiêm trọng nhất của cả khoá học.**
> Hai tiến trình benchmark chạy chồng nhau trên cùng một server đã từng
> cho **315,90** và **495,11 tok/s** cho **cùng một cấu hình** (lệch 57%).
> `sweep.sh` tự kiểm tra trước khi chạy, nhưng hãy kiểm tra thủ công:
> ```bash
> kubectl exec -n token-factory $POD -- ps -eo pid,etime,args | grep "[b]ench serve"
> ```

## 5. Kết quả mong đợi

*(Số liệu thật của workshop được điền ở [99-compare-results](../99-compare-results/README.md).)*

Cách đọc đường cong — **đây là điểm mấu chốt**:

| Hình dạng đo được | Nghĩa là | Việc tiếp theo |
|---|---|---|
| `tok/s` tăng **gần tuyến tính** từ c=1 | **không** có vùng phẳng | Giả thuyết **sai**. Dừng lại, **không** chạy bài 07/08. |
| `tok/s` **gần như không đổi** tới một ngưỡng rồi mới tăng | **có** vùng phẳng | Ngưỡng đó là vùng cần quét ở bài 08. |

Cách tính nhanh từ file kết quả:

```bash
kubectl exec -n token-factory deploy/bench-client -- python3 -c '
import json,glob,re
rows=[]
for f in glob.glob("/results/06-g4-agg-c*.json"):
    c=int(re.search(r"-c(\d+)\.json",f).group(1)); d=json.load(open(f))
    rows.append((c,d["output_throughput"],d["median_tpot_ms"],d["median_ttft_ms"]))
print(f"{\"c\":>4} {\"tok/s\":>9} {\"tok/s/req\":>10} {\"TPOT ms\":>9} {\"TTFT ms\":>9}")
for c,t,tp,tt in sorted(rows):
    print(f"{c:>4} {t:>9.1f} {t/c:>10.1f} {tp:>9.2f} {tt:>9.0f}")
'
```

Cột **`tok/s/req`** là thứ cần nhìn: nếu nó **giảm mạnh** khi c tăng ở vùng
c nhỏ thì đó chính là vùng phẳng (throughput tổng gần như đứng yên trong khi
số request tăng).

## 6. Giải thích

### Vì sao `--limit-mm-per-prompt={"image":8,"video":0}`

Mặc định vLLM là **999** cho mỗi modality. Con số đó đi thẳng vào bước
profiling bộ nhớ lúc khởi động. Để mặc định, log báo:

```
Encoder cache will be initialized with a budget of 8192 tokens,
and profiled with 3 VIDEO items of the maximum feature size.
```

Ngân sách encoder bị đo khuôn theo **video** — thứ bài toán không dùng. Tắt
video đi thì cùng ngân sách đó được profiling bằng **29 image items**.

Giới hạn được thực thi nghiêm ở tầng API:

```
8 ảnh  -> 200 OK
9 ảnh  -> 400 "At most 8 image(s) may be provided in one prompt."
```

### Vì sao KV cache lớn bất thường

Gemma-4 dùng **sliding window attention** cho 25/30 layer (`sliding_window:
1024`), chỉ 5 layer là full attention. Nên KV của phần lớn layer bị chặn ở
1024 token mỗi sequence thay vì tăng theo độ dài context.

Đây cũng là lý do Gemma-4 **dễ hơn Qwen3.8 cho PD** (bài 08): KV nhỏ nghĩa
là chặng truyền NIXL rẻ.

### Vì sao `--max-num-seqs=128`

Giữ bằng bài 01–05 để so sánh xuyên suốt khoá học. Đây là **trần** của batch
decode; biến thực sự được quét là `--max-concurrency` ở phía client.

### Dọn dẹp

```bash
kubectl delete -f deployment.yaml
```

---

**Tiếp theo:** [Bài 07 — spec decode với assistant drafter](../07-gemma4-spec-assistant/README.md)
