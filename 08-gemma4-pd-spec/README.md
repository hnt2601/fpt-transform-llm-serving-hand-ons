# Bài 08 — PD Disaggregation + spec decode trên MoE

> Đây là bài **kiểm chứng giả thuyết trung tâm** của Phần 3.

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Phép so sánh đúng](#2-phép-so-sánh-đúng)
3. [Vì sao 1P:1D chứ không phải 2P:1D](#3-vì-sao-1p1d-chứ-không-phải-2p1d)
4. [Điều kiện tiên quyết](#4-điều-kiện-tiên-quyết)
5. [Các bước thực hiện](#5-các-bước-thực-hiện)
6. [Kết quả mong đợi](#6-kết-quả-mong-đợi)
7. [Giải thích](#7-giải-thích)

---

## 1. Giới thiệu

| Thành phần | Vai trò | Cổng | GPU |
|---|---|---|---|
| `vllm-g4-p` | `kv_producer` (prefill) | 8001 | 1 |
| `vllm-g4-d` | `kv_consumer` (decode) | 8002 | 1 |
| `vllm-g4-r` | `vllm-router` PD | 30000 | 0 (CPU) |

Client gọi vào **router** ở cổng 30000, không gọi thẳng engine.

## 2. Phép so sánh đúng

Phần 1–2 dạy một bài học đắt: so sánh PD với **một** replica agg là so sai.
Phải so ở **cùng số GPU**.

| Cấu hình | GPU | batch decode mỗi engine |
|---|---|---|
| agg × 2 replica (bài 06/07 nhân đôi) | 2 | `C/2` mỗi replica |
| **1P 1D** (bài này) | 2 | **`C` đầy đủ** trên D |

Đây chính là chỗ MoE khác dense. Với dense, chia batch làm đôi gần như không
mất gì (đường cong tuyến tính từ batch=1) — nên agg×2 thắng. Với MoE, nếu
`C/2` rơi vào **vùng phẳng** mà `C` thì không, việc gộp toàn bộ decode vào
một engine là lợi thế thật mà "thêm replica" **không** mô phỏng được.

**Dự đoán cần kiểm chứng:** PD thắng **đậm nhất ở concurrency trung bình** —
vùng mà `C` đã vượt ngưỡng phủ expert nhưng `C/2` thì chưa. Ngược hẳn với
model dense (bài 01–05), nơi agg càng thắng đậm khi tải tăng.

Nếu đồ thị có hình dạng đó, ta đã chứng minh được **cơ chế**, không chỉ một
con số.

## 3. Vì sao 1P:1D chứ không phải 2P:1D

Bài 05 phải dùng **2P:1D** vì prefill của Qwen3.8-27B **dense** là nút thắt
nguồn cung, đo được:

```
cung  = 14.000 tok/s ÷ 8.000 token mỗi prompt = 1,75 req/s
cầu   = C / (1000 × TPOT)  =  2,20 req/s ở c32
→ một prefill KHÔNG nuôi nổi decode
```

MoE lật ngược bài toán. FLOPs prefill tỉ lệ với **active params**, không phải
tổng params: **~4B thay vì 26B** — rẻ đi khoảng 6,5 lần. Nút thắt nguồn cung
biến mất.

Quan trọng hơn: với MoE ta **cố ý** muốn **ít engine decode nhưng to**. Thêm
một engine D thứ hai sẽ **chia đôi** batch decode, đẩy cả hai xuống vùng
phẳng — đúng cái đang cố tránh.

> Nếu đo thấy engine P bão hoà (TTFT tăng vọt trong khi GPU của D nhàn), hãy
> tăng `replicas` của `vllm-g4-p`, **không** tăng của `vllm-g4-d`.

## 4. Điều kiện tiên quyết

- **2 GPU rảnh trên CÙNG MỘT NODE** (để KV đi qua NVLink thay vì mạng).

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
kubectl delete -f ../07-gemma4-spec-assistant/deployment.yaml --ignore-not-found
```

- Kiểm tra topology NVLink (dùng lại job của bài 04):

```bash
kubectl apply -f ../04-pd-disagg-dspark/00-nvlink-check-job.yaml
kubectl wait --for=condition=complete job/nvlink-check -n token-factory --timeout=300s || true
kubectl logs -n token-factory job/nvlink-check
```

Tìm `NV#` giữa GPU0 và GPU1. Nếu thấy `PHB`/`SYS` thì hai GPU đi qua PCIe
hoặc qua mạng — chặng truyền KV sẽ đắt hơn nhiều.

## 5. Các bước thực hiện

> `cd 08-gemma4-pd-spec`

### Bước 1 — Deploy

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deploy/vllm-g4-p -n token-factory --timeout=15m
kubectl rollout status deploy/vllm-g4-d -n token-factory --timeout=15m
kubectl rollout status deploy/vllm-g4-r -n token-factory --timeout=10m
```

### Bước 2 — Kiểm tra hai engine ở cùng node

```bash
kubectl get pods -n token-factory -l lab=08-gemma4-pd -o wide
```

Cột `NODE` của `vllm-g4-p` và `vllm-g4-d` phải **giống nhau**.

### Bước 3 — Xác nhận NIXL bắt tay thành công

```bash
kubectl logs -n token-factory -l app=vllm-g4-d --tail=600 | grep -Ei "nixl|handshake|kv_connector|compatibility"
```

### Bước 4 — Smoke test qua router (BƯỚC QUAN TRỌNG NHẤT)

```bash
kubectl exec -n token-factory deploy/bench-client -- curl -s \
  http://vllm-g4-r:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-26b-a4b","messages":[{"role":"user","content":"Viet ham Python tinh Fibonacci thu n. Chi code."}],"max_tokens":120,"temperature":0}'
```

> ### ⚠️ Lỗi nguy hiểm nhất của toàn khoá học
>
> NIXL kiểm tra **compatibility hash** của cấu hình KV cache giữa hai engine
> trước khi truyền. Nếu `--speculative-config` **lệch nhau** giữa prefill và
> decode, hash khác nhau và decode sinh ra **văn bản rác** — kèm **HTTP 200**,
> **không** có lỗi nào nổi lên.
>
> Bài 04 từng trả về `!ductductduct...` với mã 200 và benchmark vẫn chạy
> "thành công", cho ra số liệu hoàn toàn vô nghĩa.
>
> **Vì vậy `--speculative-config` phải GIỐNG HỆT ở cả hai engine**, dù prefill
> không bao giờ dùng tới drafter. Hãy **đọc** output của bước này, đừng chỉ
> kiểm tra mã HTTP.

### Bước 5 — Kiểm tra ảnh đi qua được PD

Ảnh chỉ được xử lý ở **prefill**; decode chỉ nhận KV. Cần xác nhận đường đi
này hoạt động:

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp ../06-gemma4-baseline-agg/test-images.py token-factory/$POD:/tmp/test-images.py
kubectl exec -n token-factory $POD -- python3 /tmp/test-images.py http://vllm-g4-r:30000
```

### Bước 6 — Quét concurrency

```bash
sed -e 's|value: "06-g4-agg"|value: "08-g4-pd"|' \
    -e 's|http://vllm-g4-agg:8000|http://vllm-g4-r:30000|' \
    ../06-gemma4-baseline-agg/sweep-job.yaml > sweep-job.yaml

kubectl delete job sweep-bench -n token-factory --ignore-not-found
kubectl delete configmap sweep-script -n token-factory --ignore-not-found
kubectl apply -f sweep-job.yaml
kubectl logs -n token-factory -f job/sweep-bench
```

### Bước 7 — Chạy đối chứng agg × 2

Để so ở cùng 2 GPU, dựng bài 07 với 2 replica:

```bash
kubectl delete -f deployment.yaml
kubectl apply -f ../07-gemma4-spec-assistant/deployment.yaml
kubectl scale deploy/vllm-g4-spec -n token-factory --replicas=2
kubectl rollout status deploy/vllm-g4-spec -n token-factory --timeout=15m
```

Service `vllm-g4-spec` tự round-robin giữa hai pod. Rồi quét lại với
`LAB=07-g4-spec-x2`.

### Bước 8 — Đối chiếu hai đường cong

```bash
kubectl exec -n token-factory deploy/bench-client -- python3 -c '
import json,glob,re
def curve(pat):
    out={}
    for f in glob.glob(pat):
        c=int(re.search(r"-c(\d+)\.json",f).group(1))
        out[c]=json.load(open(f))["output_throughput"]
    return out
a=curve("/results/07-g4-spec-x2-c*.json")
p=curve("/results/08-g4-pd-c*.json")
print(f"{\"c\":>4} {\"agg x2\":>9} {\"1P1D\":>9} {\"PD/agg\":>8}")
for c in sorted(set(a)&set(p)):
    print(f"{c:>4} {a[c]:>9.1f} {p[c]:>9.1f} {p[c]/a[c]:>7.2f}x")
'
```

## 6. Kết quả mong đợi

*(Số liệu thật ở [99-compare-results](../99-compare-results/README.md).)*

| Hình dạng cột `PD/agg` | Nghĩa là |
|---|---|
| Có **bướu** > 1.0 ở c trung bình, tụt về ≤ 1.0 ở c cao | Cơ chế được xác nhận — PD gộp batch qua được vùng phẳng |
| **≤ 1.0 ở mọi c** | Cơ chế không đủ mạnh với fan-out 16× của model này |
| Tăng đều theo c | Có thứ khác đang chi phối — kiểm tra lại nút thắt prefill |

## 7. Giải thích

### Gemma-4 dễ hơn Qwen3.8 cho PD ở đâu

| | Qwen3.8-27B (bài 04/05) | Gemma-4-26B-A4B (bài này) |
|---|---|---|
| Kiến trúc | lai: 48/64 layer Gated DeltaNet | **attention thuần** |
| Trạng thái phải truyền | KV **+ conv state** | chỉ **KV** |
| Biến môi trường bắt buộc | `VLLM_SSM_CONV_STATE_LAYOUT=DS` | *(không cần)* |
| Chi phí truyền theo độ dài prompt | conv state **cố định** → không amortize | KV amortize bình thường |
| KV mỗi token | toàn bộ 16 layer attention | 25/30 layer bị chặn ở window 1024 |

Thiếu `VLLM_SSM_CONV_STATE_LAYOUT=DS` ở bài 04/05 làm engine chết ngay lúc
khởi động:

```
AssertionError: 3-read Mamba conv transfer requires DS conv state layout.
```

Gemma-4 không có vấn đề đó — và đó là một phần lý do chọn nó cho Phần 3.

### Vì sao `--block-size=128`

KV đi qua NIXL theo từng block. Block lớn → ít lần truyền, mỗi lần nhiều dữ
liệu. **Phải giống hệt nhau ở cả hai engine**, nếu không NIXL không ghép được
block.

### Vì sao decode dùng `cudagraph_mode: FULL_DECODE_ONLY`

Engine decode **không bao giờ** chạy prefill, nên chỉ cần capture CUDA graph
cho đường decode: khởi động nhanh hơn, ít bộ nhớ graph hơn, và graph của
nhánh speculative không bị phá vỡ. Đây là tối ưu **chỉ có thể làm khi đã
tách PD** — một lợi ích thật của kiến trúc này, độc lập với throughput.

### Vì sao `kv_load_failure_policy: fail`

Mặc định, khi truyền KV lỗi, vLLM **âm thầm tính lại prefill** ở phía decode.
Hệ thống vẫn trả lời đúng — nhưng benchmark đo PD trở thành đo agg, vô nghĩa.
`fail` bắt lỗi nổi lên rõ ràng.

### Dọn dẹp

```bash
kubectl delete -f deployment.yaml
```

---

**Tiếp theo:** [99 — Tổng hợp kết quả](../99-compare-results/README.md)
