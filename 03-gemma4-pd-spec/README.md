# Bài 03 — PD Disaggregation + DSpark

Tách prefill (P) và decode (D) thành hai engine vLLM riêng, nối với nhau bằng
NIXL, dùng cùng cấu hình DSpark của bài 02. Có hai bản:

| Bản | File | GPU | Dùng cho |
|---|---|---|---|
| **1 GPU** | `deployment-1gpu.yaml` | 1 — hai engine chia đôi một H100 | Workshop hands-on |
| **2 GPU** | `deployment-2gpu.yaml` | 2 cùng node — mỗi engine một GPU | So sánh hiệu năng |

Cả hai bản chạy trong **một pod** gồm hai container:

| Container | Thành phần | Cổng |
|---|---|---|
| `engines` | prefill `[P]` (`kv_producer`) | 8001 |
| `engines` | decode `[D]` (`kv_consumer`) | 8002 |
| `router` | `vllm-router` PD | 30000 |

Client gọi vào **router** ở cổng 30000.

| | Bản 1 GPU | Bản 2 GPU |
|---|---|---|
| Deployment / Service | `vllm-g4-pd` | `vllm-g4-pd2` |
| `--gpu-memory-utilization` | P `0.44` + D `0.44` trên cùng GPU | `0.90` mỗi engine |
| `--max-model-len` | `32768` | `131072` |
| Sweep job | `sweep-job-1gpu.yaml` → `sweep-bench-pd` | `sweep-job-2gpu.yaml` → `sweep-bench-pd2` |
| Kết quả sweep | `/results/03-g12-pd-1gpu-c*.json` | `/results/03-g12-pd-2gpu-c*.json` |

`--speculative-config`, `--block-size`, `--max-num-batched-tokens` và
`--max-num-seqs` giống hệt nhau ở P và D.

## Điều kiện tiên quyết

- Đã chạy [bài 01](../01-gemma4-baseline-agg/README.md) và [bài 02](../02-gemma4-spec-dspark/README.md).
- Bản 1 GPU: 1 GPU H100 rảnh. Bản 2 GPU: 2 GPU rảnh **trên cùng một node**.

```bash
cd 03-gemma4-pd-spec
```

---

## Phần A — Bản 1 GPU

### Bước A1 — Deploy

```bash
kubectl apply -f deployment-1gpu.yaml
kubectl rollout status deploy/vllm-g4-pd -n token-factory --timeout=25m
```

Hai engine khởi động nối tiếp (P trước, D sau), nên lần đầu mất vài phút.

### Bước A2 — Đọc KV cache của từng engine

```bash
kubectl logs -n token-factory deploy/vllm-g4-pd -c engines \
  | grep -E "Available KV cache|KV cache size|san sang"
```

### Bước A3 — Smoke test qua router

```bash
kubectl exec -n token-factory deploy/bench-client -- curl -s \
  http://vllm-g4-pd:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-12b","messages":[{"role":"user","content":"Viet ham Python tinh Fibonacci thu n. Chi code."}],"max_tokens":200}'
```

> **Đọc nội dung trả về, đừng chỉ nhìn mã HTTP.** Nếu `--speculative-config`
> hoặc `--block-size` lệch nhau giữa P và D, decode sinh ra văn bản rác mà vẫn
> trả HTTP 200.

### Bước A4 — Kiểm tra ảnh đi qua PD

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp ../01-gemma4-baseline-agg/test-images.py token-factory/$POD:/tmp/test-images.py
kubectl exec -n token-factory $POD -- python3 /tmp/test-images.py http://vllm-g4-pd:30000
```

### Bước A5 — Xác nhận KV đi qua NIXL

```bash
kubectl logs -n token-factory deploy/vllm-g4-pd -c engines --since=5m \
  | grep -E "\[D\].*(KV Transfer metrics|External prefix cache hit rate)" | tail -3
```

Mong đợi `External prefix cache hit rate: 100.0%` — D không tự prefill mà nhận
toàn bộ KV từ P. Ghi lại `Throughput (MB/s)` của `KV Transfer metrics`.

### Bước A6 — Quét concurrency

```bash
kubectl delete job sweep-bench-pd -n token-factory --ignore-not-found
kubectl apply -f sweep-job-1gpu.yaml
kubectl logs -n token-factory -f job/sweep-bench-pd
```

### Bước A7 — Dọn dẹp

```bash
kubectl delete -f deployment-1gpu.yaml
kubectl delete -f sweep-job-1gpu.yaml --ignore-not-found
```

---

## Phần B — Bản 2 GPU

### Bước B1 — Deploy

```bash
kubectl apply -f deployment-2gpu.yaml
kubectl rollout status deploy/vllm-g4-pd2 -n token-factory --timeout=25m
```

### Bước B2 — Kiểm tra hai GPU nối NVLink

```bash
kubectl exec -n token-factory deploy/vllm-g4-pd2 -c engines -- nvidia-smi topo -m
```

Ô giữa `GPU0` và `GPU1` phải là `NV#` (ví dụ `NV18`), không phải `PHB`/`SYS`.

### Bước B3 — Smoke test, ảnh và KV transfer

```bash
kubectl exec -n token-factory deploy/bench-client -- curl -s \
  http://vllm-g4-pd2:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-12b","messages":[{"role":"user","content":"Viet ham Python tinh Fibonacci thu n. Chi code."}],"max_tokens":200}'

POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp ../01-gemma4-baseline-agg/test-images.py token-factory/$POD:/tmp/test-images.py
kubectl exec -n token-factory $POD -- python3 /tmp/test-images.py http://vllm-g4-pd2:30000

kubectl logs -n token-factory deploy/vllm-g4-pd2 -c engines --since=5m \
  | grep "\[D\].*KV Transfer metrics" | tail -3
```

### Bước B4 — Quét concurrency

```bash
kubectl delete job sweep-bench-pd2 -n token-factory --ignore-not-found
kubectl apply -f sweep-job-2gpu.yaml
kubectl logs -n token-factory -f job/sweep-bench-pd2
```

### Bước B5 — Theo dõi GPU của P và D trong lúc sweep

GPU `0` là P, GPU `1` là D:

```bash
kubectl exec -n token-factory deploy/vllm-g4-pd2 -c engines -- \
  nvidia-smi --query-gpu=index,utilization.gpu,power.draw,clocks.sm,memory.used --format=csv -l 5
```

Trạng thái bên trong từng engine (số request đang chạy/chờ, KV cache):

```bash
kubectl logs -n token-factory deploy/vllm-g4-pd2 -c engines --since=1m \
  | grep -E "\[(P|D)\].*Engine 000" | tail -6
```

### Bước B6 — Dọn dẹp

```bash
kubectl delete -f deployment-2gpu.yaml
kubectl delete -f sweep-job-2gpu.yaml --ignore-not-found
```

---

## So sánh với bài 01 và 02

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp ../04-compare-results/compare-spec.py token-factory/$POD:/tmp/compare-spec.py

kubectl exec -n token-factory $POD -- python3 /tmp/compare-spec.py 01-g12-agg     03-g12-pd-2gpu baseline pd-2gpu
kubectl exec -n token-factory $POD -- python3 /tmp/compare-spec.py 02-g12-dspark  03-g12-pd-2gpu dspark   pd-2gpu
kubectl exec -n token-factory $POD -- python3 /tmp/compare-spec.py 02-g12-dspark  03-g12-pd-1gpu dspark   pd-1gpu
```

Ghi kết quả của bạn:

| c | Output tok/s: 01 / 02 / 03-1gpu / 03-2gpu | TPOT p50 ms: 01 / 02 / 03-1gpu / 03-2gpu | ITL p50 ms: 01 / 02 / 03-1gpu / 03-2gpu |
|---:|---|---|---|
| 1 | | | |
| 8 | | | |
| 32 | | | |
| 64 | | | |
| 128 | | | |

| Bản 2 GPU, trong lúc sweep | GPU 0 (P) | GPU 1 (D) |
|---|---|---|
| `utilization.gpu` | | |
| `power.draw` | | |

Câu hỏi để quan sát:

- Bản 2 GPU dùng gấp đôi phần cứng. So theo **tok/s mỗi GPU**, nó đứng ở đâu so với bài 02?
- TPOT của D thay đổi thế nào khi `c` tăng? Có mức `c` nào mà hiệu năng giảm đột ngột không? Xem `Running`/`Waiting` của D ở mức đó.
- GPU của P và GPU của D có được dùng đều nhau không? Tỉ lệ 1P:1D có hợp với workload 8k vào / 1k ra không?
- Ở bản 1 GPU, hai engine dùng chung những tài nguyên nào, và tách được những gì?

---

**Tiếp theo:** [Bài 04 — Tổng hợp kết quả](../04-compare-results/README.md)
