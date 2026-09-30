# Bài 01 — Baseline: Gemma-4-12B ở chế độ agg

Chạy Gemma-4-12B FP8 ở chế độ aggregated trên **1 GPU**, không speculative
decoding. Đây là mốc so sánh cho bài 02 và 03.

| Thành phần | Giá trị |
|---|---|
| Model | `/models/RedHatAI/gemma-4-12B-it-FP8-Dynamic` |
| Deployment / Service | `vllm-g4-agg`, cổng `8000` |
| GPU | 1 |
| Kết quả sweep | `/results/01-g12-agg-c*.json` |

## Điều kiện tiên quyết

- Đã hoàn thành [bài 00](../00-prerequisites/README.md), gồm cả Bước 7 (thư viện offline).
- 1 GPU H100 rảnh.

```bash
cd 01-gemma4-baseline-agg
```

## Bước 1 — Deploy

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deploy/vllm-g4-agg -n token-factory --timeout=15m
```

## Bước 2 — Đọc dung lượng KV cache

```bash
kubectl logs -n token-factory -l app=vllm-g4-agg --tail=1000 \
  | grep -E "Available KV cache|KV cache size"
```

Ghi lại hai con số này vào bảng ở cuối bài.

## Bước 3 — Smoke test text

```bash
kubectl exec -n token-factory deploy/bench-client -- curl -s \
  http://vllm-g4-agg:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-12b","messages":[{"role":"user","content":"Viet ham Python tinh Fibonacci thu n. Chi code."}],"max_tokens":200}'
```

## Bước 4 — Kiểm tra giới hạn 8 ảnh

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp test-images.py token-factory/$POD:/tmp/test-images.py
kubectl exec -n token-factory $POD -- python3 /tmp/test-images.py http://vllm-g4-agg:8000
```

Mong đợi:

```
[ 8 anh] 200  red, green, blue, yellow, magenta, cyan, white, orange
[ 9 anh] 400  ... At most 8 image(s) may be provided in one prompt.
[10 anh] 400  ... At most 8 image(s) may be provided in one prompt.
```

## Bước 5 — Quét concurrency

```bash
kubectl delete job sweep-bench-agg -n token-factory --ignore-not-found
kubectl apply -f sweep-job.yaml
kubectl logs -n token-factory -f job/sweep-bench-agg
```

Sweep chạy lần lượt `c = 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128` trên
`throughput_8k`. Chờ tới dòng `=== xong: /results/01-g12-agg-c*.json ===`.

## Bước 6 — Xem kết quả

```bash
kubectl exec -n token-factory deploy/bench-client -- python3 -c '
import json,glob,re
rows=[]
for f in glob.glob("/results/01-g12-agg-c*.json"):
    c=int(re.search(r"-c(\d+)\.json",f).group(1)); d=json.load(open(f))
    rows.append((c,d["output_throughput"],d["median_tpot_ms"],d["median_itl_ms"],d["median_ttft_ms"]))
print("%4s %9s %9s %9s %9s" % ("c","tok/s","TPOT ms","ITL ms","TTFT ms"))
for r in sorted(rows):
    print("%4d %9.1f %9.2f %9.2f %9.0f" % r)
'
```

Ghi kết quả của bạn:

| c | Output tok/s | TPOT p50 (ms) | ITL p50 (ms) | TTFT p50 (ms) |
|---:|---:|---:|---:|---:|
| 1 | | | | |
| 8 | | | | |
| 32 | | | | |
| 64 | | | | |
| 128 | | | | |

| KV cache | Giá trị |
|---|---|
| Available KV cache memory | |
| GPU KV cache size (tokens) | |

Câu hỏi để quan sát:

- Throughput tăng thế nào khi `c` tăng, và bão hoà ở mức nào?
- TPOT ở `c = 1` là bao nhiêu? Đây là tốc độ tối đa một người dùng đơn lẻ nhận được.
- TTFT bắt đầu tăng mạnh từ mức `c` nào?

## Dọn dẹp

```bash
kubectl delete -f deployment.yaml
kubectl delete -f sweep-job.yaml --ignore-not-found
```

---

**Tiếp theo:** [Bài 02 — Speculative decoding với DSpark](../02-gemma4-spec-dspark/README.md)
