# Bài 02 — Speculative decoding với DSpark

Giữ nguyên cấu hình bài 01 và bật speculative decoding bằng drafter DSpark
trên **1 GPU**. So sánh trực tiếp với kết quả bài 01.

| Thành phần | Giá trị |
|---|---|
| Model | `/models/RedHatAI/gemma-4-12B-it-FP8-Dynamic` |
| Drafter | `/models/speculators/deepseek-ai/dspark_gemma4_12b_block7` |
| `--speculative-config` | `{"method":"dspark","model":"…/dspark_gemma4_12b_block7","num_speculative_tokens":7}` |
| Deployment / Service | `vllm-g4-spec`, cổng `8000` |
| GPU | 1 |
| Kết quả sweep | `/results/02-g12-dspark-c*.json` |

`num_speculative_tokens` bằng `block_size` của drafter (xem `config.json` của drafter).

## Điều kiện tiên quyết

- Đã chạy [bài 01](../01-gemma4-baseline-agg/README.md) và có `/results/01-g12-agg-c*.json`.
- 1 GPU H100 rảnh.

```bash
cd 02-gemma4-spec-dspark
```

## Bước 1 — Deploy

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deploy/vllm-g4-spec -n token-factory --timeout=15m
```

## Bước 2 — Xác nhận drafter được nạp

```bash
kubectl logs -n token-factory -l app=vllm-g4-spec --tail=1500 \
  | grep -E "speculative_config=|Available KV cache|KV cache size|multimodal embeddings" \
  | cut -c1-250
```

So KV cache với bài 01 và ghi vào bảng ở cuối bài.

## Bước 3 — Smoke test

```bash
kubectl exec -n token-factory deploy/bench-client -- curl -s \
  http://vllm-g4-spec:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-12b","messages":[{"role":"user","content":"Viet ham Python tinh Fibonacci thu n. Chi code."}],"max_tokens":200}'
```

## Bước 4 — Kiểm tra giới hạn 8 ảnh

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp ../01-gemma4-baseline-agg/test-images.py token-factory/$POD:/tmp/test-images.py
kubectl exec -n token-factory $POD -- python3 /tmp/test-images.py http://vllm-g4-spec:8000
```

Kết quả phải giống bài 01.

## Bước 5 — Quét concurrency

```bash
kubectl delete job sweep-bench-spec -n token-factory --ignore-not-found
kubectl apply -f sweep-job.yaml
kubectl logs -n token-factory -f job/sweep-bench-spec
```

## Bước 6 — Đo acceptance length

Trong lúc sweep đang chạy:

```bash
kubectl logs -n token-factory -l app=vllm-g4-spec --tail=3000 \
  | grep "Mean acceptance length" | tail -5
```

## Bước 7 — So với bài 01

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp ../04-compare-results/compare-spec.py token-factory/$POD:/tmp/compare-spec.py
kubectl exec -n token-factory $POD -- python3 /tmp/compare-spec.py 01-g12-agg 02-g12-dspark baseline dspark
```

Ghi kết quả của bạn:

| c | Output tok/s (01 → 02) | TPOT p50 ms (01 → 02) | ITL p50 ms (01 → 02) | TTFT p50 ms (01 → 02) |
|---:|---|---|---|---|
| 1 | | | | |
| 8 | | | | |
| 32 | | | | |
| 64 | | | | |
| 128 | | | | |

| | Bài 01 | Bài 02 |
|---|---|---|
| GPU KV cache size (tokens) | | |
| Mean acceptance length | — | |

Câu hỏi để quan sát:

- TPOT giảm bao nhiêu lần? Mức giảm thay đổi thế nào khi `c` tăng?
- ITL của bài 02 cao hơn hay thấp hơn bài 01? Vì sao? (Gợi ý: mỗi lần trả về có thể mang nhiều token.)
- KV cache thay đổi thế nào khi thêm drafter, và ảnh hưởng gì tới TTFT ở `c` cao?

## Dọn dẹp

```bash
kubectl delete -f deployment.yaml
kubectl delete -f sweep-job.yaml --ignore-not-found
```

---

**Tiếp theo:** [Bài 03 — PD disaggregation + DSpark](../03-gemma4-pd-spec/README.md)
