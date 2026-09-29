# Bài 04 — Tổng hợp và so sánh kết quả

Gom kết quả sweep của bài 01–03 thành một bảng, và so từng cặp cấu hình theo
từng mức concurrency. Bài này không cần GPU.

| Kết quả | Cấu hình |
|---|---|
| `/results/01-g12-agg-c*.json` | 01 — agg, không spec, 1 GPU |
| `/results/02-g12-dspark-c*.json` | 02 — agg + DSpark, 1 GPU |
| `/results/03-g12-pd-1gpu-c*.json` | 03 — PD + DSpark, 1 GPU |
| `/results/03-g12-pd-2gpu-c*.json` | 03 — PD + DSpark, 2 GPU |

```bash
cd 04-compare-results
```

## Bước 1 — Kiểm tra đã đủ kết quả

```bash
kubectl exec -n token-factory deploy/bench-client -- \
  sh -c 'ls /results | grep -E "^0[1-3]-g12-.*-c[0-9]+\.json$" | sed -E "s/-c[0-9]+\.json//" | sort | uniq -c'
```

Mỗi cấu hình đã chạy phải có 11 file (`c = 1 … 128`).

## Bước 2 — Bảng tổng hợp

```bash
kubectl delete job compare-results -n token-factory --ignore-not-found
kubectl apply -f compare-job.yaml
kubectl wait --for=condition=complete job/compare-results -n token-factory --timeout=300s
kubectl logs job/compare-results -n token-factory
```

Job in ra bảng TTFT / TPOT / ITL / output tok/s theo từng cấu hình và từng mức
`c`, kèm bảng tăng tốc TPOT so với bài 01, rồi ghi ra
`/results/summary.md` và `/results/summary.csv`.

Lấy file về máy:

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp token-factory/$POD:/results/summary.md ./summary.md
kubectl cp token-factory/$POD:/results/summary.csv ./summary.csv
```

## Bước 3 — So từng cặp cấu hình

`compare-spec.py` in tỉ lệ B/A cho output tok/s, TPOT, ITL và TTFT ở mỗi mức
`c`:

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp compare-spec.py token-factory/$POD:/tmp/compare-spec.py

kubectl exec -n token-factory $POD -- python3 /tmp/compare-spec.py 01-g12-agg    02-g12-dspark  agg    dspark
kubectl exec -n token-factory $POD -- python3 /tmp/compare-spec.py 01-g12-agg    03-g12-pd-2gpu agg    pd-2gpu
kubectl exec -n token-factory $POD -- python3 /tmp/compare-spec.py 02-g12-dspark 03-g12-pd-2gpu dspark pd-2gpu
kubectl exec -n token-factory $POD -- python3 /tmp/compare-spec.py 02-g12-dspark 03-g12-pd-1gpu dspark pd-1gpu
```

Cột `B/A`: với tok/s, `> 1` là B tốt hơn; với TPOT, ITL, TTFT, `< 1` là B tốt hơn.

## Bước 4 — Điền bảng so sánh ba chiều

Chuẩn hoá theo số GPU để so công bằng.

| c | Output tok/s | | | Output tok/s **mỗi GPU** | | |
|---:|---:|---:|---:|---:|---:|---:|
| | 01 agg (1 GPU) | 02 DSpark (1 GPU) | 03 PD (2 GPU) | 01 | 02 | 03 |
| 1 | | | | | | |
| 8 | | | | | | |
| 32 | | | | | | |
| 64 | | | | | | |
| 128 | | | | | | |

| c | TPOT p50 (ms) | | | ITL p50 (ms) | | |
|---:|---:|---:|---:|---:|---:|---:|
| | 01 | 02 | 03 | 01 | 02 | 03 |
| 1 | | | | | | |
| 8 | | | | | | |
| 32 | | | | | | |
| 64 | | | | | | |
| 128 | | | | | | |

## Bước 5 — Câu hỏi thảo luận

1. Speculative decoding giảm TPOT bao nhiêu lần so với baseline? Mức giảm thay
   đổi thế nào khi concurrency tăng, và vì sao?
2. Vì sao ITL của cấu hình có speculative decoding lại có thể **cao hơn**
   baseline dù TPOT thấp hơn? Chỉ số nào phản ánh đúng tốc độ sinh token hơn?
3. PD disaggregation cải thiện chỉ số nào, và trả giá bằng gì? So theo tok/s
   **mỗi GPU** thì kết luận có thay đổi không?
4. Với 2 GPU và workload 8k vào / 1k ra, bạn sẽ chọn 1P1D hay chạy 2 replica
   agg + DSpark? Dựa trên số liệu nào?
5. Nếu tỉ lệ vào/ra thay đổi (prompt dài hơn nhiều, output ngắn hơn), lựa chọn
   ở câu 4 có đổi không?

## Dọn dẹp

```bash
kubectl delete job compare-results -n token-factory --ignore-not-found
```

---

**Quay lại:** [Tổng quan khoá học](../README.md)
