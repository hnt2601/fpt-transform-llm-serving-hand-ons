# Bài 05 — PD Disaggregation + MTP

## Giới thiệu

Bài 04 cho một kết quả khó chịu: **PD + DSpark thắng áp đảo về TPOT và ITL nhưng thua về throughput** — chỉ số quan trọng nhất với một Token Factory.

Bài này thay DSpark bằng MTP trong cùng kiến trúc PD, và lý do không phải "thử xem sao" mà xuất phát từ một con số cụ thể đo được ở bài 04.

> **Chạy mọi lệnh từ thư mục `05-pd-disagg-mtp/`:**
>
> ```bash
> cd 05-pd-disagg-mtp
> ```

## Mục lục

1. [Vì sao MTP hợp với PD hơn DSpark](#1-vì-sao-mtp-hợp-với-pd-hơn-dspark)
2. [Bước 1: Deploy](#bước-1-deploy)
3. [Bước 2: Kiểm tra KV transfer](#bước-2-kiểm-tra-kv-transfer)
4. [Bước 3: Benchmark](#bước-3-benchmark)
5. [Bước 4: Đọc kết quả](#bước-4-đọc-kết-quả)
6. [Dọn dẹp](#dọn-dẹp)

## 1. Vì sao MTP hợp với PD hơn DSpark

Bài 04 phát hiện ra rằng **NIXL bắt hai engine có cấu hình KV giống hệt nhau** (compatibility hash). Hệ quả: prefill **buộc phải** mang theo speculator dù nó không bao giờ chạy decode.

Cái giá của ràng buộc đó khác nhau rất nhiều giữa hai phương pháp:

| | Speculator | KV cache mỗi engine (đo ở bài 02, 03) |
|---|---:|---:|
| **MTP** | **477 MB** — nằm sẵn trong checkpoint target | **973.279 token** |
| DSpark | 4.0 GB — checkpoint rời | 550.320 token |

Và đây là con số quyết định, từ [bài 04](../04-pd-disagg-dspark/):

```
Bài 01 baseline   (1 GPU):                    1.200.036 token KV
Bài 04 PD+DSpark  (2 GPU): 573.664 + 632.321 = 1.205.985 token KV
```

> **PD+DSpark trên 2 GPU cho tổng KV bằng đúng baseline trên 1 GPU.** GPU thứ hai không mua thêm được dung lượng nào — nó bị speculator 4 GB nhân đôi ăn mất.
>
> Với MTP, speculator chỉ 477 MB. Kỳ vọng mỗi engine giữ được ~973k token như ở bài 02, tức **tổng ~1.9 triệu token** — gấp rưỡi baseline.

**Giả thuyết của bài 05:** PD+MTP giữ được lợi ích decode của PD (TPOT và ITL p99 rất tốt) **mà không phải trả bằng throughput**, vì ngân sách KV không còn bị bóp.

Đây là giả thuyết cần kiểm chứng, không phải kết luận. Bài 04 đã dạy rằng lập luận kiến trúc hợp lý vẫn có thể sai khi gặp thực tế.

### Những gì giữ nguyên từ bài 04

Toàn bộ phần khung PD không đổi, và **mọi bài học từ bài 04 vẫn áp dụng**:

| Cấu hình | Giá trị | Lý do |
|---|---|---|
| `VLLM_SSM_CONV_STATE_LAYOUT=DS` | cả hai engine | Kiến trúc lai: 48/64 lớp có conv state phải truyền qua NIXL |
| `--speculative-config` | **giống hệt** hai bên | NIXL compatibility hash |
| `--block-size 128` | cả hai | Truyền KV theo block |
| `cudagraph_mode: FULL_DECODE_ONLY` | chỉ decode | Decode không bao giờ prefill |
| NIXL side channel | 5557 / 5558 | Phải khác nhau |

## Bước 1: Deploy

Xoá bài 04 trước — chúng ta cần cả 2 GPU:

```bash
kubectl delete -f ../04-pd-disagg-dspark/deployment.yaml
kubectl wait --for=delete pod -l lab=04-pd-dspark -n token-factory --timeout=300s

kubectl apply -f deployment.yaml
kubectl wait --for=jsonpath='{.status.phase}'=Running \
  pod -l lab=05-pd-mtp -n token-factory --timeout=300s
```

Theo dõi (mỗi lệnh một terminal):

```bash
kubectl logs -f deploy/vllm-p-mtp      -n token-factory
kubectl logs -f deploy/vllm-d-mtp      -n token-factory
kubectl logs -f deploy/vllm-router-mtp -n token-factory
```

**Việc đầu tiên phải làm khi engine sẵn sàng** — đọc ngân sách KV của cả hai bên:

```bash
for app in vllm-p-mtp vllm-d-mtp; do
  POD=$(kubectl get pod -l app=$app -n token-factory -o jsonpath='{.items[0].metadata.name}')
  echo -n "$app: "
  kubectl logs $POD -n token-factory | grep -oE "GPU KV cache size: [0-9,]+ tokens"
done
```

Đây là phép kiểm chứng trực tiếp giả thuyết ở phần 1. Nếu tổng hai bên **không** vượt xa 1.2 triệu token của bài 04, giả thuyết sai và phần còn lại của bài không cần chạy.

<details>
<summary><b>Số đo tham chiếu</b> — giả thuyết ĐÚNG</summary>

```
vllm-p-mtp  :  35.68 GiB |   952.072 token |  7.26x @128k
vllm-d-mtp  :  36.84 GiB |   982.319 token |  7.49x @128k
────────────────────────────────────────────────────────
TỔNG        :            | 1.934.391 token
```

| Cấu hình | GPU | Tổng KV cache | So với baseline |
|---|---:|---:|---:|
| Baseline | 1 | 1.200.036 | 1.00× |
| PD+DSpark | 2 | 1.205.985 | **1.00×** |
| **PD+MTP** | **2** | **1.934.391** | **1.61×** |

Điểm mấu chốt nằm ở **prefill**:

| prefill | PD+DSpark | PD+MTP |
|---|---:|---:|
| KV cache | 573.664 token | **952.072 token** |
| Session @128k | 4.38× | **7.26×** |

NIXL bắt prefill mang theo speculator để khớp compatibility hash. Với DSpark đó là **4 GB** — ăn mất một nửa ngân sách KV của prefill. Với MTP chỉ **477 MB**, và nó vốn nằm sẵn trong checkpoint target nên gần như miễn phí.

Kết quả: mỗi engine giữ được ~950–980k token, **đúng bằng MTP agg trên 1 GPU** (973.279). Lần này GPU thứ hai thực sự mua thêm được dung lượng.

</details>

> **Một lỗi bạn sẽ gặp nếu tự tạo bài này bằng cách copy bài 04 rồi đổi tên.**
>
> Nếu dùng `sed` đổi hàng loạt `vllm-router` → `vllm-router-mtp`, bạn sẽ đổi nhầm cả **tên gói pip và tên binary**, không chỉ tên Deployment/Service:
>
> ```
> ERROR: Could not find a version that satisfies the requirement vllm-router-mtp
> ```
>
> Router vào `CrashLoopBackOff`. Tên Kubernetes object đổi được tự do; tên **gói phần mềm** thì không.

## Bước 2: Kiểm tra KV transfer

**Đừng bỏ qua bước này.** Bài 04 cho thấy kiểu hỏng nguy hiểm nhất: HTTP 200, benchmark chạy bình thường, nhưng nội dung sinh ra là rác.

```bash
kubectl exec deploy/bench-client -n token-factory -- bash -c '
curl -s http://vllm-router-mtp:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Viet ham Python dao nguoc linked list.\"}],\"max_tokens\":200}" \
  -o /tmp/r.json -w "http=%{http_code} time=%{time_total}s\n"
python3 -c "
import json; d=json.load(open(\"/tmp/r.json\"))
m=d[\"choices\"][0][\"message\"]
print(\"content:\", (m.get(\"content\") or \"(RONG - KV transfer hong)\")[:150])"'
```

**Phải đọc nội dung**, không chỉ nhìn HTTP status. Nếu thấy `(RONG)` hoặc chuỗi lặp vô nghĩa, tìm `hash mismatch` trong log decode.

Xác nhận hai engine làm đúng phần việc:

```bash
kubectl logs deploy/vllm-p-mtp -n token-factory | grep "Engine 000" | tail -1   # prompt cao, generation ~0
kubectl logs deploy/vllm-d-mtp -n token-factory | grep "Engine 000" | tail -1   # generation cao, prompt ~0
```

## Bước 3: Benchmark

Cùng lệnh, cùng seed, cùng dataset như bài 01–04:

```bash
kubectl exec deploy/bench-client -n token-factory -- bash -c '
for C in 1 8 32 64; do
  echo "############ CONCURRENCY ${C} ############"
  vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --base-url http://vllm-router-mtp:30000 --model qwen3.8-27b \
    --tokenizer /models/Qwen/Qwen3.8-27B-FP8 \
    --dataset-name speed_bench --dataset-path /datasets/speed-bench \
    --speed-bench-dataset-subset throughput_8k --speed-bench-output-len 1000 \
    --num-prompts $(( C * 8 )) --max-concurrency ${C} --request-rate inf \
    --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 --seed 42 \
    --save-result --result-dir /results \
    --result-filename "05-pd-mtp-c${C}.json" --label "05-pd-mtp-c${C}"
done'
```

## Bước 4: Đọc kết quả

Bài này chỉ có ý nghĩa khi so **năm chiều**. Điền bảng:

| Conc | 01 base | 02 MTP | 03 DSpark | 04 PD+DSpark | 05 PD+MTP |
|---|---|---|---|---|---|
| TPOT p50 @c32 | | | | | |
| Output tok/s @c32 | | | | | |
| TTFT p50 @c32 | | | | | |
| ITL p99 @c32 | | | | | |

<details>
<summary><b>Số đo tham chiếu</b> — năm chiều, SPEED-Bench <code>throughput_8k</code>, seed 42</summary>

| @c32 | baseline | MTP agg | DSpark agg | PD+DSpark | **PD+MTP** |
|---|---:|---:|---:|---:|---:|
| TPOT p50 (ms) | 36.72 | 30.15 | 21.19 | **10.66** | 14.53 |
| TPOT p99 (ms) | 38.44 | 41.86 | 31.07 | **16.30** | 16.56 |
| Output tok/s | 885.1 | **990.1** | 669.9 | 476.1 | 504.7 |
| TTFT p50 (ms) | 2.635 | **1.044** | 25.455 | 54.722 | 54.004 |
| ITL p99 (ms) | 438.0 | 454.9 | 305.7 | 41.2 | **39.3** |
| Thời lượng (s) | 289.2 | **258.6** | 382.1 | 537.7 | 507.3 |

| @c8 | baseline | MTP agg | DSpark agg | PD+DSpark | **PD+MTP** |
|---|---:|---:|---:|---:|---:|
| TPOT p50 (ms) | 16.56 | 12.19 | 13.13 | 9.38 | **8.61** |
| Output tok/s | 422.5 | **593.9** | 563.4 | 335.6 | 372.1 |
| ITL p99 (ms) | **18.5** | 268.5 | 280.5 | 30.1 | 22.5 |

</details>

### Kết quả: giả thuyết KV đúng, nhưng KHÔNG chuyển thành throughput

PD+MTP **thắng PD+DSpark ở mọi chỉ số** — đúng như dự đoán từ ngân sách KV:

| @c32 | PD+DSpark | PD+MTP | |
|---|---:|---:|---|
| Output tok/s | 476.1 | 504.7 | +6% |
| ITL p99 | 41.2 ms | **39.3 ms** | tốt hơn |
| TTFT p50 | 54.722 ms | 54.004 ms | tốt hơn chút |
| Thời lượng | 537.7 s | 507.3 s | nhanh hơn 6% |

**Nhưng ngân sách KV gấp 1.61 lần chỉ cho throughput cao hơn 6%.**

> **Đây là chỗ tác giả dự đoán sai, và cái sai đó đáng học.**
>
> Ở [bài 04](../04-pd-disagg-dspark/) tôi kết luận: "ba vấn đề PD chưa giải quyết đều quy về **một gốc duy nhất** — NIXL bắt prefill mang speculator 4 GB". Kết luận đó **sai**.
>
> Bài 05 nới đúng nút thắt đó ra gấp 1.6 lần, và gần như **không thay đổi gì**. Nghĩa là KV cache **chưa bao giờ là** nút thắt chính của PD ở mức tải này.
>
> Nút thắt thật là **cơ chế bàn giao**: agg chồng lấn prefill với decode trong cùng batch, PD tuần tự hoá chúng. Xem [phần phân tích cơ chế ở bài 04](../04-pd-disagg-dspark/#vì-sao-pd-chậm-hơn-agg--cơ-chế-đo-bằng-bằng-chứng).

### Ở c64, PD+MTP SỤP còn PD+DSpark thì không

Nhận định "PD+MTP tốt hơn PD+DSpark ở mọi mức" chỉ đúng tới c32. Ở c64 nó đảo ngược:

| @c64 | PD+DSpark | PD+MTP | |
|---|---:|---:|---|
| ITL p99 | **41.0 ms** | 153.3 ms | tệ gấp **3.7×** |
| TPOT p50 | **10.64 ms** | 25.16 ms | tệ gấp **2.4×** |
| TTFT p50 | 119.8 s | **78.8 s** | PD+MTP tốt hơn |
| Output tok/s | 480.1 | 475.6 | ngang nhau |

DSpark **giữ nguyên** độ mượt từ c32 sang c64 (41.2 → 41.0 ms). MTP thì từ 39.3 nhảy lên 153.3 ms.

> **Cách giải thích khớp với cơ chế bubble.** DSpark có acceptance length ~2.7 so với MTP ~2.18, nghĩa là mỗi bước decode sinh ra nhiều token hơn 24%. Ít bước decode hơn → **ít vòng bàn giao hơn** → ít bubble hơn.
>
> Ở c32 ngân sách KV còn thoải mái nên khác biệt này bị che. Ở c64, khi hàng đợi dài và bàn giao trở thành nút thắt thật, acceptance length cao hơn **mua được sự ổn định**.
>
> Đây là lần duy nhất trong cả chuỗi bài mà acceptance length cao hơn thực sự mang lại lợi ích đo được — và nó chỉ xảy ra trong kiến trúc PD, nơi mỗi bước decode tốn một round-trip.

### Với ràng buộc throughput, MTP agg vẫn thắng

| @c32 | MTP agg | PD+MTP | |
|---|---:|---:|---|
| Output tok/s | **990.1** | 504.7 | PD mất **49%** |
| ITL p99 | 454.9 ms | **39.3 ms** | PD tốt hơn **11.6×** |
| TTFT p50 | **1.0 s** | 54.0 s | PD tệ hơn **52×** |

**Nếu throughput là chỉ số quan trọng nhất — MTP agg, không phải PD.**

PD chỉ đáng khi SLA ràng buộc **độ mượt từng token** (ITL p99) và bạn chấp nhận TTFT rất xấu. Với agentic coding, ITL 39 ms so với 455 ms là khác biệt cảm nhận được rõ — nhưng chờ 54 giây cho token đầu thì không ai chấp nhận.

### Ba câu hỏi phải trả lời

1. **Tổng KV cache có tăng thật không?** So tổng hai engine với 1.205.985 token của bài 04. Đây là toàn bộ lý do bài 05 tồn tại.

2. **PD+MTP có giữ được ITL p99 tốt như PD+DSpark không?** Bài 04 đạt **41.2 ms @c32** so với 438 ms của baseline. Nếu PD+MTP giữ được mức đó mà throughput cao hơn, nó là cấu hình tốt nhất cho Token Factory.

3. **Throughput có vượt được MTP agg không?** MTP agg đạt **990.1 tok/s @c32** — cao nhất toàn chuỗi bài. Nếu PD+MTP không vượt được, kết luận là **PD không đáng cho workload này**, và đó cũng là một kết luận có giá trị.

### Khung quyết định

| Nếu PD+MTP… | Kết luận cho Token Factory |
|---|---|
| Throughput ≥ MTP agg **và** ITL p99 tốt hơn | **Cấu hình tốt nhất.** Đáng đầu tư GPU thứ hai |
| Throughput < MTP agg nhưng ITL p99 tốt hơn nhiều | Đánh đổi. Chọn theo SLA: ràng buộc độ mượt thì PD, ràng buộc công suất thì MTP agg |
| Throughput < MTP agg **và** ITL p99 không hơn | **PD không đáng** với workload này. Dùng MTP agg, thêm GPU chạy replica thứ hai |

## Dọn dẹp

```bash
kubectl delete -f deployment.yaml
kubectl wait --for=delete pod -l lab=05-pd-mtp -n token-factory --timeout=300s
```

---

**Tiếp theo:** [Bài 99 — Tổng hợp và so sánh](../99-compare-results/)
