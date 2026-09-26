# Bài 99 — Tổng hợp, so sánh và kết luận

## Giới thiệu

Bạn đã chạy 4 cấu hình và có ~18 file JSON trong PVC `bench-results`. Bài này biến chúng thành một bảng quyết định: **cấu hình nào nên chạy trong Token Factory, và vì sao.**

> **Nhắc lại về phần cứng khi so sánh.** Bài 01–03 chạy trên **1 GPU**, bài 04 trên **2 GPU**. Mọi bảng throughput dưới đây phải đọc kèm cột **throughput/GPU**, nếu không bạn sẽ kết luận sai rằng PD luôn tốt hơn.

## Mục lục

1. [Trích xuất kết quả](#1-trích-xuất-kết-quả)
2. [Bảng tổng hợp](#2-bảng-tổng-hợp)
3. [Bốn câu hỏi phải trả lời được](#3-bốn-câu-hỏi-phải-trả-lời-được)
4. [Khuyến nghị cấu hình cho Token Factory](#4-khuyến-nghị-cấu-hình-cho-token-factory)
5. [Việc chưa làm trong khoá này](#5-việc-chưa-làm-trong-khoá-này)

## 1. Trích xuất kết quả

Deploy job tổng hợp:

```bash
kubectl apply -f compare-job.yaml
kubectl logs -f job/compare-results -n token-factory
```

Job đọc mọi file `*.json` trong `/results` và in ra bảng CSV + Markdown. Lấy file về máy:

```bash
kubectl cp token-factory/$(kubectl get pod -l app=bench-client -n token-factory \
  -o jsonpath='{.items[0].metadata.name}'):/results/summary.md ./summary.md
```

## 2. Bảng tổng hợp

Điền vào đây (hoặc dán output của job):

### 2.1 TPOT p50 (ms) — thấp hơn là tốt hơn

| Concurrency | 01 agg | 02 MTP | 03 DSpark | 04 PD+DSpark |
|---|---|---|---|---|
| 1 | | | | |
| 8 | | | | |
| 32 | | | | |
| 64 | | | | |

<details>
<summary><b>Số đo tham chiếu bài 01–03</b> — 1× H100 80GB, TP1, SPEED-Bench <code>throughput_8k</code>, seed 42</summary>

**TPOT p50 (ms)**

| Conc | 01 agg | 02 MTP | 03 DSpark |
|---:|---:|---:|---:|
| 1 | 12.44 | 7.14 | **6.63** |
| 8 | 16.56 | **12.19** | 13.13 |
| 32 | 36.72 | 30.15 | **21.19** |
| 64 | 62.54 | 46.97 | **21.35** |

**Output throughput (tok/s)**

| Conc | 01 agg | 02 MTP | 03 DSpark |
|---:|---:|---:|---:|
| 1 | 76.7 | 131.0 | **143.4** |
| 8 | 422.5 | **593.9** | 563.4 |
| 32 | 885.1 | **990.1** | 669.9 |
| 64 | **984.6** | 883.1 | 672.6 |

**TTFT p50 (ms)**

| Conc | 01 agg | 02 MTP | 03 DSpark |
|---:|---:|---:|---:|
| 1 | 599 | 585 | **580** |
| 8 | 2506 | **624** | 648 |
| 32 | 2635 | **1044** | 25.455 |
| 64 | **2266** | 21.135 | 73.060 |

**Chỉ số đặc thù**

| | KV cache | Session @128k | Acceptance length | Khởi động |
|---|---:|---:|---:|---:|
| 01 agg | 39.47 GiB | 9.16 | — | 7m07s |
| 02 MTP | 36.25 GiB (−19%) | 7.43 | 2.18–2.29 | 3m31s |
| 03 DSpark | 28.58 GiB (−54%) | 4.20 | 2.72–2.97 | 4m14s |

> **Đọc bảng này theo cột, không theo hàng.** Không cấu hình nào thắng toàn diện: DSpark thắng TPOT ở 3/4 mức nhưng thua throughput ở 3/4 mức, và TTFT của nó ở c64 là **73 giây**.

</details>

### 2.2 Output throughput (tok/s) — cao hơn là tốt hơn

| Concurrency | 01 agg (1 GPU) | 02 MTP (1 GPU) | 03 DSpark (1 GPU) | 04 PD+DSpark (2 GPU) | **04 chia cho 2 GPU** |
|---|---|---|---|---|---|
| 1 | | | | | |
| 8 | | | | | |
| 32 | | | | | |
| 64 | | | | | |
| 128 | — | — | — | | |
| 192 | — | — | — | | |

Cột cuối là cột dùng để ra quyết định mua sắm: nếu nó thấp hơn cột `03 DSpark`, thì với cùng số GPU, hai engine agg sau một load balancer cho throughput cao hơn PD.

### 2.3 TTFT p99 (ms) — thấp hơn là tốt hơn

| Concurrency | 01 agg | 02 MTP | 03 DSpark | 04 PD+DSpark |
|---|---|---|---|---|
| 1 | | | | |
| 8 | | | | |
| 32 | | | | |
| 64 | | | | |

### 2.4 Chỉ số đặc thù

| Cấu hình | GPU (đều TP1) | Acceptance length | KV cache (token) | Session 128k đồng thời | Mức phình TPOT p99 khi có tải prefill nặng |
|---|---|---|---|---|---|
| 01 agg | 1 | — (không dùng spec) | | | |
| 02 MTP | 1 | | | | |
| 03 DSpark | 1 | | | | |
| 04 PD+DSpark (decode) | 2 | | | | |

Cột cuối **không phụ thuộc số GPU** — đó là lý do nó là phép so sánh sạch nhất giữa agg và PD.

### 2.5 Phần 3 — Gemma-4-26B-A4B (MoE), bài 06–08

So hai bộ sweep bất kỳ theo từng mức concurrency bằng `compare-spec.py`:

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp compare-spec.py token-factory/$POD:/tmp/compare-spec.py
kubectl exec -n token-factory $POD -- python3 /tmp/compare-spec.py 06-g4-agg-fa4 07-g4-spec-n4 baseline spec
```

<details>
<summary><b>Số đo tham chiếu bài 06–08</b> — H100 80GB, TP1, SPEED-Bench <code>throughput_8k</code>, sampling temperature 1.0 mặc định</summary>

**Output throughput (tok/s) / TPOT p50 (ms)**

| Conc | 06 agg (1 GPU) | 07 spec K=4 (1 GPU) | 07 spec × 2 replica (2 GPU) | 08 1P1D NVLink (2 GPU) |
|---:|---:|---:|---:|---:|
| 1 | 191 / 5.19 | 315 / 2.31 | 369 / 2.27 | **415 / 2.04** |
| 8 | 936 / 8.48 | 1735 / 3.56 | 1303 / 3.71 | **1777 / 3.32** |
| 32 | 2106 / 14.96 | 3373 / 6.37 | **4163 / 4.96** | 3569 / 5.89 |
| 64 | 2781 / 21.81 | 4524 / 9.68 | **6725 / 6.73** | 5225 / 8.03 |
| 128 | 2970 / 32.04 | 4657 / 14.30 | **8044 / 9.86** | 3863 / 20.42 |

**Kết luận Phần 3:**

- **Spec decode (assistant MTP, K=4) thắng baseline ở mọi c:** TPOT ~0,43–0,45x,
  throughput 1,11–1,85x. Đổi lại TTFT tệ hơn 1,3–1,9x ở c=16–48.
- **K=4 là điểm tối ưu.** K=6 và K=8 tăng acceptance (3,03 / 4,08 so với ~3,3)
  nhưng throughput dưới tải thấp hơn (K=8 chỉ 0,65–0,97x so với K=4 ở c ≥ 8) vì
  assistant MTP draft tuần tự. Chúng chỉ thắng TPOT ở c=1–2.
- **PD thua thêm replica.** Cùng 2 GPU, 1P1D qua NVLink chỉ đạt 0,76–0,86x
  agg×2 ở c ≥ 16, và 0,48x ở c=128 khi KV cache của D đầy. Giả thuyết "PD thắng
  ở concurrency trung bình nhờ gom batch MoE" **bị bác bỏ** — chi tiết ở
  [bài 08, mục 6](../08-gemma4-pd-spec/README.md#6-kết-quả).
- **PD hai pod 1 GPU = KV đi TCP** (~365 MB/s), kể cả khi cùng node: throughput
  kẹt ~2.200 tok/s. Muốn NVLink thì P và D phải thấy GPU của nhau (cùng pod).

</details>

### 2.6 Tuỳ chọn speculative: `probabilistic` + `block`, adaptive verification

Hai model đều mặc định `temperature=1.0` và `vllm bench serve` không ép greedy,
nên mọi số đo ở đây đều **lấy mẫu**. A/B chỉ đổi đúng hai khoá
`draft_sample_method: greedy → probabilistic` và
`rejection_sample_method: standard → block`, hai pod chạy song song:

| Model / drafter | K | Acceptance length | Throughput B/A | TPOT B/A | Áp dụng? |
|---|---:|---|---|---|---|
| Qwen3.8 / DSpark (song song) | 8 | 2.77 → **3.16** | **1.03–1.13x** | 0.89–0.95x | ✅ bài 03, 04 |
| Gemma-4 / assistant MTP (tuần tự) | 4 | 3.21 → 3.34 | nhiễu, 0.85–1.24x | ~1.00x | ❌ |
| Gemma-4 / assistant MTP (tuần tự) | 8 | 4.03 → 4.08 | nhiễu, 0.79–2.02x | ~1.00x (trừ c=1–2) | ❌ |

`enable_adaptive_verification` **không chạy được trên H100**: nó cần attention
backend đọc query length từ device, trong vLLM 0.29 chỉ có FlashInfer
trtllm-gen (SM100 / Blackwell) và MLA indexer (DeepSeek). Xem
[bài 03, mục 2](../03-spec-decode-dspark/README.md#2-adaptive-verification).

## 3. Bốn câu hỏi phải trả lời được

Nếu bạn trả lời được cả bốn, bạn đã đạt mục tiêu của khoá.

### Câu 1 — Vì sao speculative decoding có tác dụng, và vì sao tác dụng đó teo dần khi concurrency tăng?

Câu trả lời phải nhắc tới: decode là **memory-bandwidth-bound**; ở batch nhỏ GPU dư compute nên verify nhiều token gần như miễn phí; ở batch lớn GPU đã bão hoà compute nên mỗi draft token bị từ chối trở thành lãng phí thật.

### Câu 2 — Vì sao DSpark đạt acceptance length cao hơn MTP, và cái giá phải trả là gì?

Phải nhắc tới: speculator đọc hidden state phụ từ **8 lớp của target** (không đoán mù); draft **song song** cả block 8 token thay vì 8 lượt tuần tự; Markov head bù cho việc thiếu quan hệ nhân quả; **confidence head** dự đoán xác suất chấp nhận từng vị trí. Cái giá: thêm ~4 GB VRAM, thêm một checkpoint phải quản lý, và ở concurrency cao backbone 5 lớp trở thành chi phí thật — đó là lý do cần adaptive verification.

Điểm cộng: phải nêu được rằng acceptance **không suy giảm theo độ dài context** (4.3–4.8 từ 8K tới 1M+ token), một tính chất rất hợp với agentic coding.

### Câu 3 — PD disaggregation giải quyết vấn đề gì mà speculative decoding không giải quyết được?

Phải nhắc tới: **decode interference**. Speculative decoding làm decode nhanh hơn nhưng decode vẫn bị prefill chen ngang. PD tách scheduler nên TPOT p99 ổn định. Và: PD cho phép bật speculative decoding **chỉ ở nửa cần nó**.

### Câu 4 — Vì sao PD disaggregation cần ít nhất 2 GPU cho model 27B?

Phải nhắc tới: mỗi engine nạp **một bản trọng số riêng**. Trên 1× H100: 2 × 27.5 GB + context ≈ 65 GB, chỉ còn ~14.5 GB KV cache chia đôi (~230k token mỗi bên) — ở `max-model-len 131072` chỉ đủ 1–2 sequence dài. Và GPU sharing (time-slicing/MPS) chỉ tách *scheduler*, không tách *phần cứng*: không thêm SM, không thêm băng thông. Với 2 GPU, mỗi bên có KV cache ~35–40 GB, ngang một deployment agg đầy đủ.

**Câu hỏi nối tiếp, khó hơn:** nếu bạn có 2 GPU, PD 1:1 có chắc tốt hơn **2 replica độc lập của bài 03 sau một load balancer** không? Cả hai đều là 2 GPU, đều TP1 — chỉ khác kiến trúc. Hãy trả lời bằng số liệu, không bằng trực giác; bài 04 phần 2 có gợi ý phép đo này.

Và phải phân biệt được: **TP là chia một model qua nhiều GPU, PD là chạy hai engine trên nhiều GPU.** Chuỗi bài giữ TP1 xuyên suốt nên GPU thứ hai ở bài 04 hoàn toàn dành cho engine thứ hai, không dành cho việc xẻ nhỏ model.

## 4. Khuyến nghị cấu hình cho Token Factory

Điền dựa trên số liệu **của bạn**, không phải số liệu trong tài liệu:

| Kịch bản | Cấu hình khuyến nghị | Căn cứ từ số liệu |
|---|---|---|
| 1 GPU H100, ít người dùng (< 8 phiên đồng thời) | | |
| 1 GPU H100, giờ cao điểm (32–64 phiên) | | |
| 2 GPU H100, ưu tiên độ trễ ổn định | | |
| 2 GPU H100, ưu tiên throughput/chi phí | | |

Khung suy luận chung — hãy kiểm chứng bằng số liệu của bạn:

- **Mặc định nên bật:** FP8 weights + FP8 KV cache + prefix caching + chunked prefill. Bốn thứ này gần như không có nhược điểm cho agentic coding.
- **MTP là lựa chọn đầu tiên** khi thêm speculative decoding: một cờ, không tốn VRAM, không cần quản checkpoint.
- **DSpark đáng đổi ~4 GB VRAM** khi phần lớn tải chạy ở concurrency thấp–trung bình và workload thiên về sinh code. Nếu agent của bạn chủ yếu gọi tool (acceptance ~3.57) thay vì sinh code (~4.20), lợi ích nhỏ hơn — hãy đo bằng trace thật.
- **Với DSpark, thêm `draft_sample_method: probabilistic` + `rejection_sample_method: block`** khi phục vụ ở temperature > 0 (+3–13% đo được, lossless). Đừng mặc định chép sang drafter khác: với assistant MTP của Gemma-4 hai cờ này không đo được lợi ích (mục 2.6).
- **Model MoE (Gemma-4-26B-A4B): thêm replica agg thay vì tách PD** khi có GPU thứ hai (mục 2.5).
- **PD disaggregation cần ≥ 2 GPU cho model 27B.** Chọn nó khi TPOT p99 ổn định quan trọng hơn throughput trung bình, hoặc khi bạn cần scale prefill và decode độc lập theo tải.

## 5. Việc chưa làm trong khoá này

Khoá này cố ý giữ phạm vi hẹp để tập trung vào PD serving và speculative decoding. Những hướng tiếp theo:

| Hướng | Vì sao đáng làm | Bắt đầu từ đâu |
|---|---|---|
| **KV-aware routing** | Proxy trong bài 04 là toy proxy, chọn instance ngẫu nhiên. Router thật định tuyến theo prefix đã cache → tiết kiệm prefill rất lớn cho agent (mỗi lượt gửi lại cả hội thoại) | [production-stack tutorial 17, 18](../../production-stack/tutorials/) |
| **Offload KV cache ra CPU/NVMe** | Mở rộng KV cache vượt giới hạn HBM — đúng vấn đề ta gặp ở bài 04 | [production-stack tutorial 05, 06](../../production-stack/tutorials/) |
| **Autoscaling** | Tải agentic coding rất thất thường theo giờ làm việc | [production-stack tutorial 10, 20](../../production-stack/tutorials/) |
| **Đo bằng trace thật** | SPEED-Bench đã là prompt thật, nhưng tỉ lệ code ⇄ tool_call trong agent của bạn mới quyết định con số cuối. Dùng `--dataset-name custom --dataset-path <trace.jsonl>` | [vllm bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/) |
| **So PD 1:1 với 2 replica agg cho Qwen3.8** | Đã làm cho Gemma-4 ở bài 08 (agg×2 thắng). Model dense 27B chưa được đo cùng cách. Chỉ cần `kubectl scale --replicas=2` ở bài 03 | Bài 04, mục 2; bài 08, mục 6 |
| **Thử tensor parallel** | Chuỗi bài cố ý giữ TP1. TP2 đáng thử khi cần ép TPOT xuống thấp hơn nữa và chấp nhận chi phí all-reduce | — |
| **Tỉ lệ P:D khác 1:1** | Với ≥ 3 GPU, tỉ lệ 2:1 hoặc 1:2 thường tốt hơn | Bài 04, Bước 6 |
| **Đánh giá chất lượng, không chỉ tốc độ** | Speculative decoding là lossless về mặt phân phối, nhưng FP8 weights/KV thì không. Cần đo HumanEval/MBPP trước–sau khi lượng tử hoá | — |
| **Tỉ lệ P:D và scale độc lập** | Câu hỏi thật của production: mua bao nhiêu GPU cho mỗi pool | Bài 04 Bước 6 |

---

**Quay lại:** [Tổng quan khoá học](../README.md)
