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
- **PD disaggregation cần ≥ 2 GPU cho model 27B.** Chọn nó khi TPOT p99 ổn định quan trọng hơn throughput trung bình, hoặc khi bạn cần scale prefill và decode độc lập theo tải.

## 5. Việc chưa làm trong khoá này

Khoá này cố ý giữ phạm vi hẹp để tập trung vào PD serving và speculative decoding. Những hướng tiếp theo:

| Hướng | Vì sao đáng làm | Bắt đầu từ đâu |
|---|---|---|
| **KV-aware routing** | Proxy trong bài 04 là toy proxy, chọn instance ngẫu nhiên. Router thật định tuyến theo prefix đã cache → tiết kiệm prefill rất lớn cho agent (mỗi lượt gửi lại cả hội thoại) | [production-stack tutorial 17, 18](../../production-stack/tutorials/) |
| **Offload KV cache ra CPU/NVMe** | Mở rộng KV cache vượt giới hạn HBM — đúng vấn đề ta gặp ở bài 04 | [production-stack tutorial 05, 06](../../production-stack/tutorials/) |
| **Autoscaling** | Tải agentic coding rất thất thường theo giờ làm việc | [production-stack tutorial 10, 20](../../production-stack/tutorials/) |
| **Đo bằng trace thật** | SPEED-Bench đã là prompt thật, nhưng tỉ lệ code ⇄ tool_call trong agent của bạn mới quyết định con số cuối. Dùng `--dataset-name custom --dataset-path <trace.jsonl>` | [vllm bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/) |
| **So PD 1:1 với 2 replica agg** | Cùng 2 GPU, cùng TP1, khác kiến trúc — quyết định thật khi có GPU thứ hai. Chỉ cần `kubectl scale --replicas=2` ở bài 03 | Bài 04, mục 2 |
| **Thử tensor parallel** | Chuỗi bài cố ý giữ TP1. TP2 đáng thử khi cần ép TPOT xuống thấp hơn nữa và chấp nhận chi phí all-reduce | — |
| **Tỉ lệ P:D khác 1:1** | Với ≥ 3 GPU, tỉ lệ 2:1 hoặc 1:2 thường tốt hơn | Bài 04, Bước 6 |
| **Đánh giá chất lượng, không chỉ tốc độ** | Speculative decoding là lossless về mặt phân phối, nhưng FP8 weights/KV thì không. Cần đo HumanEval/MBPP trước–sau khi lượng tử hoá | — |
| **Tỉ lệ P:D và scale độc lập** | Câu hỏi thật của production: mua bao nhiêu GPU cho mỗi pool | Bài 04 Bước 6 |

---

**Quay lại:** [Tổng quan khoá học](../README.md)
