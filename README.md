# FPT Transform — Token Factory: Tối ưu Serving cho Agentic Coding

**Chuỗi bài hands-on:** tối ưu phục vụ (serving) open-weight model **Qwen3.8-27B** cho workload **Agentic Coding**, triển khai bằng **Kubernetes**, đo bằng **`vllm bench serve`**.

**Phần cứng:** bài 01–03 chạy trên **1× NVIDIA H100 80GB**; bài 04 (PD disaggregation) cần **2× H100 80GB**. Toàn chuỗi bài dùng **TP1** — một engine luôn là một GPU; bài 04 dùng GPU thứ hai vì nó chạy **engine thứ hai**, không phải vì xẻ nhỏ model. Lý do giải thích ở [mục 3](#3-cấu-hình-chuẩn-dùng-xuyên-suốt).

Mục tiêu: người học tự tay đi từ một baseline "chạy được" đến một cấu hình đã tối ưu, và **hiểu được vì sao** mỗi kỹ thuật lại có tác dụng — chứ không chỉ copy-paste tham số.

---

## 1. Bối cảnh: tại sao là Agentic Coding?

Token Factory của FPT phục vụ các agent coding (Cline / OpenHands / Claude Code-like). Workload này có đặc điểm rất riêng, và chính những đặc điểm đó quyết định kỹ thuật tối ưu nào đáng tiền:

| Đặc điểm workload | Hệ quả cho serving |
|---|---|
| Prompt rất dài (system prompt + tool schema + file context), phình dần theo lượt: 4k–100k token | Prefill nặng, tốn compute; TTFT dễ bị kéo dài |
| Output trung bình–dài (code + reasoning): 500–2000 token | Decode chiếm phần lớn wall-clock; bị chặn bởi **memory bandwidth**, không phải compute |
| Prefix dùng lại nhiều giữa các lượt trong cùng session | Prefix caching ăn rất mạnh |
| Người dùng ngồi chờ từng token | **TPOT / ITL** là KPI quan trọng nhất, sau đó mới tới throughput |
| Output là code → rất "dễ đoán" (cú pháp lặp, indent, tên biến) | **Speculative decoding có acceptance rate cao bất thường** → đây là đòn bẩy lớn nhất |

> Điểm mấu chốt của cả chuỗi bài: **decode là memory-bound**. GPU đọc toàn bộ 27B tham số từ HBM chỉ để sinh ra **1 token**. Nếu ta sinh được 3–4 token cho cùng một lần đọc trọng số đó, ta gần như được tăng tốc miễn phí. Đó chính là speculative decoding.

---

## 2. Lộ trình bài học

| Bài | Nội dung | Kỹ thuật | KPI kỳ vọng cải thiện |
|---|---|---|---|
| [00](00-prerequisites/) | Chuẩn bị cluster, HF token, PVC, tải model, bench client | — | — |
| [01](01-baseline-agg/) | **Baseline**: vLLM aggregated mode | Chunked prefill + prefix caching + FP8 KV | Mốc tham chiếu |
| [02](02-spec-decode-mtp/) | Tối ưu 1: **Speculative decoding — MTP** | MTP head có sẵn trong Qwen3.8 | TPOT ↓, output tok/s ↑ |
| [03](03-spec-decode-dspark/) | Tối ưu 2: **Speculative decoding — DSpark** | Drafter song song + adaptive verification | TPOT ↓↓, acceptance length ↑↑ |
| [04](04-pd-disagg-dspark/) | Tối ưu 3: **PD Disaggregation + DSpark** | Tách Prefill/Decode qua NixlConnector | TTFT ổn định, hết interference |
| [99](99-compare-results/) | Tổng hợp, so sánh, kết luận | — | — |

**Phần 1** của khoá = bài 00 + 01. **Phần 2** = bài 02 → 04.

Mỗi thư mục bài gồm:
- `README.md` — giải thích + hướng dẫn từng bước
- các file `*.yaml` — manifest Kubernetes để `kubectl apply`

---

## 3. Cấu hình chuẩn dùng xuyên suốt

Để các phép so sánh có ý nghĩa, **mọi bài đều giữ nguyên** các tham số sau:

| Tham số | Giá trị | Lý do |
|---|---|---|
| Model | `Qwen/Qwen3.8-27B-FP8` | FP8 → ~27.5 GB trọng số, vừa 1×H100 80GB và còn chỗ cho KV cache |
| Speculator (bài 03, 04) | `RedHatAI/Qwen3.8-27B-speculator.dspark-preview` | Định dạng `speculators` native của vLLM, `block_size = 8` |
| Engine | `vllm/vllm-openai:v0.29.0` | Pin cứng — cùng phiên bản thì benchmark mới so được |
| `--max-model-len` | `131072` (128k) | Đủ cho phiên agentic coding dài; giữ cố định để KV budget so sánh được. Model hỗ trợ native 262k |
| `--tensor-parallel-size` | `1` | Model FP8 chỉ 27.5 GB → vừa 1 GPU. TP2 chỉ thêm chi phí all-reduce mà không giải quyết vấn đề gì |
| `--kv-cache-dtype` | `fp8` | Gấp đôi số token KV chứa được |
| `--max-num-batched-tokens` | `8192` | Chunked prefill, tránh prefill dài chặn decode |
| Dataset benchmark | `random`, prefix 2048 / input 8000 / output 1000 | Mô phỏng agentic coding: system prompt chung + context file + code sinh ra |
| Concurrency sweep | 1, 8, 32, 64 | Từ single-user (độ trễ) tới batch (throughput) |

> **Ngân sách GPU.** H100 80GB, trọng số FP8 ≈ 27.5 GB → còn ~40 GB cho KV cache. Mỗi bài **phải xoá deployment của bài trước** rồi mới deploy bài mới.
>
> **Vì sao 128k context vừa 1 GPU.** Qwen3.8-27B dùng kiến trúc lai: chỉ **16 trong 64 lớp** là Gated Attention có KV cache tăng theo độ dài, 48 lớp còn lại là Gated DeltaNet với state cố định. Ước lượng ~32 KB/token ở FP8 → 40 GB chứa được **~1.3 triệu token**. Bài 01 mục 2 có phép tính đầy đủ.
>
> **Bài 04 cần 2 GPU, và đây là một phần của bài học.** PD disaggregation chạy hai engine, mỗi engine nạp một bản trọng số riêng. Nhồi cả hai vào 1× H100: 2 × 27.5 GB trọng số + context ≈ 65 GB, chỉ còn **~14.5 GB KV cache chia đôi cho hai bên** (~230k token mỗi bên) — ở `max-model-len 131072` chỉ đủ cho 1–2 sequence dài. Thêm nữa, GPU sharing (time-slicing/MPS) chỉ tách *scheduler*, không tách *phần cứng*: không có thêm SM, không có thêm băng thông. Bạn trả toàn bộ chi phí của kiến trúc PD mà gần như không nhận lại lợi ích nào.
>
> Với 2 GPU, mỗi engine có trọn 80 GB → KV cache ~40 GB và ~35 GB, **ngang ngửa một deployment agg đầy đủ ở cả hai bên**. Đó mới là điều kiện để PD thể hiện đúng giá trị của nó.
>
> Hệ quả khi đọc kết quả: bài 04 dùng gấp đôi phần cứng của bài 01–03, nên **đừng so throughput tuyệt đối**. Hãy so **throughput/GPU** và **độ ổn định của TPOT p99** — bài 04 và bài 99 hướng dẫn cụ thể.
>
> **TP vs PD — đừng nhầm hai thứ.** Tensor parallel chia *một* model qua nhiều GPU; PD chạy *hai* engine trên nhiều GPU. Chuỗi bài giữ TP1 xuyên suốt để GPU thứ hai ở bài 04 hoàn toàn thuộc về engine thứ hai, khiến mọi so sánh rõ ràng về đơn vị phần cứng.

---

## 4. Bộ KPI phải ghi lại ở mỗi bài

Sau mỗi bài, điền vào bảng trong [99-compare-results/](99-compare-results/):

| KPI | Lấy từ đâu | Ý nghĩa |
|---|---|---|
| **TTFT** p50 / p99 | output `vllm bench serve` | Độ trễ tới token đầu → cảm nhận "agent phản hồi nhanh" |
| **TPOT** p50 / p99 | output `vllm bench serve` | Thời gian mỗi token sau token đầu → tốc độ gõ code |
| **Output throughput** (tok/s) | output `vllm bench serve` | Công suất nhà máy token |
| **Request throughput** (req/s) | output `vllm bench serve` | Số phiên agent phục vụ được |
| **Acceptance length** | `/metrics` endpoint | Trung bình bao nhiêu token được chấp nhận mỗi bước speculative |

---

## 4b. Môi trường workshop

Workshop FPT Transform đã chuẩn bị sẵn để không ai phải ngồi chờ tải dữ liệu:

| Đã chuẩn bị sẵn | Ở đâu | Hệ quả |
|---|---|---|
| Trọng số model | Host path **`/mnt/hps/fp8_models`** trên node GPU | Bỏ qua bước tải model ở bài 00 → tiết kiệm ~40 phút |
| Image Docker | Đã `docker pull` sẵn trên node | Pod khởi động ngay; mọi manifest đặt `imagePullPolicy: IfNotPresent` |

Bài 00 có **hai nhánh**: nhánh **Workshop** (bọc host path thành PVC) và nhánh **Tự học ở nhà** (tải từ HuggingFace). Job tải model vẫn được giữ nguyên trong repo để bạn dựng lại toàn bộ môi trường sau workshop.

Bài 01–04 **không khác gì giữa hai nhánh** — cả hai đều tạo ra một PVC tên `model-cache` mount vào `/models`.

## 5. Yêu cầu trước khi bắt đầu

- Kubernetes có GPU H100 80GB, đã cài NVIDIA GPU Operator (hoặc device plugin):
  - **1 GPU** đủ cho bài 01–03
  - **2 GPU** cho bài 04 (ưu tiên cùng node, có NVLink)
- `kubectl` đã trỏ đúng cluster
- Tài khoản HuggingFace + `HF_TOKEN`
- Disk trống ≥ 120 GB cho PVC chứa model
- Kết nối internet từ node (hoặc mirror nội bộ của HF)

Bắt đầu tại **[00-prerequisites/](00-prerequisites/)**.

---

## 6. Ghi chú về phiên bản

Mọi manifest pin cứng **`vllm/vllm-openai:v0.29.0`**. Đây cũng là bản mà model card của DSpark speculator dùng để đánh giá, nên các con số acceptance trong bài 03 so sánh được.

Đừng đổi sang `:latest` khi đang làm bài: benchmark giữa 4 bài chỉ có ý nghĩa nếu chạy trên cùng một phiên bản engine. Nếu `--speculative-config` báo lỗi `unknown method`, bạn đang dùng image cũ hơn 0.29.0.

Tài liệu gốc tham chiếu:
- [vLLM — Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [vLLM — Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [vLLM — NixlConnector Usage](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- [vLLM — `vllm bench serve`](https://docs.vllm.ai/en/stable/cli/bench/serve/)
- [Speculators library](https://github.com/vllm-project/speculators)
- Model card: [`RedHatAI/Qwen3.8-27B-speculator.dspark-preview`](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark-preview)
