# FPT Transform — Token Factory: Tối ưu Serving cho Agentic Coding

Chuỗi bài hands-on tối ưu phục vụ model **Gemma-4-12B (FP8)** bằng **vLLM** trên
**Kubernetes**, đo bằng **`vllm bench serve`** với dataset **SPEED-Bench**.

Người học đi từ một baseline chạy được, bật speculative decoding (DSpark), rồi
tách prefill/decode (PD disaggregation), và tự đo để so sánh các cấu hình.

---

## 1. Lộ trình bài học

| Bài | Nội dung | GPU |
|---|---|---|
| [00](00-prerequisites/) | Chuẩn bị: namespace, secret, storage, trọng số, dataset, bench client | 0 |
| [01](01-gemma4-baseline-agg/) | **Baseline** — vLLM aggregated, không speculative decoding | 1 |
| [02](02-gemma4-spec-dspark/) | **Speculative decoding với DSpark** | 1 |
| [03](03-gemma4-pd-spec/) | **PD disaggregation + DSpark** — bản 1 GPU (workshop) và bản 2 GPU | 1 hoặc 2 |
| [04](04-compare-results/) | Tổng hợp và so sánh kết quả | 0 |

Mỗi thư mục bài gồm `README.md` (các bước chạy) và các manifest `*.yaml` để
`kubectl apply`.

## 2. Cấu hình dùng xuyên suốt

| Tham số | Giá trị |
|---|---|
| Engine | `vllm/vllm-openai:v0.29.0` |
| Model | `RedHatAI/gemma-4-12B-it-FP8-Dynamic` |
| Drafter DSpark (bài 02, 03) | `deepseek-ai/dspark_gemma4_12b_block7`, `num_speculative_tokens: 7` |
| `--tensor-parallel-size` | `1` |
| `--max-num-batched-tokens` | `8192` |
| `--max-num-seqs` | `128` |
| `--max-model-len` | `131072` (bản PD 1 GPU: `32768`) |
| `--attention-backend` | `FLASH_ATTN` |
| `--limit-mm-per-prompt` | `{"image":8,"video":0,"audio":0}` |
| Dataset benchmark | SPEED-Bench `throughput_8k` |
| Mức concurrency quét | `1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128` |

## 3. Chỉ số cần ghi lại

| Chỉ số | Ý nghĩa |
|---|---|
| **Output throughput** (tok/s) | Công suất sinh token |
| **TPOT** p50 / p99 | Thời gian trung bình mỗi token sau token đầu |
| **ITL** p50 / p99 | Khoảng cách giữa hai lần trả token liên tiếp |
| **TTFT** p50 / p99 | Độ trễ tới token đầu tiên |
| **Acceptance length** | Số token được chấp nhận mỗi bước speculative (bài 02, 03) |

Mọi sweep lưu JSON vào PVC `bench-results` (`/results`), để bài 04 tổng hợp.

## 4. Yêu cầu

- Kubernetes có GPU H100 80GB, đã cài NVIDIA GPU Operator hoặc device plugin.
- 1 GPU cho bài 01, 02 và bản 1 GPU của bài 03; 2 GPU **cùng một node** cho
  bản 2 GPU của bài 03.
- `kubectl` đã trỏ đúng cluster.
- Tài khoản HuggingFace và `HF_TOKEN`.
- Nhánh tự học: dung lượng trống cho PVC chứa trọng số (~22 GB trọng số).

## 5. Môi trường workshop

| Đã chuẩn bị sẵn | Ở đâu |
|---|---|
| Trọng số model | Host path `/mnt/hps/fp8_models` trên node GPU |
| Image Docker | Đã pull sẵn trên node (`imagePullPolicy: IfNotPresent`) |

Bài 00 có hai nhánh: **Workshop** (bọc host path thành PVC) và **Tự học**
(tải từ HuggingFace). Từ bài 01 trở đi, hai nhánh chạy giống hệt nhau.

## 6. Bắt đầu

```bash
git clone https://github.com/hnt2601/fpt-transform-llm-serving-hand-ons.git
cd fpt-transform-llm-serving-hand-ons/00-prerequisites
```

Rồi làm theo [00-prerequisites/README.md](00-prerequisites/README.md).

Giữa các bài, xoá deployment của bài trước để giải phóng GPU:

```bash
kubectl get deploy -n token-factory
kubectl delete deploy <tên> -n token-factory
```

## 7. Tài liệu tham khảo

- [vLLM — Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [vLLM — Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [vLLM — NixlConnector Usage](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- [vLLM — `vllm bench serve`](https://docs.vllm.ai/en/stable/cli/bench/serve/)
- Model: [`RedHatAI/gemma-4-12B-it-FP8-Dynamic`](https://huggingface.co/RedHatAI/gemma-4-12B-it-FP8-Dynamic)
- Drafter: [`deepseek-ai/dspark_gemma4_12b_block7`](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7)
