# Bài 04 — Tối ưu 3: PD Disaggregation + DSpark (2× H100)

## Giới thiệu

Bài 01 đã chỉ ra điểm đau: prefill và decode tranh nhau một GPU, và một prefill 30k token làm nghẽn toàn bộ decode đang chạy. Bài 02 và 03 tăng tốc decode nhưng **không hề động tới vấn đề đó**.

Bài này tách hẳn hai giai đoạn thành **hai engine trên hai GPU riêng**, nối với nhau bằng **NixlConnector** để truyền KV cache, và đặt DSpark lên đúng chỗ nó có tác dụng: **instance decode**.

> **Thay đổi về phần cứng so với bài 01–03.** Ba bài trước chạy trên **1× H100 80GB**. Bài này cần **2× H100 80GB**. Lý do nằm ở phép tính bộ nhớ ngay dưới đây — nhồi cả prefill lẫn decode vào một GPU là bất khả thi với model 27B, và cố làm sẽ khiến bài học sai lệch.

## Mục lục

1. [Kiến trúc PD disaggregation](#1-kiến-trúc-pd-disaggregation)
2. [Vì sao PD cần 2 GPU](#2-vì-sao-pd-cần-2-gpu)
3. [Bước 1: Xác nhận 2 GPU](#bước-1-xác-nhận-2-gpu)
4. [Bước 2: Deploy prefill + decode + proxy](#bước-2-deploy-prefill--decode--proxy)
5. [Bước 3: Kiểm tra KV transfer](#bước-3-kiểm-tra-kv-transfer)
6. [Bước 4: Benchmark](#bước-4-benchmark)
7. [Bước 5: Đo lại phép thử interference của bài 01](#bước-5-đo-lại-phép-thử-interference-của-bài-01)
8. [Bước 6: Tìm tỉ lệ P:D đúng](#bước-6-tìm-tỉ-lệ-pd-đúng)
9. [Xử lý sự cố](#xử-lý-sự-cố)
10. [Dọn dẹp](#dọn-dẹp)

## 1. Kiến trúc PD disaggregation

```
                    ┌────────────────────┐
   request ───────▶ │   Proxy / Router   │  (CPU only)
                    └─────────┬──────────┘
                              │ 1. gửi prompt đi prefill
                              ▼
              ┌───────────────────────────────┐
              │  PREFILL instance    [GPU 0]  │
              │  kv_role: kv_producer         │
              │  - prefill 8k-128k token      │
              │  - KHÔNG speculative decoding │
              │  - max_num_batched_tokens lớn │
              └───────────────┬───────────────┘
                              │ 2. đẩy KV cache qua NIXL
                              │    (NVLink / RDMA / UCX)
                              ▼
              ┌───────────────────────────────┐
              │  DECODE instance     [GPU 1]  │
              │  kv_role: kv_consumer         │
              │  + DSpark speculative decoding│
              │  - nhận KV, chỉ sinh token    │
              │  - max_num_seqs lớn           │
              └───────────────┬───────────────┘
                              │ 3. stream token về client
                              ▼
```

### Vì sao tách lại có lợi

| | Prefill | Decode |
|---|---|---|
| Bị chặn bởi | Compute (FLOPs) | Memory bandwidth |
| Batch lý tưởng | Ít seq, mỗi seq rất dài | Nhiều seq, 1 token/seq |
| Speculative decoding | **Vô dụng** — prefill vốn đã xử lý song song cả nghìn token | **Rất hiệu quả** |
| Tham số scheduler mong muốn | `max_num_batched_tokens` lớn | `max_num_seqs` lớn |

Khi chung một engine, bạn buộc phải chọn **một** bộ tham số thoả hiệp cho hai nhu cầu trái ngược. Tách ra, mỗi bên được tối ưu riêng. Quan trọng nhất: **decode không bao giờ bị prefill làm nghẽn nữa**, nên TPOT p99 trở nên ổn định.

### Vì sao DSpark chỉ đặt ở decode

Prefill đã xử lý song song hàng nghìn token trong một lần forward — GPU đã bão hoà compute. Thêm speculative decoding vào đó chỉ làm chậm. Đây là một lợi ích của PD mà agg mode không có: **bạn bật speculative decoding cho đúng nửa cần nó**, thay vì bật cho cả hai và chịu chi phí ở nửa không cần.

## 2. Vì sao PD cần 2 GPU

Đây là phép tính bạn phải làm được — và là lý do bài này khác ba bài trước về phần cứng.

### Nếu cố nhồi vào 1× H100 80GB

```
                              Prefill    Decode
Trọng số target FP8            27.5 GB    27.5 GB
Speculator DSpark BF16          —          4.0 GB
CUDA context + activation       3.0 GB     3.5 GB
                              ────────   ────────
Tối thiểu                      30.5 GB    35.0 GB
                              ────────────────────
Tổng tối thiểu                        65.5 GB
Còn lại cho KV cache của CẢ HAI    ≈ 14.5 GB / 80 GB
```

So với bài 01 (~40 GB KV cache) và bài 03 (~33 GB), cách này chỉ còn **~14.5 GB chia cho hai bên** — mỗi bên khoảng 7 GB, tức chỉ khoảng **230k token**. Ở `--max-model-len 131072`, mỗi bên chỉ đủ chỗ cho **1–2 sequence dài**; bạn sẽ preempt liên tục hoặc OOM ngay ở concurrency thấp.

Nguyên nhân gốc: **trọng số model bị nạp hai lần trên cùng một GPU.** Với model 27B, 55 GB trọng số trùng lặp đã ăn gần hết 80 GB. Thêm vào đó, GPU sharing (time-slicing/MPS) không tạo thêm SM hay băng thông — nó chỉ tách scheduler, không tách phần cứng. Bạn trả toàn bộ chi phí của PD mà gần như không nhận được lợi ích nào.

> **Lưu ý: đây không phải vấn đề mà tensor parallel giải quyết được.** TP chia *một* model qua nhiều GPU; PD chạy *hai* model trên nhiều GPU. Nếu bạn đặt TP2 cho cả prefill lẫn decode, mỗi engine chiếm cả 2 GPU và bạn lại quay về chỗ cũ — chỉ khác là giờ tốn thêm all-reduce. Chuỗi bài giữ **TP1 xuyên suốt**: một engine luôn là một GPU, nên mọi phép so sánh đều rõ ràng về đơn vị phần cứng.

### Với 2× H100 80GB

```
                    GPU 0 (prefill)      GPU 1 (decode)
Tổng HBM                80.0 GB              80.0 GB
- Trọng số target FP8   27.5 GB              27.5 GB
- Speculator DSpark        —                  4.0 GB
- CUDA context + graphs  3.5 GB               4.5 GB
                       ─────────            ─────────
  KV cache            ≈ 49.0 GB            ≈ 44.0 GB
```

Với `--gpu-memory-utilization 0.88` mỗi bên: prefill được ~40 GB KV cache (≈1.3M token), decode ~35 GB (≈1.1M token). **Không còn thâm hụt so với bài 01/03** — mỗi bên có ngân sách KV tương đương một deployment agg đầy đủ, đủ chỗ cho `--max-model-len 131072`, cộng thêm lợi ích tách scheduler.

Đây chính là điểm mấu chốt về kiến trúc: **PD disaggregation là cách đánh đổi GPU lấy khả năng tối ưu riêng từng giai đoạn và lấy độ ổn định độ trễ.** Nó chỉ có lãi khi bạn thực sự có GPU để đánh đổi.

### So sánh công bằng: bài 04 dùng 2 GPU, bài 01–03 dùng 1 GPU

Bài này dùng gấp đôi phần cứng của ba bài trước, nên **so sánh throughput tuyệt đối là không công bằng**. Hãy đọc kết quả theo hai cách:

| Cách so | Câu hỏi trả lời được |
|---|---|
| **Throughput tuyệt đối** | "Thêm 1 GPU nữa thì phục vụ được thêm bao nhiêu?" — nếu PD 2 GPU không đạt gần 2× bài 03, phần chênh là chi phí của kiến trúc PD |
| **Throughput / GPU** | "Đồng tiền bỏ ra hiệu quả hơn hay kém hơn?" — chia throughput cho 2 rồi so với bài 03 |
| **TPOT p99 khi có tải prefill nặng** | "Trải nghiệm người dùng có ổn định hơn không?" — **đây là chỗ PD thắng rõ nhất**, và không phụ thuộc số GPU |

Bước 5 sẽ đo riêng cột thứ ba.

> **Phép so sánh bổ sung đáng làm nhất — và nó vẫn là TP1.** Chạy **2 replica độc lập của bài 03** (agg + DSpark, mỗi replica 1 GPU TP1) sau một Service round-robin, rồi benchmark đúng cách như bài này. Khi ấy bạn so được **cùng 2 GPU, cùng TP1, khác kiến trúc**: "2 engine agg độc lập" so với "PD 1 prefill + 1 decode".
>
> Đây chính là quyết định thật mà đội vận hành phải đưa ra khi có GPU thứ hai, và nó **không cần tới tensor parallel**: chỉ cần `kubectl scale deploy/vllm-dspark --replicas=2` ở bài 03. Phép so này sạch hơn nhiều so với việc đổi sang TP2, vì TP2 thay đổi cả cách model chạy bên trong một engine.

## Bước 1: Xác nhận 2 GPU

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
```

Kết quả mong đợi:

```
NAME            GPU
h100-node-01    2
```

Nếu hai GPU nằm trên **hai node khác nhau**, mọi thứ vẫn chạy nhưng có hai điều phải xử lý:

1. PVC `model-cache` phải là `ReadWriteMany` (NFS) để cả hai node đọc được trọng số. Xem lại `00-prerequisites/02-model-pvc.yaml`.
2. KV cache đi qua mạng thay vì NVLink. NIXL hỗ trợ RDMA/UCX, nhưng **băng thông mạng trở thành nút thắt mới** — với InfiniBand thì ổn, với Ethernet 25GbE thì việc truyền KV sẽ chậm hơn cả việc tính lại prefill. Kiểm tra bằng cách so TTFT của bài này với bài 01.

Cấu hình trong `deployment.yaml` mặc định dùng `podAntiAffinity` mềm để **ưu tiên** hai engine nằm trên cùng node (tận dụng NVLink), nhưng không bắt buộc.

Kiểm tra NVLink giữa hai GPU nếu cùng node:

```bash
kubectl apply -f 00-gpu-topology-job.yaml
kubectl logs -f job/gpu-topology -n token-factory
```

Tìm `NV#` trong ma trận (ví dụ `NV18`) giữa GPU0 và GPU1 — nghĩa là có NVLink. Nếu chỉ thấy `PHB`/`SYS`, KV transfer sẽ đi qua PCIe, chậm hơn nhưng vẫn dùng được.

## Bước 2: Deploy prefill + decode + proxy

```bash
kubectl get pods -n token-factory     # xác nhận bài 03 đã xoá
kubectl apply -f deployment.yaml
```

File này tạo 3 thành phần:

| Thành phần | Vai trò | GPU | Speculative decoding |
|---|---|---|---|
| `vllm-prefill` | `kv_producer`, port 8000 | 1 (GPU 0) | Không |
| `vllm-decode` | `kv_consumer`, port 8000 | 1 (GPU 1) | **DSpark** |
| `pd-proxy` | Điều phối request, port 8192 | 0 | — |

Theo dõi (lần này phải chờ **hai** engine cùng load):

```bash
kubectl logs -f deploy/vllm-prefill -n token-factory
kubectl logs -f deploy/vllm-decode  -n token-factory
kubectl logs -f deploy/pd-proxy     -n token-factory
```

> Không cần time-slicing, không cần MPS, không cần chia `--gpu-memory-utilization` thủ công. Mỗi pod xin `nvidia.com/gpu: 1` và device plugin tự cấp một GPU riêng cho mỗi pod. Đây là một lợi ích phụ không nhỏ của việc làm đúng: **cấu hình đơn giản hơn hẳn, và có cách ly lỗi thật** — một engine OOM không kéo engine kia xuống.

### Giải thích cấu hình KV transfer

**Instance prefill:**

```yaml
- '--kv-transfer-config={"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
```

**Instance decode:**

```yaml
- '--kv-transfer-config={"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
```

| Khoá | Ý nghĩa |
|---|---|
| `kv_connector: NixlConnector` | NIXL là lớp truyền KV được khuyến nghị của vLLM — send/recv hoàn toàn bất đồng bộ, hỗ trợ nhiều backend (UCX, GDS, RDMA) |
| `kv_role` | `kv_producer` = bên prefill sinh KV; `kv_consumer` = bên decode tiêu thụ KV |
| `kv_load_failure_policy: fail` | Nếu truyền KV thất bại thì **báo lỗi rõ ràng** thay vì âm thầm tính lại prefill. Khi học, luôn để `fail` — nếu không bạn sẽ benchmark một hệ thống đang lặng lẽ chạy sai mà không biết |

Và biến môi trường bắt buộc:

```yaml
- name: VLLM_NIXL_SIDE_CHANNEL_PORT
  value: "5600"        # prefill
  # value: "5601"      # decode — PHẢI khác nhau NẾU cùng node
```

Side channel là kênh **bắt tay ban đầu** giữa prefill và decode (trao đổi metadata về vùng nhớ). Hai instance trên cùng một host bắt buộc dùng port khác nhau; manifest đã đặt sẵn 5600/5601 nên an toàn trong cả hai trường hợp.

### Vì sao tham số hai bên khác nhau

| Tham số | Prefill | Decode | Lý do |
|---|---|---|---|
| `--max-num-batched-tokens` | `16384` | `2048` | Prefill muốn nhồi thật nhiều token/batch; decode chỉ cần đủ cho các bước verify |
| `--max-num-seqs` | `16` | `128` | Prefill xử lý ít seq nhưng dài; decode cần batch rộng để tận dụng băng thông |
| `--speculative-config` | không có | DSpark (8 token) | Speculative decoding vô dụng ở prefill |
| `--gpu-memory-utilization` | `0.88` | `0.88` | Mỗi bên có GPU riêng — không phải chia thủ công như khi dùng chung |
| `--tensor-parallel-size` | `1` | `1` | Một engine = một GPU. PD dùng 2 GPU vì có **2 engine**, không phải vì tăng TP |
| `--enable-prefix-caching` | có | có | Agent gửi lại system prompt mỗi lượt |

> **`max_num_seqs` của decode được nâng lên 128** (bài 01–03 để 64). Đây là chỗ tách kiến trúc bắt đầu trả cổ tức: GPU decode không còn phải dành bộ nhớ và lịch chạy cho prefill, nên nó ôm được nhiều session đồng thời hơn. Hãy đối chiếu con số `Maximum concurrency` trong log của hai engine với bài 01.

## Bước 3: Kiểm tra KV transfer

Đây là bước dễ bị bỏ qua nhất và cũng dễ sai nhất. Gửi một request qua proxy:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- \
  curl -s http://pd-proxy:8192/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b",
       "messages":[{"role":"user","content":"Giải thích thuật toán binary search và cài đặt bằng Go."}],
       "max_tokens":400}' | python3 -m json.tool
```

Nếu có kết quả trả về, xác nhận **cả hai instance đều đã làm việc**:

```bash
# Prefill phải có prompt token, và rất ít/không có generation token
kubectl logs deploy/vllm-prefill -n token-factory | tail -20

# Decode phải có generation token
kubectl logs deploy/vllm-decode -n token-factory | tail -20
```

Kiểm tra bắt tay NIXL đã thành công:

```bash
kubectl logs deploy/vllm-decode -n token-factory | grep -i "nixl"
```

> **Cách sai phổ biến nhất:** hệ thống vẫn trả lời đúng, nhưng thực tế KV không được truyền — decode instance âm thầm tự prefill lại. Kết quả: benchmark trông "bình thường" nhưng bạn đang đo một kiến trúc hoàn toàn khác (thực chất là hai engine agg độc lập). Dấu hiệu nhận biết: log của decode có số lượng prompt token lớn. Nếu thấy vậy, kiểm tra side channel port và `kv_load_failure_policy`.

## Bước 4: Benchmark

**Cùng lệnh, cùng seed** — chỉ đổi endpoint sang proxy:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- bash

for C in 1 8 32 64; do
  vllm bench serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --base-url http://pd-proxy:8192 \
    --model qwen3.8-27b \
    --tokenizer /models/Qwen/Qwen3.8-27B-FP8 \
    --dataset-name random \
    --random-prefix-len 2048 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --random-range-ratio 0.2 \
    --num-prompts $(( C * 8 )) \
    --max-concurrency ${C} \
    --request-rate inf \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --seed 42 \
    --save-result \
    --result-dir /results \
    --result-filename "04-pd-dspark-c${C}.json" \
    --label "04-pd-dspark-c${C}"
done
```

Vì giờ có ngân sách KV cache đầy đủ ở cả hai bên, hãy chạy thêm mức concurrency cao để tìm trần thật của kiến trúc:

```bash
for C in 128 192; do
  vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --base-url http://pd-proxy:8192 --model qwen3.8-27b \
    --tokenizer /models/Qwen/Qwen3.8-27B-FP8 \
    --dataset-name random \
    --random-prefix-len 2048 --random-input-len 8000 \
    --random-output-len 1000 --random-range-ratio 0.2 \
    --num-prompts $(( C * 4 )) --max-concurrency ${C} --request-rate inf \
    --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 --seed 42 \
    --save-result --result-dir /results \
    --result-filename "04-pd-dspark-c${C}.json" --label "04-pd-dspark-c${C}"
done
```

Điền bảng — chú ý cột cuối:

| Conc | TPOT p50 (03 agg, 1 GPU) | TPOT p50 (04 PD, 2 GPU) | Out tok/s (03) | Out tok/s (04) | **Out tok/s per GPU (04)** |
|---|---|---|---|---|---|
| 1 | | | | | |
| 8 | | | | | |
| 32 | | | | | |
| 64 | | | | | |
| 128 | — | | — | | |
| 192 | — | | — | | |

**Cột cuối là cột quan trọng.** Nếu `out tok/s per GPU` của bài 04 thấp hơn bài 03, nghĩa là với workload này, hai engine agg độc lập sau một load balancer sẽ hiệu quả hơn PD — một kết luận hoàn toàn hợp lệ và rất đáng ghi nhận. Nếu nó cao hơn hoặc ngang bằng **trong khi TPOT p99 ổn định hơn nhiều**, PD là lựa chọn đúng.

## Bước 5: Đo lại phép thử interference của bài 01

Đây là **phép đo quan trọng nhất của bài này**, và là phép đo duy nhất không bị nhiễu bởi việc bài 04 dùng gấp đôi GPU. Lặp lại đúng thí nghiệm ở Bài 01 Bước 5.

Terminal 1 — tải prefill nặng:

```bash
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://pd-proxy:8192 --model qwen3.8-27b \
  --tokenizer /models/Qwen/Qwen3.8-27B-FP8 \
  --dataset-name random --random-input-len 30000 --random-output-len 50 \
  --num-prompts 64 --max-concurrency 16 --request-rate inf --ignore-eos
```

Terminal 2 — đồng thời đo trải nghiệm single-user:

```bash
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://pd-proxy:8192 --model qwen3.8-27b \
  --tokenizer /models/Qwen/Qwen3.8-27B-FP8 \
  --dataset-name random --random-input-len 1000 --random-output-len 500 \
  --num-prompts 20 --max-concurrency 1 --request-rate inf --ignore-eos \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99
```

Điền bảng:

| Cấu hình | TPOT p50 khi rảnh | TPOT p99 khi có tải prefill nặng | **Mức phình** |
|---|---|---|---|
| 01 agg | | | |
| 03 DSpark agg | | | |
| **04 PD + DSpark** | | | |

**Cột "mức phình" chính là câu trả lời của bài học.** Với bài 01 và 03, tải prefill nặng chen thẳng vào hàng đợi decode nên TPOT p99 phình mạnh. Với PD, GPU decode **không hề nhìn thấy** công việc prefill đó — nó chỉ nhận KV cache đã tính xong. Mức phình phải nhỏ hơn rõ rệt.

Với Token Factory, điều này quan trọng hơn throughput trung bình: người dùng chấp nhận "hơi chậm nhưng đều", chứ không chấp nhận "nhanh nhưng thỉnh thoảng đứng hình 3 giây" khi có đồng nghiệp vừa mở một repo lớn.

## Bước 6: Tìm tỉ lệ P:D đúng

Bài này dùng tỉ lệ **1 prefill : 1 decode**. Đó là điểm khởi đầu, không phải câu trả lời. Tỉ lệ đúng phụ thuộc hình dạng workload của bạn.

### Ước lượng từ chính số liệu của bạn

Trong output benchmark, lấy `Total input tokens` và `Total generated tokens`. Với profile của chuỗi bài này: 10048 token input (2048 prefix + 8000) và 1000 token output mỗi request — tỉ lệ 10:1.

Nhưng tỉ lệ token **không phải** tỉ lệ GPU, vì prefill xử lý song song còn decode thì tuần tự. Cách đo đúng là xem **mức bão hoà thực tế của từng engine**:

```bash
# Chạy tải ở concurrency cao rồi quan sát cả hai
kubectl exec -it deploy/bench-client -n token-factory -- \
  curl -s http://vllm-prefill:8000/metrics | grep -E "num_requests_running|num_requests_waiting"

kubectl exec -it deploy/bench-client -n token-factory -- \
  curl -s http://vllm-decode:8000/metrics | grep -E "num_requests_running|num_requests_waiting"
```

| Quan sát | Kết luận | Hành động |
|---|---|---|
| `waiting` dồn ở **prefill**, decode nhàn | Prefill là nút thắt | Tăng số replica prefill → tỉ lệ 2:1 |
| `waiting` dồn ở **decode**, prefill nhàn | Decode là nút thắt | Tăng số replica decode → tỉ lệ 1:2 |
| Cả hai đều bận đều | Tỉ lệ 1:1 đang đúng | Giữ nguyên, scale cả hai |

Với agentic coding (input dài, output dài), thường prefill là nút thắt ở đầu phiên và decode là nút thắt ở giữa phiên — nên tỉ lệ tối ưu thay đổi theo giờ. Đây chính là lý do PD thường đi kèm autoscaling riêng cho từng pool.

### Thử nghiệm mở rộng (nếu có thêm GPU)

`deployment.yaml` cho phép đổi `replicas` của từng Deployment. Proxy nhận nhiều host qua `--prefiller-hosts`/`--decoder-hosts`, nhưng toy proxy chỉ phân phối đơn giản — với nhiều replica bạn nên chuyển sang router thật (xem [bài 99](../99-compare-results/)).

### Ba câu hỏi để chốt bài

1. **Ở concurrency nào PD bắt đầu thắng rõ rệt?** Ở tải thấp, chi phí truyền KV qua NIXL chiếm tỉ trọng lớn trong TTFT nên PD có thể thua. Tìm điểm giao.

2. **Chi phí truyền KV là bao nhiêu?** So TTFT p50 ở concurrency 1 của bài 04 với bài 01. Phần chênh chính là thời gian đẩy KV cache từ GPU 0 sang GPU 1. Nếu có NVLink, con số này nên rất nhỏ; nếu đi qua mạng Ethernet, nó có thể lớn tới mức xoá sạch lợi ích.

3. **PD có cộng hưởng với DSpark không?** So acceptance length của bài 03 và bài 04:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- \
  curl -s http://vllm-decode:8000/metrics | grep -E "spec_decode"
```

Kỳ vọng bài 04 cao hơn hoặc ổn định hơn, vì: (a) decode instance được tối ưu thuần tuý cho decode; (b) không có prefill chen ngang nghĩa là CUDA graph của nhánh speculative ít bị phá vỡ hơn. Nếu số liệu không cho thấy điều đó, hãy ghi nhận đúng những gì đo được — đây là một giả thuyết cần kiểm chứng, không phải một lời hứa.

## Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Pod thứ hai kẹt `Pending` | Node chỉ có 1 GPU, hoặc GPU thứ hai đang bị pod cũ giữ | `kubectl describe node` xem `nvidia.com/gpu` allocatable và allocated. Xác nhận bài 03 đã xoá xong |
| Cả hai pod trên node khác nhau, decode không nhận được KV | PVC `ReadWriteOnce` không mount được ở node thứ hai | Đổi sang `ReadWriteMany` (NFS), hoặc dùng `nodeAffinity` ép cùng node |
| Decode treo, không sinh token | Bắt tay NIXL thất bại | Kiểm tra `VLLM_NIXL_SIDE_CHANNEL_PORT` hai bên khác nhau; Service `nixl` đã expose; xem log tìm `nixl` |
| Request trả về nhưng rất chậm | KV không được truyền, decode tự prefill lại | Xem log decode có prompt token lớn không. Giữ `kv_load_failure_policy: fail` để lỗi nổi lên rõ ràng |
| TTFT tệ hơn hẳn bài 01 | KV transfer đi qua PCIe/Ethernet thay vì NVLink | Chạy `00-gpu-topology-job.yaml`. Nếu là `SYS`/`PHB`, cân nhắc đặt hai engine trên hai GPU có NVLink |
| Proxy báo `connection refused` | Một trong hai engine chưa ready | Proxy phải khởi động **sau** cả hai; manifest đã có initContainer chờ sẵn |
| Throughput/GPU thấp hơn bài 03 | Có thể đúng với workload của bạn | Không phải lỗi. Xem [phần 2](#so-sánh-công-bằng-bài-04-dùng-2-gpu-bài-01-03-dùng-1-gpu) — hãy so cả TPOT p99 trước khi kết luận |

## Dọn dẹp

```bash
kubectl delete -f deployment.yaml
kubectl wait --for=delete pod -l lab=04-pd-dspark -n token-factory --timeout=300s
```

---

**Tiếp theo:** [Bài 99 — Tổng hợp và so sánh](../99-compare-results/)
