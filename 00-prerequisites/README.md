# Bài 00 — Chuẩn bị môi trường

## Giới thiệu

Bài này dựng nền cho toàn bộ chuỗi: namespace, secret HuggingFace, storage, và một **bench client pod** để chạy `vllm bench serve`.

Bài có **hai nhánh**, chọn đúng nhánh của bạn:

| Nhánh | Khi nào dùng | Khác biệt |
|---|---|---|
| **A — Workshop** | Bạn đang dự workshop FPT Transform | Trọng số model **đã có sẵn** tại host path `/mnt/hps/fp8_models`, image Docker đã pull sẵn. **Bỏ qua bước tải model** → tiết kiệm ~40 phút |
| **B — Tự học ở nhà** | Bạn tự dựng lại sau workshop | Tải trọng số từ HuggingFace xuống PVC |

Hai nhánh chỉ khác nhau ở **Bước 3 và Bước 4**. Bài 01–04 dùng chung y hệt nhau — vì cả hai nhánh đều tạo ra một PVC tên `model-cache` mount vào `/models`.

## Mục lục

1. [Điều kiện tiên quyết](#1-điều-kiện-tiên-quyết)
2. [Bước 1: Tạo namespace](#bước-1-tạo-namespace)
3. [Bước 2: Tạo secret HF_TOKEN](#bước-2-tạo-secret-hf_token)
4. [Bước 3A: Storage — nhánh Workshop](#bước-3a-storage--nhánh-workshop)
5. [Bước 3B: Storage — nhánh tự học ở nhà](#bước-3b-storage--nhánh-tự-học-ở-nhà)
6. [Bước 4: Xác minh trọng số](#bước-4-xác-minh-trọng-số)
7. [Bước 5: Deploy bench client](#bước-5-deploy-bench-client)
8. [Bước 6: Kiểm tra GPU](#bước-6-kiểm-tra-gpu)
9. [Layout thư mục trọng số](#layout-thư-mục-trọng-số)

## 1. Điều kiện tiên quyết

```bash
# Kiểm tra cluster thấy GPU
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
```

Kết quả mong đợi:

```
NAME            GPU
h100-node-01    2
```

- **Bài 01–03** cần 1 GPU. **Bài 04** cần 2 GPU.
- Nếu cột GPU trống → chưa cài NVIDIA GPU Operator:

  ```bash
  helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
  helm install --wait gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace
  ```

### Phiên bản dùng xuyên suốt

| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| vLLM | **`vllm/vllm-openai:v0.29.0`** | Pin cứng ở mọi manifest. Đây cũng là bản mà model card của DSpark speculator dùng để đánh giá |
| Model target | `Qwen/Qwen3.8-27B-FP8` | |
| DSpark speculator | `RedHatAI/Qwen3.8-27B-speculator.dspark-preview` | |

> **Workshop: image đã được pull sẵn trên node.** Mọi manifest đặt `imagePullPolicy: IfNotPresent` nên pod khởi động ngay, không chờ kéo ~10 GB image. Kiểm tra:
>
> ```bash
> kubectl debug node/h100-node-01 -it --image=busybox -- \
>   chroot /host crictl images | grep vllm-openai
> ```
>
> Nếu tự học ở nhà, lần deploy đầu tiên sẽ mất vài phút để kéo image — bình thường.

## Bước 1: Tạo namespace

```bash
kubectl apply -f 00-namespace.yaml
kubectl config set-context --current --namespace=token-factory
```

## Bước 2: Tạo secret HF_TOKEN

**Không** commit token vào git. Tạo secret trực tiếp:

```bash
kubectl create secret generic hf-token \
  --from-literal=token="hf_xxxxxxxxxxxxxxxxxxxx" \
  -n token-factory
```

File `01-hf-secret.yaml` chỉ là **template tham khảo** cho GitOps (môi trường thật hãy dùng SealedSecret/ExternalSecret/Vault).

> **Nhánh Workshop:** trọng số đã có sẵn trên host nên vLLM không cần gọi ra HuggingFace. Secret vẫn nên tạo (có thể để giá trị giả) vì manifest tham chiếu tới nó.

---

## Bước 3A: Storage — nhánh Workshop

Trọng số đã được ban tổ chức stage sẵn tại **`/mnt/hps/fp8_models`** trên node GPU. Ta chỉ cần bọc thư mục đó thành một PVC.

**Trước khi apply, sửa tên node** trong `02-storage-workshop.yaml`:

```yaml
nodeAffinity:
  required:
    nodeSelectorTerms:
      - matchExpressions:
          - key: kubernetes.io/hostname
            operator: In
            values:
              - h100-node-01        # <-- SỬA cho khớp `kubectl get nodes`
```

Rồi apply:

```bash
kubectl apply -f 02-storage-workshop.yaml
kubectl get pv,pvc -n token-factory
```

Kết quả mong đợi:

```
NAME                                CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM
persistentvolume/model-weights-pv   200Gi      RWO            Retain           Bound    token-factory/model-cache

NAME                                  STATUS   VOLUME             CAPACITY
persistentvolumeclaim/model-cache     Bound    model-weights-pv   200Gi
persistentvolumeclaim/vllm-cache      Bound    pvc-xxxx           20Gi
persistentvolumeclaim/bench-results   Bound    pvc-xxxx           5Gi
```

### Ba chi tiết đáng chú ý trong manifest này

| Chi tiết | Vì sao |
|---|---|
| `persistentVolumeReclaimPolicy: Retain` | **Quan trọng nhất.** Với `Delete`, xoá PVC sẽ xoá luôn dữ liệu trên host — nghĩa là xoá mất trọng số dùng chung của cả lớp. `Retain` khiến việc đó không thể xảy ra |
| `hostPath.type: Directory` | Pod **fail ngay lập tức** nếu `/mnt/hps/fp8_models` không tồn tại, thay vì âm thầm tạo thư mục rỗng rồi để vLLM crash sau 3 phút với lỗi khó hiểu |
| `nodeAffinity` | `hostPath` chỉ có nghĩa trên đúng node có thư mục đó. Thiếu nodeAffinity, pod có thể bị lên lịch sang node khác và thấy thư mục rỗng |

### Vì sao có PVC `vllm-cache` riêng

Thư mục trọng số trên host được mount **read-only** (nhiều học viên dùng chung, không ai được phép ghi đè). Nhưng vLLM cần ghi cache biên dịch `torch.compile`. Vì vậy ta tách ra một PVC nhỏ riêng, và các manifest đặt `VLLM_CACHE_ROOT=/cache`.

Lợi ích phụ đáng kể cho workshop: cache này **tồn tại xuyên suốt cả 4 bài**, nên từ lần deploy thứ hai trở đi thời gian khởi động giảm rõ rệt.

---

## Bước 3B: Storage — nhánh tự học ở nhà

Dùng file này **thay cho** `02-storage-workshop.yaml`:

```bash
kubectl apply -f 02-storage-home.yaml
kubectl get pvc -n token-factory
```

Sau đó tải trọng số:

```bash
kubectl apply -f 03-model-download-job.yaml
kubectl logs -f job/model-download -n token-factory
```

Job tải **2 repo**:

| Repo | Dùng ở bài | Kích thước |
|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | 01, 02, 03, 04 | ~28 GB |
| `RedHatAI/Qwen3.8-27B-speculator.dspark-preview` | 03, 04 | ~4.0 GB |

Đợi đến khi thấy:

```
==> Download completed
```

```bash
kubectl wait --for=condition=complete job/model-download -n token-factory --timeout=3600s
kubectl delete job model-download -n token-factory
```

> **Vì sao job chỉ tải 3 file của repo speculator.** Repo đó là một checkpoint huấn luyện đầy đủ, chứa cả `optimizer_state_dict.pt` (4.23 GB, state của Muon optimizer) và `scheduler_state_dict.pt`. vLLM **không đọc tới hai file đó** — chúng chỉ dùng nếu bạn muốn huấn luyện tiếp speculator. Job dùng `--include "config.json" "config.py" "model.safetensors"` để tải đúng phần cần cho serving, tiết kiệm 4.2 GB và một nửa thời gian tải.
>
> Đây là thói quen nên có khi lấy checkpoint từ HuggingFace: **mở tab Files and versions xem repo thực sự chứa gì** trước khi `hf download` cả repo.

---

## Bước 4: Xác minh trọng số

**Chạy bước này dù bạn theo nhánh nào.** Nó tốn 10 giây và tiết kiệm cho bạn 10 phút chờ pod crash-loop vì sai một ký tự trong đường dẫn.

```bash
kubectl apply -f 06-verify-models-job.yaml
kubectl logs -f job/verify-models -n token-factory
```

Kết quả mong đợi:

```
===== Cây thư mục thực tế dưới /models (2 cấp) =====
/models
/models/Qwen
/models/Qwen3.8-27B-FP8
/models/RedHatAI
/models/Qwen3.8-27B-speculator.dspark

===== [1/2] Target model =====
  OK  config.json
  OK  số file safetensors: 6
===== [2/2] DSpark speculator =====
  OK     config.json
  OK     config.py
  OK     model.safetensors

DAT — trọng số đã sẵn sàng. Tiếp tục sang bài 01.
```

Nếu job **FAIL**, log sẽ in ra cây thư mục thật và hướng dẫn xử lý. Xem tiếp [Layout thư mục trọng số](#layout-thư-mục-trọng-số).

```bash
kubectl delete job verify-models -n token-factory
```

## Bước 5: Deploy bench client

```bash
kubectl apply -f 04-bench-client.yaml
kubectl wait --for=condition=ready pod -l app=bench-client -n token-factory --timeout=300s
```

Pod này chứa sẵn `vllm` CLI (chỉ dùng phần client, không xin GPU) và mount PVC `bench-results` để lưu JSON kết quả của cả 4 bài — nhờ vậy bài 99 so sánh được mọi thứ trong một bảng.

```bash
kubectl exec -it deploy/bench-client -n token-factory -- bash
```

> **Client phải đủ mạnh.** Manifest xin 8 CPU vì ở concurrency 64 với output 1000 token, chính bench client có thể trở thành bottleneck và cho ra số liệu latency sai lệch. Nếu node của bạn eo hẹp CPU, hãy giảm mức concurrency cao nhất thay vì giảm CPU của client.

## Bước 6: Kiểm tra GPU

```bash
kubectl apply -f 05-gpu-check-job.yaml
kubectl logs -f job/gpu-check -n token-factory
kubectl delete job gpu-check -n token-factory
```

Kết quả mong đợi:

```
NVIDIA H100 80GB HBM3, 81559 MiB, 5xx.xx, 9.0
```

Ghi lại con số bộ nhớ. Toàn bộ tính toán ngân sách ở các bài sau dựa trên **80 GB**; nếu bạn dùng H100 40GB hoặc H100 NVL, các giá trị `--gpu-memory-utilization` sẽ phải điều chỉnh. Cột cuối (`9.0`) là compute capability — H100 là SM90, con số này có liên quan ở bài 03.

---

## Layout thư mục trọng số

Trên host workshop, trọng số nằm **phẳng** ngay dưới `/mnt/hps/fp8_models`, không có cấp thư mục tổ chức theo namespace HuggingFace:

```
/mnt/hps/fp8_models/                          <- mount vào /models trong container
├── Qwen3.8-27B-FP8/                          <- target model  (bài 01, 02, 03, 04)
│   ├── config.json
│   └── model-0000x-of-0000y.safetensors
└── Qwen3.8-27B-speculator.dspark/            <- DSpark speculator (bài 03, 04)
    ├── config.json
    ├── config.py
    └── model.safetensors
```

Vì vậy mọi manifest và lệnh benchmark dùng đúng hai đường dẫn:

```
/models/Qwen3.8-27B-FP8
/models/Qwen3.8-27B-speculator.dspark
```

Nhánh tự học ở nhà tải về **đúng cấu trúc phẳng này**, nên hai nhánh dùng chung manifest không cần sửa gì.

### Nếu layout trên máy bạn khác

Job ở [Bước 4](#bước-4-xác-minh-trọng-số) sẽ in ra cây thư mục thật và báo lỗi. Có hai cách xử lý:

**Cách 1 — tạo symlink trên host** (khuyến nghị: sửa một lần, không đụng vào manifest):

```bash
# Ví dụ: trọng số thật đang nằm ở /mnt/hps/fp8_models/qwen3.8-27b-fp8-v2/
cd /mnt/hps/fp8_models
sudo ln -s qwen3.8-27b-fp8-v2 Qwen3.8-27B-FP8
```

**Cách 2 — sửa manifest.** Các chỗ cần đổi:

| File | Số chỗ | Nội dung |
|---|---|---|
| `01-baseline-agg/deployment.yaml` | 1 | arg đầu tiên của `vllm serve` |
| `02-spec-decode-mtp/deployment.yaml` | 1 | arg đầu tiên |
| `03-spec-decode-dspark/deployment.yaml` | 3 | arg đầu tiên + `model` trong `--speculative-config` (dòng đang dùng và dòng comment) |
| `04-pd-disagg-dspark/deployment.yaml` | 3 | arg đầu tiên của prefill và của decode + `model` trong `--speculative-config` |
| `00-prerequisites/04-bench-client.yaml` | 1 | biến `MODEL_PATH` |
| `00-prerequisites/06-verify-models-job.yaml` | 2 | biến `TARGET` và `DRAFT` |

Và trong các README, mọi cờ `--tokenizer /models/Qwen3.8-27B-FP8` của lệnh benchmark.

Kiểm tra còn sót chỗ nào:

```bash
grep -rn "/models/Qwen3.8" . --include="*.yaml" --include="*.md"
```

---

**Tiếp theo:** [Bài 01 — Baseline vLLM aggregated mode](../01-baseline-agg/)
