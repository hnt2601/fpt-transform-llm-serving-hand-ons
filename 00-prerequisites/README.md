# Bài 00 — Chuẩn bị môi trường

## Giới thiệu

Bài này dựng nền cho toàn bộ chuỗi: namespace, secret HuggingFace, PVC chứa model, job tải model về PVC, và một **bench client pod** để chạy `vllm bench serve`.

Tải model về PVC **một lần duy nhất** là điểm quan trọng: 4 bài sau sẽ deploy lại vLLM nhiều lần, nếu mỗi lần đều tải 27 GB từ HuggingFace thì bạn sẽ mất nhiều giờ chờ đợi và kết quả benchmark bị nhiễu bởi thời gian khởi động.

## Mục lục

1. [Điều kiện tiên quyết](#1-điều-kiện-tiên-quyết)
2. [Bước 1: Tạo namespace](#bước-1-tạo-namespace)
3. [Bước 2: Tạo secret HF_TOKEN](#bước-2-tạo-secret-hf_token)
4. [Bước 3: Tạo PVC chứa model](#bước-3-tạo-pvc-chứa-model)
5. [Bước 4: Tải model xuống PVC](#bước-4-tải-model-xuống-pvc)
6. [Bước 5: Deploy bench client](#bước-5-deploy-bench-client)
7. [Bước 6: Kiểm tra GPU](#bước-6-kiểm-tra-gpu)

## 1. Điều kiện tiên quyết

```bash
# Kiểm tra cluster thấy GPU
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
```

Kết quả mong đợi:

```
NAME            GPU
h100-node-01    1
```

Nếu cột GPU trống → chưa cài NVIDIA GPU Operator. Cài bằng:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
helm install --wait gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace
```

## Bước 1: Tạo namespace

```bash
kubectl apply -f 00-namespace.yaml
```

Từ đây trở đi mọi lệnh đều dùng namespace `token-factory`. Đặt làm mặc định cho tiện:

```bash
kubectl config set-context --current --namespace=token-factory
```

## Bước 2: Tạo secret HF_TOKEN

**Không** commit token vào git. Tạo secret trực tiếp bằng lệnh:

```bash
kubectl create secret generic hf-token \
  --from-literal=token="hf_xxxxxxxxxxxxxxxxxxxx" \
  -n token-factory
```

File `01-hf-secret.yaml` trong thư mục này chỉ là **template tham khảo** cho GitOps (hãy dùng SealedSecret/ExternalSecret trong môi trường thật).

## Bước 3: Tạo PVC chứa model

```bash
kubectl apply -f 02-model-pvc.yaml
kubectl get pvc -n token-factory
```

Kết quả mong đợi:

```
NAME         STATUS   VOLUME     CAPACITY   ACCESS MODES
model-cache  Bound    pvc-xxxx   120Gi      RWO
```

### Giải thích các trường quan trọng

- **`storage: 120Gi`** — đủ cho `Qwen3.8-27B-FP8` (~28 GB) + DSpark speculator (~4 GB) + dư cho compile cache và checkpoint phụ.
- **`accessModes: ReadWriteOnce`** — đủ dùng vì tất cả pod đều nằm trên cùng 1 node (chúng ta chỉ có 1 GPU). Nếu cluster của bạn nhiều node, cân nhắc `ReadWriteMany` với NFS để bài 04 chạy prefill/decode trên node khác nhau.
- **`storageClassName`** — sửa cho khớp StorageClass của cluster bạn (`kubectl get sc`).

## Bước 4: Tải model xuống PVC

```bash
kubectl apply -f 03-model-download-job.yaml
kubectl logs -f job/model-download -n token-factory
```

Job này tải **2 repo**:

| Repo | Dùng ở bài | Kích thước |
|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | 01, 02, 03, 04 | ~28 GB |
| `RedHatAI/Qwen3.8-27B-speculator.dspark-preview` | 03, 04 | ~4.0 GB |

> **Vì sao job chỉ tải 3 file của repo speculator.** Repo này là một checkpoint huấn luyện đầy đủ, chứa cả `optimizer_state_dict.pt` (4.23 GB, state của Muon optimizer) và `scheduler_state_dict.pt`. vLLM **không đọc tới hai file đó** — chúng chỉ dùng nếu bạn muốn tiếp tục huấn luyện speculator. Job dùng `--include "config.json" "config.py" "model.safetensors"` để tải đúng phần cần cho serving, tiết kiệm 4.2 GB dung lượng PVC và một nửa thời gian tải.
>
> Đây là thói quen nên có khi lấy checkpoint từ HuggingFace: **mở tab Files and versions xem repo thực sự chứa gì** trước khi `hf download` cả repo.

Đợi đến khi thấy:

```
==> Download completed
/models/Qwen/Qwen3.8-27B-FP8
/models/RedHatAI/Qwen3.8-27B-speculator.dspark
```

Kiểm tra lại:

```bash
kubectl wait --for=condition=complete job/model-download -n token-factory --timeout=3600s
```

## Bước 5: Deploy bench client

```bash
kubectl apply -f 04-bench-client.yaml
kubectl wait --for=condition=ready pod -l app=bench-client -n token-factory --timeout=300s
```

Pod này chứa sẵn `vllm` CLI (chỉ dùng phần client, không xin GPU) và một PVC nhỏ `bench-results` để lưu file JSON kết quả của cả 4 bài. Nhờ vậy bài 99 có thể so sánh mọi thứ trong một bảng.

Vào shell của nó:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- bash
```

## Bước 6: Kiểm tra GPU

```bash
kubectl apply -f 05-gpu-check-job.yaml
kubectl logs -f job/gpu-check -n token-factory
```

Kết quả mong đợi — xác nhận đúng H100 80GB:

```
NVIDIA H100 80GB HBM3, 81559 MiB
```

Ghi lại con số này. Toàn bộ tính toán ngân sách bộ nhớ ở các bài sau dựa trên **80 GB**; nếu bạn dùng H100 40GB hoặc H100 NVL, các giá trị `--gpu-memory-utilization` sẽ phải điều chỉnh.

## Dọn dẹp các job đã xong

```bash
kubectl delete job model-download gpu-check -n token-factory
```

PVC và bench client **giữ lại** — dùng xuyên suốt 4 bài sau.

---

**Tiếp theo:** [Bài 01 — Baseline vLLM aggregated mode](../01-baseline-agg/)
