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

Kết quả mong đợi (tên node trên cluster quản lý thường dài, đó là bình thường):

```
NAME                                                      GPU
fke-ncp-modas-stg-...-workers-z1-f9f69-dn24z              8
fke-ncp-modas-stg-...-workers-z1-f9f69-wknqc              7
```

- **Bài 01–03** cần 1 GPU. **Bài 04** cần 2 GPU.
- Cluster có nhiều GPU hơn không sao — mỗi deployment chỉ xin đúng phần mình cần. Nhưng nhớ **xoá deployment của bài trước** trước khi sang bài mới, nếu không bạn đang so sánh trên mức tranh chấp tài nguyên khác nhau.
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
# Chỉ tạo nếu chưa có — xem ghi chú bên dưới
kubectl get ns token-factory >/dev/null 2>&1 || kubectl apply -f 00-namespace.yaml

kubectl config set-context --current --namespace=token-factory
```

> **Nhánh Workshop: namespace đã được cấp sẵn, hãy bỏ qua bước tạo.**
>
> Trên cluster quản lý (FPT Cloud FKE), mỗi học viên được cấp trước một namespace kèm ServiceAccount bị giới hạn quyền. Tài khoản đó **tạo được** namespace mới nhưng **không patch được** namespace đã có, nên `kubectl apply -f 00-namespace.yaml` sẽ báo:
>
> ```
> Error from server (Forbidden): namespaces "token-factory" is forbidden:
> User "system:serviceaccount:token-factory:..." cannot patch resource "namespaces"
> ```
>
> Đây **không phải lỗi của bạn** — namespace đã tồn tại và dùng được ngay. Dòng lệnh `kubectl get ... || kubectl apply ...` ở trên xử lý cả hai trường hợp.

Xác nhận namespace rỗng trước khi bắt đầu:

```bash
kubectl get all -n token-factory
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

Manifest này **không cần sửa gì** trước khi apply: `/mnt/hps` là shared storage hiện diện giống nhau trên mọi worker node, nên PV không khai báo `nodeAffinity` và pod nằm ở node nào cũng đọc được trọng số.

Apply:

```bash
kubectl apply -f 02-storage-workshop.yaml
kubectl get pvc -n token-factory
```

Kết quả mong đợi — **cả ba đều `Bound` ngay lập tức**:

```
NAME            STATUS   VOLUME                        CAPACITY   STORAGECLASS
bench-results   Bound    token-factory-bench-results   5Gi        hps-local
model-cache     Bound    token-factory-model-weights   500Gi      hps-local
vllm-cache      Bound    token-factory-vllm-cache      20Gi       hps-local
```

Nếu có PVC nào ở `Pending` quá vài giây, xem [Xử lý sự cố storage](#xử-lý-sự-cố-storage).

### Năm chi tiết đáng chú ý trong manifest này

| Chi tiết | Vì sao |
|---|---|
| `persistentVolumeReclaimPolicy: Retain` | **Quan trọng nhất.** Với `Delete`, xoá PVC sẽ xoá luôn dữ liệu — nghĩa là xoá mất thư viện model dùng chung của cả tổ chức. `Retain` khiến việc đó không thể xảy ra |
| `hostPath.type: Directory` | Pod **fail ngay lập tức** nếu `/mnt/hps/fp8_models` không tồn tại, thay vì âm thầm tạo thư mục rỗng rồi để vLLM crash sau 3 phút với lỗi khó hiểu |
| `claimRef` | Ghim PV vào đúng một PVC (namespace + tên). Chắc chắn hơn label selector: không PVC nào khác chiếm mất được, và PV cũng không bind nhầm sang claim trùng tên ở namespace khác |
| `storageClassName: hps-local` | Quy ước của cluster cho mọi PV trỏ vào `/mnt/hps`. **Không** dùng StorageClass mặc định — lý do ở mục dưới |
| Không có `nodeAffinity` | `/mnt/hps` là shared storage, hiện diện giống nhau trên mọi node. Ghim node sẽ làm pod không lên lịch được mà chẳng được lợi gì |

### Vì sao có PVC `vllm-cache` riêng

Thư mục trọng số trên host được mount **read-only** (nhiều học viên dùng chung, không ai được phép ghi đè). Nhưng vLLM cần ghi cache biên dịch `torch.compile`. Vì vậy ta tách ra một PVC nhỏ riêng, và các manifest đặt `VLLM_CACHE_ROOT=/cache`.

Lợi ích phụ đáng kể cho workshop: cache này **tồn tại xuyên suốt cả 4 bài**, nên từ lần deploy thứ hai trở đi thời gian khởi động giảm rõ rệt.

---

### Vì sao KHÔNG dùng StorageClass mặc định

Cách thông thường là để trống `storageClassName` cho `vllm-cache` và `bench-results` rồi để cluster cấp phát động. **Trên cluster workshop, cách đó hỏng.**

StorageClass mặc định `storageclass-20k` (`csi.vastdata.com`) bind PVC thành công nhưng **không attach được vào pod**:

```
AttachVolume.Attach failed for volume "pvc-xxxx" :
  rpc error: code = Unknown desc = [ControllerPublishVolume]:
  No VIP Pool named 'NCP-PRODUCT-STG-2821'
```

Triệu chứng nhìn thấy: pod kẹt `ContainerCreating` **vô thời hạn** — rất dễ nhầm là "đang kéo image chậm". Vì vậy cả ba PVC của chuỗi bài đều đi qua `/mnt/hps` bằng `hps-local`.

> **Bài học vận hành:** `PVC Bound` **không** có nghĩa là storage dùng được. Bind và attach là hai giai đoạn khác nhau; lỗi attach chỉ lộ ra khi có pod thật mount vào. Luôn kiểm chứng bằng một pod thật trước khi kết luận storage OK.

### Xử lý sự cố storage

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| PVC ở `Pending` mãi | SC dùng `WaitForFirstConsumer` (bind khi có pod đầu tiên), hoặc không có PV khớp | `kubectl describe pvc <tên>`. Với `hps-local` + `claimRef`, PVC phải `Bound` ngay |
| Pod kẹt `ContainerCreating` rất lâu | Lỗi **attach**, không phải kéo image | `kubectl describe pod <tên>` xem mục `Events` tìm `FailedAttachVolume` |
| `model-cache` không bind | PV và PVC lệch `storageClassName`, hoặc `claimRef` trỏ sai namespace | Cả hai phải là `hps-local`; `claimRef.namespace` phải là `token-factory` |
| PV `Released` không tái dùng được | PV còn giữ `claimRef` của claim cũ đã xoá | Dùng tên PV riêng cho khoá học (manifest đặt tiền tố `token-factory-`) để tránh đụng PV có sẵn trên cluster |

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
===== [1/2] Target model =====
Mong đợi: /models/Qwen/Qwen3.8-27B-FP8
  OK  config.json
  OK  số file safetensors: 68
  OK  mtp.safetensors (bài 02 dùng MTP head này)
  dung lượng: 30.5G

===== [2/2] DSpark speculator =====
Mong đợi: /models/speculators/RedHatAI/Qwen3.8-27B-speculator.dspark-preview
  OK     config.json
  OK     config.py
  OK     model.safetensors

==================================================
DAT — trọng số đã sẵn sàng. Tiếp tục sang bài 01.
==================================================
```

Dòng `mtp.safetensors` đáng chú ý: đó là **MTP head huấn luyện sẵn nằm ngay trong checkpoint target**, thứ mà [bài 02](../02-spec-decode-mtp/) dùng để bật speculative decoding chỉ bằng một cờ. Nếu dòng đó báo cảnh báo, bài 02 sẽ không chạy được.

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

Kiểm chứng client hoạt động trước khi sang bài 01:

```bash
kubectl exec deploy/bench-client -n token-factory -- bash -c '
  python3 -c "import vllm; print(\"vLLM\", vllm.__version__)"
  ls $MODEL_PATH/config.json
  touch /results/.probe && echo "/results ghi được" && rm /results/.probe
  touch /models/.probe 2>/dev/null || echo "/models read-only (đúng thiết kế)"
'
```

Kết quả mong đợi:

```
vLLM 0.29.0
/models/Qwen/Qwen3.8-27B-FP8/config.json
/results ghi được
/models read-only (đúng thiết kế)
```

Vào shell để làm benchmark ở các bài sau:

```bash
kubectl exec -it deploy/bench-client -n token-factory -- bash
```

> **Mẹo tra cứu cờ `vllm bench serve`.** Từ 0.29.0, `--help` chỉ liệt kê **nhóm cấu hình** chứ không liệt kê từng cờ. Muốn xem đầy đủ phải dùng:
>
> ```bash
> vllm bench serve --help=all
> ```

> **Client phải đủ mạnh.** Manifest xin 8 CPU vì ở concurrency 64 với output 1000 token, chính bench client có thể trở thành bottleneck và cho ra số liệu latency sai lệch. Nếu node của bạn eo hẹp CPU, hãy giảm mức concurrency cao nhất thay vì giảm CPU của client.

## Bước 6: Kiểm tra GPU

```bash
kubectl apply -f 05-gpu-check-job.yaml
kubectl logs -f job/gpu-check -n token-factory
kubectl delete job gpu-check -n token-factory
```

Kết quả mong đợi:

```
NVIDIA H100 80GB HBM3, 81559 MiB, 580.126.20, 9.0
```

Ghi lại con số bộ nhớ. Toàn bộ tính toán ngân sách ở các bài sau dựa trên **80 GB** (`81559 MiB` khả dụng); nếu bạn dùng H100 40GB hoặc H100 NVL, các giá trị `--gpu-memory-utilization` sẽ phải điều chỉnh.

Cột cuối (`9.0`) là compute capability — H100 là **SM90**. Con số này có liên quan ở [bài 03](../03-spec-decode-dspark/): tài liệu adaptive verification của vLLM tham chiếu các backend trên SM100, nên tính năng đó là mục **tuỳ chọn** phải tự kiểm chứng.

---

## Layout thư mục trọng số

`/mnt/hps/fp8_models` là **thư viện model dùng chung của cả tổ chức**, chứa hàng chục model. Nó tổ chức theo namespace HuggingFace, và các speculator nằm riêng dưới `speculators/`:

```
/mnt/hps/fp8_models/                      <- mount vào /models trong container
├── Qwen/
│   ├── Qwen3.8-27B-FP8/                  <-- TARGET (bài 01, 02, 03, 04)
│   ├── Qwen3.6-27B-FP8/
│   └── ...
├── RedHatAI/
├── MiniMaxAI/
└── speculators/
    ├── RedHatAI/
    │   ├── Qwen3.8-27B-speculator.dspark-preview/   <-- SPECULATOR (bài 03, 04)
    │   └── ...
    └── RadixArk/
```

Hai đường dẫn mà mọi manifest và lệnh benchmark dùng:

```
/models/Qwen/Qwen3.8-27B-FP8
/models/speculators/RedHatAI/Qwen3.8-27B-speculator.dspark-preview
```

### Bên trong target — hai điều đáng chú ý

```
Qwen3.8-27B-FP8/           (30.5 GB)
├── config.json
├── layers-0.safetensors ... layers-63.safetensors    <- tách theo lớp
├── outside.safetensors                               <- embedding + lm_head
├── mtp.safetensors        (477 MB)                   <- MTP head!
├── model.safetensors.index.json
└── tokenizer.json, chat_template.jinja, ...
```

1. **`mtp.safetensors` có sẵn** — đây chính là MTP head mà [bài 02](../02-spec-decode-mtp/) dùng. Không cần tải thêm gì.
2. **Trọng số tách theo từng lớp** (`layers-N.safetensors`) thay vì shard đều. Đây là layout tối ưu cho vLLM nạp song song, không phải layout HuggingFace chuẩn — đừng ngạc nhiên khi thấy 68 file safetensors.

### Bên trong speculator

```
Qwen3.8-27B-speculator.dspark-preview/    (7.6 GB tổng)
├── config.json           <- block_size: 8
├── config.py             <- cần --trust-remote-code
├── model.safetensors     (4.0 GB)  <- trọng số serving
├── optimizer_state_dict.pt (4.2 GB) <- artefact HUẤN LUYỆN, vLLM không đọc
└── scheduler_state_dict.pt
```

Hơn một nửa dung lượng thư mục là artefact huấn luyện mà vLLM không bao giờ đọc tới. Nhánh tự học ở nhà chỉ tải 3 file đầu.

### Nếu layout trên máy bạn khác

Job ở [Bước 4](#bước-4-xác-minh-trọng-số) sẽ in ra cây thư mục thật và báo lỗi. Có hai cách xử lý:

**Cách 1 — tạo symlink trên host** (khuyến nghị: sửa một lần, không đụng vào manifest):

```bash
# Ví dụ: trọng số thật đang nằm phẳng ở /mnt/hps/fp8_models/Qwen3.8-27B-FP8
cd /mnt/hps/fp8_models
sudo mkdir -p Qwen
sudo ln -s ../Qwen3.8-27B-FP8 Qwen/Qwen3.8-27B-FP8
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

Job xác minh ở Bước 4 sẽ tự `find /models -maxdepth 3 -iname "*Qwen3.8*"` và in ra các thư mục ứng viên khi thất bại, nên bạn không phải tự mò.

Và trong các README, mọi cờ `--tokenizer /models/Qwen/Qwen3.8-27B-FP8` của lệnh benchmark.

Kiểm tra còn sót chỗ nào:

```bash
grep -rn "/models/Qwen\|/models/speculators" . --include="*.yaml" --include="*.md"
```

---

**Tiếp theo:** [Bài 01 — Baseline vLLM aggregated mode](../01-baseline-agg/)
