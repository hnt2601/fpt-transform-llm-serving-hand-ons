# Bài 00 — Chuẩn bị môi trường

Dựng nền cho toàn bộ chuỗi bài: namespace, secret HuggingFace, storage, trọng
số model, dataset benchmark và pod `bench-client`.

Có **hai nhánh**, chỉ khác nhau ở Bước 3:

| Nhánh | Khi nào dùng |
|---|---|
| **A — Workshop** | Trọng số đã có sẵn tại `/mnt/hps/fp8_models` trên node |
| **B — Tự học** | Tự tải trọng số từ HuggingFace về PVC |

## Mục lục

1. [Chuẩn bị](#1-chuẩn-bị)
2. [Bước 1 — Namespace](#bước-1--namespace)
3. [Bước 2 — Secret HF_TOKEN](#bước-2--secret-hf_token)
4. [Bước 3A — Storage (workshop)](#bước-3a--storage-workshop)
5. [Bước 3B — Storage + tải trọng số (tự học)](#bước-3b--storage--tải-trọng-số-tự-học)
6. [Bước 4 — Xác minh trọng số](#bước-4--xác-minh-trọng-số)
7. [Bước 5 — Dataset benchmark](#bước-5--dataset-benchmark)
8. [Bước 6 — Bench client](#bước-6--bench-client)
9. [Bước 7 — Thư viện offline cho sweep job](#bước-7--thư-viện-offline-cho-sweep-job)
10. [Bước 8 — Kiểm tra GPU](#bước-8--kiểm-tra-gpu)
11. [Xử lý sự cố](#xử-lý-sự-cố)

---

## 1. Chuẩn bị

```bash
git clone https://github.com/hnt2601/fpt-transform-llm-serving-hand-ons.git
cd fpt-transform-llm-serving-hand-ons/00-prerequisites
```

Mọi lệnh trong bài này chạy từ thư mục `00-prerequisites/`.

Kiểm tra cluster thấy GPU:

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
```

Nhu cầu GPU của từng bài:

| Bài | GPU |
|---|---|
| 01, 02 | 1 |
| 03 — bản `deployment-1gpu.yaml` | 1 |
| 03 — bản `deployment-2gpu.yaml` | 2 (cùng một node) |
| 04 | 0 |

Phiên bản dùng xuyên suốt:

| Thành phần | Giá trị |
|---|---|
| vLLM | `vllm/vllm-openai:v0.29.0` |
| Model | `RedHatAI/gemma-4-12B-it-FP8-Dynamic` |
| Drafter DSpark | `deepseek-ai/dspark_gemma4_12b_block7` |

## Bước 1 — Namespace

```bash
kubectl get ns token-factory >/dev/null 2>&1 || kubectl apply -f 00-namespace.yaml
kubectl config set-context --current --namespace=token-factory
kubectl get pods,deploy,svc,job,pvc -n token-factory
```

> Nhánh workshop: namespace đã được cấp sẵn. Nếu `kubectl apply` báo
> `Forbidden ... cannot patch resource "namespaces"` thì bỏ qua — namespace
> đã dùng được.

## Bước 2 — Secret HF_TOKEN

```bash
kubectl create secret generic hf-token \
  --from-literal=token="hf_xxxxxxxxxxxxxxxxxxxx" \
  -n token-factory
```

Không commit token vào git. `01-hf-secret.yaml` chỉ là template tham khảo.

## Bước 3A — Storage (workshop)

```bash
kubectl apply -f 02-storage-workshop.yaml
kubectl get pvc -n token-factory
```

Cả ba PVC `bench-results`, `model-cache`, `vllm-cache` phải ở trạng thái
`Bound`. Sang [Bước 4](#bước-4--xác-minh-trọng-số).

## Bước 3B — Storage + tải trọng số (tự học)

Sửa `storageClassName` trong `02-storage-home.yaml` cho khớp cluster
(`kubectl get sc`), rồi:

```bash
kubectl apply -f 02-storage-home.yaml
kubectl get pvc -n token-factory

kubectl apply -f 03-model-download-job.yaml
kubectl wait --for=condition=ready pod -l job-name=model-download -n token-factory --timeout=300s
kubectl logs -f job/model-download -n token-factory
```

Chờ tới dòng `==> Download completed`, rồi dọn job:

```bash
kubectl wait --for=condition=complete job/model-download -n token-factory --timeout=3600s
kubectl delete job model-download -n token-factory
```

Job tải hai repo:

| Repo | Dùng ở bài |
|---|---|
| `RedHatAI/gemma-4-12B-it-FP8-Dynamic` | 01, 02, 03 |
| `deepseek-ai/dspark_gemma4_12b_block7` | 02, 03 |

## Bước 4 — Xác minh trọng số

Chạy ở **cả hai nhánh**:

```bash
kubectl apply -f 06-verify-models-job.yaml
kubectl wait --for=condition=complete job/verify-models -n token-factory --timeout=180s || true
kubectl logs job/verify-models -n token-factory
kubectl delete job verify-models -n token-factory
```

Kết quả mong đợi:

```
===== [1/2] Target model =====
Mong đợi: /models/RedHatAI/gemma-4-12B-it-FP8-Dynamic
  OK     config.json
  OK     model.safetensors
  OK     processor_config.json
  OK     tokenizer.json
  ...
===== [2/2] DSpark drafter =====
Mong đợi: /models/speculators/deepseek-ai/dspark_gemma4_12b_block7
  OK     config.json
  OK     model.safetensors
  "block_size": 7

==================================================
DAT — trọng số đã sẵn sàng. Tiếp tục sang bài 01.
==================================================
```

Nếu báo `KHONG DAT`, log sẽ liệt kê các thư mục ứng viên dưới `/models`.

> Đừng chạy `kubectl logs -f` ngay sau `kubectl apply`: container chưa kịp
> khởi động sẽ báo `ContainerCreating`. Luôn `wait` rồi mới `logs`.

## Bước 5 — Dataset benchmark

Chuỗi bài đo bằng [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench),
subset `throughput_8k`.

```bash
kubectl apply -f 07-dataset-prep-job.yaml
kubectl wait --for=condition=complete job/dataset-prep -n token-factory --timeout=3600s || true
kubectl logs job/dataset-prep -n token-factory --tail=20
kubectl delete job dataset-prep -n token-factory
```

Job có thể chạy 10–30 phút, đừng huỷ giữa chừng. Nếu dataset đã có sẵn, job
báo `ĐÃ CÓ SẴN, bỏ qua bước tải` và thoát ngay.

Tải thêm subset khác (tuỳ chọn):

```bash
kubectl apply -f 07-dataset-prep-job.yaml --dry-run=client -o yaml \
  | sed 's/name: dataset-prep/name: dataset-prep-32k/' \
  | kubectl apply -f -
kubectl set env job/dataset-prep-32k SUBSET=throughput_32k -n token-factory
```

### Nếu mạng cluster quá chậm: tải ở máy ngoài rồi copy vào PVC

```bash
python3 -m venv .venv && .venv/bin/pip install datasets pandas tiktoken numpy
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py -o prepare.py
.venv/bin/python prepare-speedbench.py --config throughput_8k --output_dir out --prepare prepare.py --skip-gated

kubectl apply -f 08-dataset-uploader-pod.yaml
kubectl wait --for=condition=ready pod/dataset-uploader -n token-factory --timeout=120s
kubectl cp out/throughput_8k.jsonl token-factory/dataset-uploader:/datasets/speed-bench/throughput_8k.jsonl
kubectl exec dataset-uploader -n token-factory -- ls -la /datasets/speed-bench/
kubectl delete pod dataset-uploader -n token-factory
```

## Bước 6 — Bench client

```bash
kubectl apply -f 04-bench-client.yaml
kubectl wait --for=condition=ready pod -l app=bench-client -n token-factory --timeout=300s
```

Kiểm tra:

```bash
kubectl exec deploy/bench-client -n token-factory -- bash -c '
  python3 -c "import vllm; print(\"vLLM\", vllm.__version__)"
  ls $MODEL_PATH/config.json
  touch /results/.probe && echo "/results ghi được" && rm /results/.probe
  ls $DATASET_DIR/throughput_8k.jsonl
'
```

Mong đợi: `vLLM 0.29.0`, đường dẫn `config.json` của model, `/results ghi
được`, và file `throughput_8k.jsonl`.

## Bước 7 — Thư viện offline cho sweep job

Sweep job của bài 01–03 cần `pandas`. Node workshop có thể không ra được PyPI,
nên chép `pandas` từ `bench-client` vào PVC `bench-results` **một lần**:

```bash
kubectl exec -n token-factory deploy/bench-client -- bash -c '
set -e
D=/usr/local/lib/python3.12/dist-packages
mkdir -p /results/pylibs
for p in pandas pytz dateutil six.py tzdata \
         pandas-*.dist-info pytz-*.dist-info \
         python_dateutil-*.dist-info six-*.dist-info tzdata-*.dist-info; do
  for m in $D/$p; do [ -e "$m" ] && cp -r "$m" /results/pylibs/ || true; done
done
ls /results/pylibs
'

kubectl exec -n token-factory deploy/bench-client -- \
  env PYTHONPATH=/results/pylibs python3 -c 'import pandas; print(pandas.__version__)'
```

## Bước 8 — Kiểm tra GPU

```bash
kubectl apply -f 05-gpu-check-job.yaml
kubectl wait --for=condition=complete job/gpu-check -n token-factory --timeout=180s || true
kubectl logs job/gpu-check -n token-factory
kubectl delete job gpu-check -n token-factory
```

Mong đợi một dòng dạng `NVIDIA H100 80GB HBM3, <bộ nhớ> MiB, <driver>, 9.0`.
Compute capability `9.0` = SM90.

---

## Xử lý sự cố

| Triệu chứng | Kiểm tra |
|---|---|
| PVC ở `Pending` | `kubectl describe pvc <tên>` |
| Pod kẹt `ContainerCreating` lâu | `kubectl describe pod <tên>` → mục `Events`, tìm `FailedAttachVolume` |
| `verify-models` báo `KHONG DAT` | Đọc danh sách thư mục ứng viên trong log, sửa đường dẫn trong manifest |
| Job `dataset-prep` báo `FSTimeoutError` | Chạy lại job — cache trên PVC giữ phần đã tải |
| Job gọi `pip install` treo ở `Running` | Node không ra được PyPI — làm [Bước 7](#bước-7--thư-viện-offline-cho-sweep-job) |
| Nghi hai benchmark chạy chồng nhau | `kubectl get jobs -n token-factory` — chỉ chạy **một** sweep trên mỗi server |

---

**Tiếp theo:** [Bài 01 — Baseline agg](../01-gemma4-baseline-agg/README.md)
