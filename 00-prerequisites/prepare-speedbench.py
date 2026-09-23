"""Sinh SPEED-Bench jsonl cho chuỗi bài FPT Transform Token Factory.

Vì sao cần script này thay vì gọi thẳng prepare.py của NVIDIA:

  `throughput_8k` có 268/1536 bản ghi (17%) lấy nội dung từ `cais/hle`
  (Humanity's Last Exam) — một dataset GATED trên HuggingFace. Không có
  quyền thì prepare.py dừng hẳn:

      datasets.exceptions.DatasetNotFoundError:
      Dataset 'cais/hle' is a gated dataset on the Hub.

  Script này lọc bỏ các bản ghi thuộc nguồn gated TRƯỚC khi gọi phần
  resolve của NVIDIA, nên chạy được mà không cần xin quyền. Phần còn lại
  (~83%) vẫn giữ nguyên repobench_python/java — chính là dữ liệu code
  completion sát với workload agentic coding nhất.

  ĐỂ CÓ BẢN ĐẦY ĐỦ: xin quyền tại https://huggingface.co/datasets/cais/hle
  rồi chạy lại KHÔNG kèm --skip-gated.

Cách dùng:
  python3 prepare-speedbench.py \
      --config throughput_8k \
      --output_dir /datasets/speed-bench \
      --prepare /tmp/prepare.py \
      --skip-gated
"""
import argparse, importlib.util, sys
from pathlib import Path
from datasets import load_dataset

GATED = ["cais/hle"]

def load_prepare(path):
    spec = importlib.util.spec_from_file_location("nv_prepare", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["nv_prepare"] = m
    spec.loader.exec_module(m)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--prepare", type=Path, required=True)
    ap.add_argument("--skip-gated", action="store_true")
    a = ap.parse_args()

    nv = load_prepare(a.prepare)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("nvidia/SPEED-Bench", a.config, split="test")
    total = len(ds)
    if a.skip_gated:
        ds = ds.filter(lambda ex: not any(g in ex["source"] for g in GATED))
        print(f"[skip-gated] bỏ {total - len(ds)}/{total} bản ghi từ {GATED}; còn {len(ds)}")

    ds = nv._resolve_external_data(ds, a.config)
    ds = ds.map(
        lambda ex: {"messages": [{"role": "user", "content": t} for t in ex["turns"]]},
        remove_columns=["turns"],
    )
    out = a.output_dir / f"{a.config}.jsonl"
    ds.to_json(out)
    print(f"đã ghi {out} ({len(ds)} bản ghi)")

if __name__ == "__main__":
    main()
