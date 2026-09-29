
"""So hai bộ sweep theo từng mức concurrency. Dùng: python3 compare-spec.py <prefix-A> <prefix-B> [nhan-A] [nhan-B]"""
import glob
import json
import re
import sys


def curve(prefix):
    out = {}
    for f in glob.glob(f"/results/{prefix}-c*.json"):
        m = re.search(rf"{re.escape(prefix)}-c(\d+)\.json$", f)
        if not m:
            continue
        d = json.load(open(f))
        out[int(m.group(1))] = d
    return out


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    na = sys.argv[3] if len(sys.argv) > 3 else pa
    nb = sys.argv[4] if len(sys.argv) > 4 else pb
    A, B = curve(pa), curve(pb)
    shared = sorted(set(A) & set(B))
    if not shared:
        print(f"Khong co diem chung giua {pa} va {pb}")
        print(f"  {pa}: {sorted(A)}")
        print(f"  {pb}: {sorted(B)}")
        return

    print(f"A = {na}   B = {nb}\n")
    hdr = ("%4s | %9s %9s %6s | %7s %7s %6s | %7s %7s %6s | %9s %9s %6s"
           % ("c", f"A tok/s", f"B tok/s", "B/A",
              "A TPOT", "B TPOT", "B/A", "A ITL", "B ITL", "B/A",
              "A TTFT", "B TTFT", "B/A"))
    print(hdr)
    print("-" * len(hdr))
    for c in shared:
        a, b = A[c], B[c]
        at, bt = a["output_throughput"], b["output_throughput"]
        ap, bp = a["median_tpot_ms"], b["median_tpot_ms"]
        ai, bi = a["median_itl_ms"], b["median_itl_ms"]
        af, bf = a["median_ttft_ms"], b["median_ttft_ms"]
        print("%4d | %9.1f %9.1f %5.2fx | %7.2f %7.2f %5.2fx | %7.2f %7.2f %5.2fx | %9.0f %9.0f %5.2fx"
              % (c, at, bt, bt / at, ap, bp, bp / ap, ai, bi, bi / ai, af, bf, bf / af))

    print("\nCot B/A cua tok/s: >1 la B tot hon.")
    print("Cot B/A cua TPOT, ITL va TTFT: <1 la B tot hon (do tre thap hon).")


if __name__ == "__main__":
    main()
