"""Đo peak VRAM của pipeline trên ảnh giả; không phải kết quả E1/E8."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from randumb.pipeline import PipelineConfig, fit_and_evaluate
from randumb.provenance import code_version
from randumb.tracking import environment_info, peak_ram_gb


def main():
    """Chạy một dim/shrink riêng, đo cấp phát và reserved; output ghi synthetic rõ ràng."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--shrink", choices=("oas", "oas_per_class", "none"), default="oas")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phép đo này cần CUDA")
    # D=25K cần tối thiểu hai ma trận D×D và omega; kiểm tra cận dưới trước.
    lower_bound = 2 * args.dim ** 2 * 8 + 3072 * args.dim * 8
    free, total = torch.cuda.mem_get_info()
    if lower_bound > free:
        raise RuntimeError(f"Không đủ VRAM: cận dưới {lower_bound / 2**30:.2f} GiB, còn {free / 2**30:.2f} GiB")
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(256, 32, 32, 3, generator=generator)
    x /= x.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    y = torch.arange(len(x)) % 10
    cfg = PipelineConfig(dim=args.dim, gamma=1.0, lam=1e-5, seed=0, flip=True,
                         feature_dtype=torch.float64, stats_dtype=torch.float64,
                         device="cuda", batch_size=128, shrink=args.shrink)
    torch.cuda.reset_peak_memory_stats()
    result = fit_and_evaluate(x, y, x[:20], y[:20], 10, cfg)
    torch.cuda.synchronize()
    report = {"kind": "synthetic_memory_check_not_E1_or_E8", "dim": args.dim, "shrink": args.shrink,
              "train_images": len(x), "n_train_seen": result["n_train_seen"], "batch_size": 128,
              "dtype": "float64", "device_total_gib": total / 2**30,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "peak_ram_gib": peak_ram_gb(),
              **{k: v for k, v in result.items() if k.startswith("time_")},
              **code_version(), **environment_info()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
