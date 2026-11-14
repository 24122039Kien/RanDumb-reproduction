"""Trích đặc trưng ViT-B/16 cho một dataset E2 và lưu vào data/features/.

Ví dụ trên Kaggle:

    python scripts/extract_vit_features.py --dataset cifar100 --backbone in21k

Chạy lại sẽ bỏ qua nếu file đã có, trừ khi thêm --overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from randumb.vit_features import (  # noqa: E402
    BACKBONES, E2_DATASETS, create_backbone, extract_features, features_path, load_e2_splits, save_features,
)


def parse_args(argv=None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh.

    Args:
        argv: Danh sách tham số, ``None`` để đọc từ ``sys.argv``.

    Returns:
        Các tham số của lần trích đặc trưng.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(E2_DATASETS))
    parser.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    parser.add_argument("--model-name", default=None, help="ghi đè tên mô hình timm trong BACKBONES")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="--no-normalize để giữ ảnh ở [0, 1] như code gốc")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="file đầu ra, mặc định data/features/<dataset>-<backbone>-<norm|raw>.pt")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    """Trích đặc trưng train và test, lưu kèm thông tin mô hình.

    Args:
        argv: Danh sách tham số, ``None`` để đọc từ ``sys.argv``.

    Returns:
        Đường dẫn file đặc trưng.
    """
    args = parse_args(argv)
    out = Path(args.out) if args.out else features_path(args.dataset, args.backbone, args.normalize)
    if out.exists() and not args.overwrite:
        print(f"Đã có {out}, bỏ qua (thêm --overwrite để trích lại).")
        return out

    splits = load_e2_splits(args.dataset, args.hf_cache_dir)
    model, info = create_backbone(args.backbone, args.device, args.model_name)
    spec = E2_DATASETS[args.dataset]

    payload = {}
    start = time.perf_counter()
    for split, (dataset, indices, labels) in splits.items():
        payload[f"{split}_x"] = extract_features(
            model, dataset, indices, spec.image_column, args.batch_size, args.device,
            args.normalize, info["mean"], info["std"],
        )
        payload[f"{split}_y"] = labels
    payload["meta"] = {
        **info,
        "dataset": args.dataset,
        "hf_id": spec.hf_id,
        "normalize": args.normalize,
        "n_train": len(payload["train_y"]),
        "n_test": len(payload["test_y"]),
        "extract_seconds": time.perf_counter() - start,
        "torch_version": torch.__version__,
    }
    save_features(out, payload)
    print(json.dumps({"out": str(out), **payload["meta"]}, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
