"""Trích đặc trưng ViT-B/16 cho Benchmark F và lưu lại để khỏi trích lại."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh.

    Returns:
        Các tham số của lần trích đặc trưng.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", choices=["in1k", "in21k"], required=True)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                        help="có chuẩn hóa mean/std trước khi đưa vào ViT hay không")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, help="thư mục lưu đặc trưng")
    return parser.parse_args()


def main() -> None:
    """Trích đặc trưng, log tên mô hình và phiên bản `timm`."""
    parse_args()
    raise NotImplementedError("Sprint 2")


if __name__ == "__main__":
    main()
