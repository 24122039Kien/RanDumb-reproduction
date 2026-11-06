"""Chạy một thực nghiệm RanDumb theo tham số dòng lệnh."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh.

    Returns:
        Các tham số của run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="khóa trong randumb.data.DATASETS")
    parser.add_argument("--dim", type=int, default=25_000, help="số chiều D của đặc trưng")
    parser.add_argument("--gamma", default="scale", help="số thực, hoặc 'scale'")
    parser.add_argument("--lam", type=float, required=True, help="hệ số ridge λ")
    parser.add_argument("--lam-mode", choices=["fixed", "relative"], default="fixed")
    parser.add_argument("--norm", choices=["zscore", "l2", "unit"], default="zscore")
    parser.add_argument("--shrink", choices=["oas", "oas_per_class", "none"], default="oas")
    parser.add_argument("--flip", action=argparse.BooleanOptionalAction, default=None,
                        help="mặc định theo DatasetSpec.use_flip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--group", required=True, help="nhóm thực nghiệm, ví dụ E1")
    parser.add_argument("--tag", action="append", default=[], help="thẻ W&B, lặp lại được")
    return parser.parse_args()


def main() -> None:
    """Chạy toàn bộ pipeline và log kết quả lên W&B."""
    parse_args()
    raise NotImplementedError("Sprint 1")


if __name__ == "__main__":
    main()
