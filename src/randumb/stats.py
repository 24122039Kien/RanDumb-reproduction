"""Tích lũy trung bình lớp và hiệp phương sai dùng chung, không lưu mẫu.

Xem Docs/Instruction.md, mục 2.2.
"""
from __future__ import annotations

from pathlib import Path

import torch


class SufficientStats:
    """Giữ `A = ΣΦᵀΦ`, tổng và số mẫu theo lớp, cập nhật được theo từng lô.

    Args:
        dim: Số chiều D của đặc trưng.
        num_classes: Số lớp.
        device: Thiết bị tính toán.
        dtype: Kiểu số của các thống kê tích lũy.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        raise NotImplementedError("Sprint 1")

    def update(self, phi: torch.Tensor, y: torch.Tensor) -> None:
        """Cộng một lô đặc trưng vào các thống kê.

        Args:
            phi: Đặc trưng, kích thước `(B, D)`.
            y: Nhãn lớp, kích thước `(B,)`.
        """
        raise NotImplementedError("Sprint 1")

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tính trung bình lớp và hiệp phương sai trong lớp dùng chung.

        Returns:
            Cặp `(means, cov)` có kích thước `(C, D)` và `(D, D)`.
        """
        raise NotImplementedError("Sprint 1")

    def save(self, path: str | Path) -> None:
        """Lưu các thống kê ra file để E5 không phải tính lại.

        Args:
            path: Đường dẫn file đầu ra.
        """
        raise NotImplementedError("Sprint 1")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SufficientStats":
        """Nạp các thống kê đã lưu.

        Args:
            path: Đường dẫn file đã lưu.
            device: Thiết bị tính toán.

        Returns:
            Đối tượng `SufficientStats` đã có dữ liệu.
        """
        raise NotImplementedError("Sprint 1")
