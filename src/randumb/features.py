"""Biểu diễn ngẫu nhiên cố định: Random Fourier Features và RP+ReLU.

Xem Docs/Instruction.md, mục 2.1. Phép chiếu được lấy mẫu một lần rồi giữ nguyên.
"""
from __future__ import annotations

import torch


class RandomFourierFeatures:
    """Xấp xỉ RBF kernel bằng đặc trưng dạng D chiều `sqrt(2/D) cos(ωᵀx + b)`.

    Args:
        input_dim: Số chiều của vector đầu vào.
        output_dim: Số chiều D của đặc trưng.
        gamma: Tham số γ của kernel `exp(-γ‖x-y‖²)`.
        seed: Seed của `torch.Generator` trên CPU, để mọi máy có cùng phép chiếu.
        device: Thiết bị tính toán.
        dtype: Kiểu số của đặc trưng.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        gamma: float,
        seed: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        raise NotImplementedError("Sprint 1")

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Chiếu một lô dữ liệu lên không gian đặc trưng.

        Args:
            x: Tensor kích thước `(B, input_dim)`.

        Returns:
            Tensor kích thước `(B, output_dim)`.
        """
        raise NotImplementedError("Sprint 1")


class RPReLU:
    """Phép chiếu ngẫu nhiên kèm ReLU `max(0, xᵀW)`, dùng để so sánh ở E6.

    Args:
        input_dim: Số chiều của vector đầu vào.
        output_dim: Số chiều của đặc trưng.
        seed: Seed của `torch.Generator` trên CPU.
        device: Thiết bị tính toán.
        dtype: Kiểu số của đặc trưng.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        raise NotImplementedError("Sprint 1")

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Chiếu một lô dữ liệu lên không gian đặc trưng.

        Args:
            x: Tensor kích thước `(B, input_dim)`.

        Returns:
            Tensor kích thước `(B, output_dim)`.
        """
        raise NotImplementedError("Sprint 1")
