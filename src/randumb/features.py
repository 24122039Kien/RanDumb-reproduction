"""Biểu diễn ngẫu nhiên cố định: Random Fourier Features và RP+ReLU.

Xem Docs/Instruction.md, mục 2.1. Phép chiếu được lấy mẫu một lần rồi giữ nguyên.

Lưu ý về công thức:
- Bài báo viết φ(x) có 2D chiều (cos + sin). Nhưng bài cũng nói dùng RFF dạng D chiều
  của scikit-learn: ``sqrt(2/D) cos(ωᵀx + b)`` với ``b ~ U(0, 2π)``.
- Nhóm chọn dạng D chiều để khớp "25K embedding" trong bài.
  Mâu thuẫn này sẽ được ghi vào phân tích sai lệch.
"""
from __future__ import annotations

import math

import torch


class RandomFourierFeatures:
    """Xấp xỉ RBF kernel bằng đặc trưng dạng D chiều ``sqrt(2/D) cos(ωᵀx + b)``.

    Lấy mẫu ``ω ~ N(0, 2γ·I)`` và ``b ~ U(0, 2π)`` **một lần** trên CPU rồi giữ
    nguyên suốt quá trình học.  Dùng ``torch.Generator`` được ghim seed để mọi máy
    cho cùng phép chiếu khi dùng cùng seed.

    Args:
        input_dim: Số chiều ``d`` của vector đầu vào.
        output_dim: Số chiều ``D`` của đặc trưng đầu ra.
        gamma: Tham số γ của kernel ``exp(-γ‖x-y‖²)``.
        seed: Seed của ``torch.Generator`` trên CPU.
        device: Thiết bị tính toán (``"cpu"`` hoặc ``"cuda"``).
        dtype: Kiểu số của đặc trưng đầu ra.
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
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.device = device
        self.dtype = dtype

        # Lấy mẫu trên CPU với Generator cố định, rồi chuyển sang device.
        # Điều này đảm bảo cùng seed → cùng phép chiếu trên mọi máy.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        # ω ~ N(0, 2γ·I), kích thước (input_dim, output_dim)
        self._omega = torch.randn(input_dim, output_dim, generator=gen, dtype=dtype).to(device) * math.sqrt(2.0 * gamma)

        # b ~ U(0, 2π), kích thước (output_dim,)
        self._bias = torch.rand(output_dim, generator=gen, dtype=dtype).to(device) * (2.0 * math.pi)

        # Hệ số chuẩn hóa
        self._scale = math.sqrt(2.0 / output_dim)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Chiếu một lô dữ liệu lên không gian đặc trưng.

        Args:
            x: Tensor kích thước ``(B, input_dim)``.

        Returns:
            Tensor kích thước ``(B, output_dim)``, kiểu ``self.dtype``.
        """
        x = x.to(device=self.device, dtype=self.dtype)
        # (B, D)
        projection = x @ self._omega + self._bias  # broadcast bias
        return self._scale * torch.cos(projection)


class RPReLU:
    """Phép chiếu ngẫu nhiên kèm ReLU ``max(0, xᵀW)``, dùng để so sánh ở E6.

    ``W ~ N(0, I)`` được chuẩn hóa cột để có chuẩn đơn vị.

    Args:
        input_dim: Số chiều của vector đầu vào.
        output_dim: Số chiều của đặc trưng.
        seed: Seed của ``torch.Generator`` trên CPU.
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
        self.device = device
        self.dtype = dtype

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        # W ~ N(0, I), kích thước (input_dim, output_dim)
        self._weight = torch.randn(input_dim, output_dim, generator=gen, dtype=dtype).to(device)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Chiếu một lô dữ liệu lên không gian đặc trưng.

        Args:
            x: Tensor kích thước ``(B, input_dim)``.

        Returns:
            Tensor kích thước ``(B, output_dim)``, kiểu ``self.dtype``.
        """
        x = x.to(device=self.device, dtype=self.dtype)
        return torch.relu(x @ self._weight)
