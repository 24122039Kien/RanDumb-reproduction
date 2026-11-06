"""Tích lũy trung bình lớp và hiệp phương sai dùng chung, không lưu mẫu.

Xem Docs/Instruction.md, mục 2.2.

Công thức tích lũy theo lô (giống kết quả với từng mẫu về mặt toán học):

    Khởi tạo: A = 0 (D×D), tong[c] = 0, dem[c] = 0, N = 0
    Với mỗi lô Φ (B×D) có nhãn y:
        A   += ΦᵀΦ
        tong[c] += tổng φ của lớp c
        dem[c]  += số mẫu lớp c
        N   += B
    μ_c = tong[c] / dem[c]
    S   = (A - Σ_c dem[c] μ_c μ_cᵀ) / N
"""
from __future__ import annotations

from pathlib import Path

import torch


class SufficientStats:
    """Giữ ``A = ΣΦᵀΦ``, tổng và số mẫu theo lớp, cập nhật được theo từng lô.

    Không lưu mẫu: chỉ giữ ma trận ``A`` kích thước ``(D, D)``.  Dùng ``float64``
    mặc định vì ta cộng dồn nhiều số rất nhỏ.

    Args:
        dim: Số chiều ``D`` của đặc trưng.
        num_classes: Số lớp ``C``.
        device: Thiết bị tính toán.
        dtype: Kiểu số của các thống kê tích lũy (nên dùng ``float64``).
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.dim = dim
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype

        self._A = torch.zeros(dim, dim, device=device, dtype=dtype)          # ΣΦᵀΦ
        self._tong = torch.zeros(num_classes, dim, device=device, dtype=dtype)  # tổng φ theo lớp
        self._dem = torch.zeros(num_classes, device=device, dtype=dtype)     # số mẫu theo lớp
        self._N = 0  # tổng số mẫu

    def update(self, phi: torch.Tensor, y: torch.Tensor) -> None:
        """Cộng một lô đặc trưng vào các thống kê.

        Args:
            phi: Đặc trưng, kích thước ``(B, D)``.
            y:   Nhãn lớp ``long``, kích thước ``(B,)``.
        """
        phi = phi.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device)

        # A += ΦᵀΦ
        self._A += phi.T @ phi

        # Cộng vào tổng và đếm theo lớp
        for c in range(self.num_classes):
            mask = y == c
            if mask.any():
                self._tong[c] += phi[mask].sum(dim=0)
                self._dem[c] += mask.sum().to(self.dtype)

        self._N += phi.shape[0]

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tính trung bình lớp và hiệp phương sai trong lớp dùng chung.

        Returns:
            Cặp ``(means, cov)`` có kích thước ``(C, D)`` và ``(D, D)``.

        Raises:
            RuntimeError: Nếu chưa gọi ``update`` lần nào.
        """
        if self._N == 0:
            raise RuntimeError("Chưa có mẫu nào được tích lũy.")

        # μ_c = tong[c] / dem[c], bảo vệ trường hợp lớp không có mẫu
        dem_safe = self._dem.clamp(min=1.0)
        means = self._tong / dem_safe.unsqueeze(1)  # (C, D)

        # S = (A - Σ_c dem[c] μ_c μ_cᵀ) / N
        correction = torch.zeros_like(self._A)
        for c in range(self.num_classes):
            if self._dem[c] > 0:
                mu_c = means[c].unsqueeze(1)  # (D, 1)
                correction += self._dem[c] * (mu_c @ mu_c.T)

        cov = (self._A - correction) / self._N
        return means, cov

    def save(self, path: str | Path) -> None:
        """Lưu các thống kê ra file để E5 không phải tính lại.

        Args:
            path: Đường dẫn file đầu ra (thường dùng đuôi ``.pt``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dim": self.dim,
                "num_classes": self.num_classes,
                "dtype_str": str(self.dtype),
                "A": self._A.cpu(),
                "tong": self._tong.cpu(),
                "dem": self._dem.cpu(),
                "N": self._N,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SufficientStats":
        """Nạp các thống kê đã lưu.

        Args:
            path:   Đường dẫn file đã lưu bằng :py:meth:`save`.
            device: Thiết bị tính toán.

        Returns:
            Đối tượng ``SufficientStats`` đã có dữ liệu, sẵn sàng để ``finalize()``.
        """
        data = torch.load(path, map_location="cpu", weights_only=True)

        # Khôi phục dtype từ chuỗi lưu, mặc định float64
        _DTYPE_MAP = {
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
        }
        dtype = _DTYPE_MAP.get(data.get("dtype_str", ""), torch.float64)

        obj = cls(
            dim=data["dim"],
            num_classes=data["num_classes"],
            device=device,
            dtype=dtype,
        )
        obj._A = data["A"].to(device=device, dtype=dtype)
        obj._tong = data["tong"].to(device=device, dtype=dtype)
        obj._dem = data["dem"].to(device=device, dtype=dtype)
        obj._N = int(data["N"])
        return obj
