"""Co hệ số OAS, ridge và dự đoán theo khoảng cách Mahalanobis.

Xem Docs/Instruction.md, mục 2.3 và 2.4.
"""
from __future__ import annotations

import torch


def oas_shrink(cov: torch.Tensor, n_samples: int) -> tuple[torch.Tensor, float]:
    """Co hiệp phương sai về `mu I` theo công thức OAS.

    Args:
        cov: Hiệp phương sai, kích thước `(D, D)`.
        n_samples: Số mẫu dùng để ước lượng `cov`.

    Returns:
        Cặp `(cov_oas, rho)`, trong đó `rho` là hệ số co.
    """
    raise NotImplementedError("Sprint 1")


def fit_lda(means: torch.Tensor, cov: torch.Tensor, lam: float) -> torch.Tensor:
    """Giải `(cov + lam I) w_c = mu_c` cho mọi lớp, không tính nghịch đảo tường minh.

    Dùng Cholesky, nếu lỗi thì dùng `lstsq`.

    Args:
        means: Trung bình lớp, kích thước `(C, D)`.
        cov: Hiệp phương sai đã co, kích thước `(D, D)`.
        lam: Hệ số ridge λ.

    Returns:
        Ma trận `w`, kích thước `(C, D)`.
    """
    raise NotImplementedError("Sprint 1")


def predict(phi: torch.Tensor, weights: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """Chọn lớp có điểm `φᵀw_c - ½ μ_cᵀw_c` lớn nhất.

    Args:
        phi: Đặc trưng, kích thước `(B, D)`.
        weights: Kết quả của `fit_lda`, kích thước `(C, D)`.
        means: Trung bình lớp, kích thước `(C, D)`.

    Returns:
        Nhãn dự đoán, kích thước `(B,)`.
    """
    raise NotImplementedError("Sprint 1")
