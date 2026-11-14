"""Co hệ số OAS, ridge và dự đoán theo khoảng cách Mahalanobis.

Xem Docs/Instruction.md, mục 2.3 và 2.4.

Công thức OAS (Oracle Approximating Shrinkage):
    mu    = trace(S) / p
    alpha = mean(S_ij²)
    rho   = min(1, (alpha + mu²) / ((n + 1)(alpha - mu²/p)))
    S_oas = (1 - rho) S + rho mu I

Dự đoán theo điểm Mahalanobis rút gọn:
    Giải (S_oas + λI) w_c = μ_c bằng Cholesky (nếu lỗi thì dùng lstsq)
    score_c = φᵀw_c - ½ μ_cᵀw_c
    ŷ = argmax_c score_c
"""
from __future__ import annotations

import torch


def oas_shrink(cov: torch.Tensor, n_samples: int, *, inplace: bool = False) -> tuple[torch.Tensor, float]:
    """Co hiệp phương sai về ``mu I`` theo công thức OAS.

    Xử lý trường hợp mẫu số bằng 0 bằng cách trả về ``rho = 1`` (co hoàn toàn).

    Args:
        cov:       Hiệp phương sai, kích thước ``(D, D)``.
        n_samples: Số mẫu dùng để ước lượng ``cov``.
        inplace: Cho phép sửa covariance đầu vào để tránh bản sao D×D.

    Returns:
        Cặp ``(cov_oas, rho)``, trong đó ``rho ∈ [0, 1]`` là hệ số co.
    """
    p = cov.shape[0]
    if n_samples < 1:
        raise ValueError("n_samples phải dương")

    # mu = trace(S) / p
    mu = cov.trace().double() / p

    # alpha = mean(S_ij²)
    # Norm reduction không tạo thêm ma trận cov**2. Bình phương scalar ở float64.
    alpha = torch.linalg.vector_norm(cov).double().square() / (p * p)

    if not torch.isfinite(alpha) or not torch.isfinite(mu):
        raise ValueError("Covariance chứa giá trị không hữu hạn")
    # Chia cả tử/mẫu cho alpha: không dùng ngưỡng tuyệt đối làm mất tính
    # bất biến của rho khi đổi thang đo covariance.
    if alpha == 0:
        rho = 1.0
    else:
        ratio = mu.square() / alpha
        denominator = (n_samples + 1) * (1 - ratio / p)
        rho = 1.0 if denominator <= 0 else float(((1 + ratio) / denominator).clamp(0.0, 1.0))

    cov_oas = cov if inplace else cov.clone()
    cov_oas.mul_(1.0 - rho)
    cov_oas.diagonal().add_(rho * mu)
    return cov_oas, rho


def fit_lda(means: torch.Tensor, cov: torch.Tensor, lam: float, *, inplace: bool = False) -> torch.Tensor:
    """Giải ``(cov + lam I) w_c = mu_c`` cho mọi lớp, không tính nghịch đảo tường minh.

    Dùng phân tích Cholesky; nếu ma trận không xác định dương thì fallback sang
    ``torch.linalg.lstsq``.

    Args:
        means: Trung bình lớp, kích thước ``(C, D)``.
        cov:   Hiệp phương sai đã co, kích thước ``(D, D)``.
        lam:   Hệ số ridge λ ≥ 0.
        inplace: Cho phép cộng ridge trực tiếp vào covariance đầu vào.

    Returns:
        Ma trận ``w``, kích thước ``(C, D)``.
    """
    if lam < 0:
        raise ValueError("lam phải không âm")
    A = cov if inplace else cov.clone()
    A.diagonal().add_(lam)

    try:
        # Cholesky: A = LLᵀ, giải Lz = μ_cᵀ rồi Lᵀw_c = z
        L = torch.linalg.cholesky(A)                 # (D, D)
        # means: (C, D) → cần giải A w_cᵀ = mu_cᵀ
        # torch.cholesky_solve nhận B dạng (D, C)
        rhs = means.T.contiguous()                   # (D, C)
        weights_T = torch.cholesky_solve(rhs, L)     # (D, C)
        weights = weights_T.T                        # (C, D)
    except torch.linalg.LinAlgError:
        # Fallback lstsq
        rhs = means.T.contiguous()                   # (D, C)
        result = torch.linalg.lstsq(A, rhs)
        weights = result.solution.T                  # (C, D)

    return weights


def predict(phi: torch.Tensor, weights: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """Chọn lớp có điểm ``φᵀw_c - ½ μ_cᵀw_c`` lớn nhất.

    Điểm này tương đương với âm khoảng cách Mahalanobis (sau khi bỏ số hạng
    ``φᵀS⁻¹φ`` giống nhau ở mọi lớp).

    Args:
        phi:     Đặc trưng, kích thước ``(B, D)``.
        weights: Kết quả của :py:func:`fit_lda`, kích thước ``(C, D)``.
        means:   Trung bình lớp, kích thước ``(C, D)``.

    Returns:
        Nhãn dự đoán ``long``, kích thước ``(B,)``.
    """
    # φ_b ᵀ w_c → (B, C)
    phi_w = phi @ weights.T                         # (B, C)

    # ½ μ_cᵀ w_c → (C,)
    half_mu_w = 0.5 * (means * weights).sum(dim=1)  # (C,)

    scores = phi_w - half_mu_w.unsqueeze(0)         # (B, C)
    return scores.argmax(dim=1)
