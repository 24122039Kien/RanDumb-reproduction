"""Kiểm thử OAS và dự đoán (Docs/Instruction.md, mục 2.6, test 4 và 5)."""
import math

import torch
import pytest
from sklearn.covariance import OAS

from randumb.classifier import oas_shrink, fit_lda, predict
from randumb.stats import SufficientStats


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def _sample_cov(n: int, d: int, seed: int = 0) -> tuple[torch.Tensor, int]:
    """Trả về hiệp phương sai mẫu (d×d) trên n mẫu ngẫu nhiên."""
    g = torch.Generator()
    g.manual_seed(seed)
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    # Hiệp phương sai mẫu (chia N, không chia N-1, nhất quán với stats.py)
    mu = X.mean(dim=0, keepdim=True)
    cov = ((X - mu).T @ (X - mu)) / n
    return cov, n


# ---------------------------------------------------------------------------
# Test 4: OAS khớp sklearn.covariance.OAS trên dữ liệu một lớp
# ---------------------------------------------------------------------------
def test_oas_matches_sklearn():
    """Test 4: ``oas_shrink`` khớp ``sklearn.covariance.OAS`` trên dữ liệu một lớp.

    Lưu ý: sklearn dùng ``store_precision=True`` mặc định và trừ trung bình
    toàn cục, còn ``oas_shrink`` nhận vào hiệp phương sai trong lớp đã tính
    sẵn.  Ta so sánh hệ số ``rho`` và ma trận kết quả.
    """
    torch.manual_seed(99)
    n, d = 200, 15

    g = torch.Generator()
    g.manual_seed(99)
    X = torch.randn(n, d, generator=g, dtype=torch.float64)

    # sklearn OAS (dùng để đối chiếu)
    oas_sk = OAS()
    oas_sk.fit(X.numpy())
    rho_sk = float(oas_sk.shrinkage_)

    # Hiệp phương sai mẫu (chia N, nhất quán với finalize() trong stats.py)
    mu = X.mean(dim=0, keepdim=True)
    cov = ((X - mu).T @ (X - mu)) / n

    cov_oas, rho = oas_shrink(cov, n_samples=n)

    # Hệ số co phải khớp trong dung sai nhỏ
    assert abs(rho - rho_sk) < 1e-6, f"rho lệch: nhóm={rho:.6f}, sklearn={rho_sk:.6f}"

    # Ma trận kết quả phải đối xứng và xác định dương
    assert (cov_oas - cov_oas.T).abs().max() < 1e-12, "Ma trận OAS không đối xứng"
    eigvals = torch.linalg.eigvalsh(cov_oas)
    assert eigvals.min() > 0, f"Ma trận OAS không xác định dương, eigval min = {eigvals.min():.4e}"


# ---------------------------------------------------------------------------
# Test 5: predict khớp tính khoảng cách Mahalanobis đầy đủ
# ---------------------------------------------------------------------------
def test_predict_matches_full_mahalanobis():
    """Test 5: ``predict`` khớp phép tính khoảng cách Mahalanobis đầy đủ, viết chậm.

    Phiên bản chậm (tham chiếu) tính S⁻¹ tường minh rồi đo khoảng cách.
    Phiên bản nhanh dùng ``fit_lda`` + ``predict``.
    """
    torch.manual_seed(5)

    C, D, B = 5, 20, 30
    lam = 1e-3

    g = torch.Generator()
    g.manual_seed(5)
    means = torch.randn(C, D, generator=g, dtype=torch.float64)

    # Tạo ma trận hiệp phương sai ngẫu nhiên xác định dương
    A = torch.randn(D, D, generator=g, dtype=torch.float64)
    cov_raw = A.T @ A / D + lam * torch.eye(D, dtype=torch.float64)

    phi = torch.randn(B, D, generator=g, dtype=torch.float64)

    # --- Phiên bản nhanh ---
    weights = fit_lda(means, cov_raw, lam=0.0)   # cov đã cộng lam rồi
    preds_fast = predict(phi, weights, means)

    # --- Phiên bản chậm (tham chiếu): tính S⁻¹ tường minh ---
    S_inv = torch.linalg.inv(cov_raw)
    scores_ref = torch.zeros(B, C, dtype=torch.float64)
    for b in range(B):
        for c in range(C):
            diff = phi[b] - means[c]  # (D,)
            scores_ref[b, c] = -0.5 * (diff @ S_inv @ diff)
    preds_slow = scores_ref.argmax(dim=1)

    assert torch.equal(preds_fast, preds_slow), (
        f"Dự đoán không khớp: {(preds_fast != preds_slow).sum().item()} trong {B} mẫu"
    )


# ---------------------------------------------------------------------------
# Test bổ sung: pipeline đầu cuối nhỏ
# ---------------------------------------------------------------------------
def test_full_pipeline_smoke():
    """Pipeline nhỏ: SufficientStats → oas_shrink → fit_lda → predict."""
    torch.manual_seed(42)

    C, D, N_train, N_test = 3, 50, 300, 60
    lam = 1e-4

    g = torch.Generator()
    g.manual_seed(42)
    phi_train = torch.randn(N_train, D, generator=g, dtype=torch.float64)
    y_train = torch.randint(0, C, (N_train,), generator=g)
    phi_test = torch.randn(N_test, D, generator=g, dtype=torch.float64)

    ss = SufficientStats(D, C, dtype=torch.float64)
    ss.update(phi_train, y_train)
    means, cov = ss.finalize()

    cov_oas, _ = oas_shrink(cov, n_samples=N_train)
    weights = fit_lda(means, cov_oas, lam=lam)
    preds = predict(phi_test, weights, means)

    assert preds.shape == (N_test,), f"Shape dự đoán sai: {preds.shape}"
    assert preds.min() >= 0 and preds.max() < C, "Nhãn dự đoán ngoài phạm vi"
