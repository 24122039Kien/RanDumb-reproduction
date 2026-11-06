"""Kiểm thử tích lũy thống kê (Docs/Instruction.md, mục 2.6, test 2 và 3)."""
import tempfile
from pathlib import Path

import torch
import pytest

from randumb.stats import SufficientStats


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def _make_data(n: int = 120, d: int = 10, num_classes: int = 3, seed: int = 0):
    """Tạo dữ liệu ngẫu nhiên nhỏ để kiểm thử."""
    g = torch.Generator()
    g.manual_seed(seed)
    phi = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return phi, y


def _accumulate_all(phi: torch.Tensor, y: torch.Tensor, num_classes: int, batch_size: int = None) -> SufficientStats:
    """Tích lũy toàn bộ dữ liệu, theo từng mẫu (batch_size=1) hoặc lô lớn."""
    d = phi.shape[1]
    ss = SufficientStats(d, num_classes, dtype=torch.float64)
    bs = batch_size or phi.shape[0]
    for start in range(0, phi.shape[0], bs):
        ss.update(phi[start:start + bs], y[start:start + bs])
    return ss


# ---------------------------------------------------------------------------
# Test 2: Cập nhật từng mẫu và theo lô cho cùng μ_c và S
# ---------------------------------------------------------------------------
def test_online_matches_batch():
    """Test 2: cập nhật từng mẫu và theo lô cho cùng ``μ_c``, ``S``, trong dung sai số thực.

    Dung sai cần thiết vì thứ tự cộng số thực khác nhau (từng mẫu vs. lô lớn).
    """
    phi, y = _make_data(n=200, d=15, num_classes=4)

    ss_online = _accumulate_all(phi, y, num_classes=4, batch_size=1)
    ss_batch = _accumulate_all(phi, y, num_classes=4, batch_size=200)

    means_online, cov_online = ss_online.finalize()
    means_batch, cov_batch = ss_batch.finalize()

    atol = 1e-10
    assert torch.allclose(means_online, means_batch, atol=atol), (
        f"Trung bình lớp không khớp: max diff = {(means_online - means_batch).abs().max():.2e}"
    )
    assert torch.allclose(cov_online, cov_batch, atol=atol), (
        f"Hiệp phương sai không khớp: max diff = {(cov_online - cov_batch).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 3: Không phụ thuộc thứ tự mẫu
# ---------------------------------------------------------------------------
def test_order_invariance():
    """Test 3: xáo thứ tự mẫu không đổi kết quả, trong dung sai số thực."""
    phi, y = _make_data(n=150, d=12, num_classes=3)

    # Tạo hoán vị ngẫu nhiên
    perm = torch.randperm(phi.shape[0])
    phi_perm = phi[perm]
    y_perm = y[perm]

    ss_orig = _accumulate_all(phi, y, num_classes=3, batch_size=30)
    ss_perm = _accumulate_all(phi_perm, y_perm, num_classes=3, batch_size=30)

    means_orig, cov_orig = ss_orig.finalize()
    means_perm, cov_perm = ss_perm.finalize()

    atol = 1e-10
    assert torch.allclose(means_orig, means_perm, atol=atol), (
        f"Trung bình lớp phụ thuộc thứ tự: max diff = {(means_orig - means_perm).abs().max():.2e}"
    )
    assert torch.allclose(cov_orig, cov_perm, atol=atol), (
        f"Hiệp phương sai phụ thuộc thứ tự: max diff = {(cov_orig - cov_perm).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test save / load
# ---------------------------------------------------------------------------
def test_save_load_roundtrip():
    """Lưu và nạp lại cho cùng kết quả finalize."""
    phi, y = _make_data(n=80, d=8, num_classes=3)

    ss = _accumulate_all(phi, y, num_classes=3)
    means_orig, cov_orig = ss.finalize()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stats.pt"
        ss.save(path)
        ss2 = SufficientStats.load(path)

    means_loaded, cov_loaded = ss2.finalize()

    assert torch.allclose(means_orig, means_loaded), "Trung bình lớp sau load/save không khớp"
    assert torch.allclose(cov_orig, cov_loaded), "Hiệp phương sai sau load/save không khớp"
