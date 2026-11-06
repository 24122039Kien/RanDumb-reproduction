"""Kiểm thử RFF và seed (Docs/Instruction.md, mục 2.6, test 1 và 6)."""
import math

import torch
import pytest

from randumb.features import RandomFourierFeatures, RPReLU


# ---------------------------------------------------------------------------
# Test 1: φ(x)ᵀφ(y) → exp(-γ‖x-y‖²) khi D tăng
# ---------------------------------------------------------------------------
def test_rff_approximates_rbf_kernel():
    """Test 1: ``φ(x)ᵀφ(y)`` tiến gần ``exp(-γ‖x-y‖²)`` khi D tăng.

    Luật số lớn đảm bảo sai số giảm theo sqrt(D).  Ta kiểm tra ở D = 50_000
    với dung sai tuyệt đối 0.02 (đủ chặt vì kernel ∈ [0,1]).

    Chú ý: dùng vector x, y gần nhau (‖x-y‖ nhỏ) để kernel_exact không gần 0,
    tránh trường hợp dung sai tương đối cho phép sai số lớn tuyệt đối.
    """
    d = 10
    gamma = 0.1   # gamma nhỏ → kernel không quá nhỏ dù x,y xa nhau
    seed = 42

    # Tạo x, y xác định — không dùng torch.manual_seed để tránh phụ thuộc thứ tự
    g = torch.Generator()
    g.manual_seed(0)
    x = torch.randn(d, generator=g, dtype=torch.float64)
    y = x + 0.5 * torch.randn(d, generator=g, dtype=torch.float64)   # gần x

    # Giá trị RBF kernel thật
    kernel_exact = math.exp(-gamma * ((x - y) ** 2).sum().item())
    # kernel_exact phải lớn hơn 0.1 để test có ý nghĩa
    assert kernel_exact > 0.1, f"kernel_exact quá nhỏ ({kernel_exact:.4f}), cần điều chỉnh tham số"

    errors = []
    for D in [1_000, 5_000, 20_000, 50_000]:
        rff = RandomFourierFeatures(d, D, gamma, seed=seed, dtype=torch.float64)
        phi_x = rff.transform(x.unsqueeze(0)).squeeze(0)   # (D,)
        phi_y = rff.transform(y.unsqueeze(0)).squeeze(0)   # (D,)
        approx = (phi_x * phi_y).sum().item()
        errors.append(abs(approx - kernel_exact))

    # Sai số tại D = 50_000 phải nhỏ hơn 2 % + dung sai tuyệt đối nhỏ
    tol = 0.02 * kernel_exact + 0.005
    assert errors[-1] < tol, (
        f"Sai số RFF tại D=50000: {errors[-1]:.4f}, kernel={kernel_exact:.4f}, tol={tol:.4f}"
    )

    # Sai số phải có xu hướng giảm (không yêu cầu đơn điệu tuyệt đối)
    assert errors[-1] < errors[0], "Sai số không giảm khi D tăng"


# ---------------------------------------------------------------------------
# Test 6: Cùng seed cho kết quả giống hệt
# ---------------------------------------------------------------------------
def test_same_seed_gives_identical_result():
    """Test 6: cùng seed cho kết quả giống hệt (bit-exact)."""
    d, D = 50, 500
    gamma = 0.5
    seed = 7

    x = torch.randn(10, d)

    rff_a = RandomFourierFeatures(d, D, gamma, seed=seed)
    rff_b = RandomFourierFeatures(d, D, gamma, seed=seed)

    phi_a = rff_a.transform(x)
    phi_b = rff_b.transform(x)

    assert torch.equal(phi_a, phi_b), "Cùng seed phải cho kết quả bit-exact"

    # Khác seed → khác kết quả
    rff_c = RandomFourierFeatures(d, D, gamma, seed=seed + 1)
    phi_c = rff_c.transform(x)
    assert not torch.equal(phi_a, phi_c), "Khác seed phải cho kết quả khác"


# ---------------------------------------------------------------------------
# Test bổ sung: RPReLU
# ---------------------------------------------------------------------------
def test_rprelu_shape_and_seed():
    """RPReLU trả về đúng kích thước và cùng seed cho kết quả giống hệt."""
    d, D, B = 32, 256, 16
    x = torch.randn(B, d)

    r1 = RPReLU(d, D, seed=0)
    r2 = RPReLU(d, D, seed=0)
    r3 = RPReLU(d, D, seed=1)

    out1 = r1.transform(x)
    out2 = r2.transform(x)
    out3 = r3.transform(x)

    assert out1.shape == (B, D), f"Shape sai: {out1.shape}"
    assert torch.equal(out1, out2), "Cùng seed phải cho kết quả bit-exact"
    assert not torch.equal(out1, out3), "Khác seed phải cho kết quả khác"
    assert (out1 >= 0).all(), "ReLU phải cho giá trị không âm"
