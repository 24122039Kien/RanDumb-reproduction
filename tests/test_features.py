"""Kiểm thử RFF và seed (Docs/Instruction.md, mục 2.6, test 1 và 6)."""
import pytest

pytestmark = pytest.mark.skip(reason="Chưa cài, làm ở Sprint 1")


def test_rff_approximates_rbf_kernel():
    """Test 1: `φ(x)ᵀφ(y)` tiến gần `exp(-γ‖x-y‖²)` khi D tăng."""


def test_same_seed_gives_identical_result():
    """Test 6: cùng seed cho kết quả giống hệt."""
