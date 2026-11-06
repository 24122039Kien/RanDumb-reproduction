"""Kiểm thử OAS và dự đoán (Docs/Instruction.md, mục 2.6, test 4 và 5)."""
import pytest

pytestmark = pytest.mark.skip(reason="Chưa cài, làm ở Sprint 1")


def test_oas_matches_sklearn():
    """Test 4: khớp `sklearn.covariance.OAS` trên dữ liệu một lớp."""


def test_predict_matches_full_mahalanobis():
    """Test 5: khớp phép tính khoảng cách Mahalanobis đầy đủ, viết chậm."""
