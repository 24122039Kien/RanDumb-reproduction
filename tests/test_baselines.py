"""Kiểm thử NCM."""
import pytest
import torch

from randumb.baselines import class_means, predict_ncm


def test_class_means_and_empty_class():
    x = torch.tensor([[0.0, 0.0], [2.0, 2.0], [10.0, 0.0]])
    y = torch.tensor([0, 0, 1])
    means = class_means(x, y, num_classes=3)
    assert torch.allclose(means, torch.tensor([[1.0, 1.0], [10.0, 0.0], [0.0, 0.0]]))


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "dot"])
def test_predict_ncm_on_separable_points(metric):
    means = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    x = torch.tensor([[4.0, 0.5], [0.3, 6.0]])
    assert predict_ncm(x, means, metric).tolist() == [0, 1]


def test_euclidean_and_cosine_can_disagree():
    # Điểm gần trung bình lớp 0 về khoảng cách, nhưng cùng hướng với lớp 1.
    means = torch.tensor([[1.0, 1.0], [10.0, 0.0]])
    x = torch.tensor([[2.0, 0.5]])
    assert predict_ncm(x, means, "euclidean").item() == 0
    assert predict_ncm(x, means, "cosine").item() == 1


def test_predict_ncm_rejects_unknown_metric():
    means = torch.tensor([[1.0, 0.0]])
    x = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError):
        predict_ncm(x, means, "khong-co")
