"""Kiểm thử pipeline đầu cuối trên dữ liệu giả, không cần mạng.

Các test gọi phần lõi (features, stats, classifier) sẽ tự bỏ qua khi phần lõi chưa
được cài trên nhánh hiện tại.
"""
import pytest
import torch

from randumb.features import RandomFourierFeatures
from randumb.pipeline import PipelineConfig, fit_and_evaluate, resolve_gamma


def _core_ready() -> bool:
    try:
        RandomFourierFeatures(2, 2, 1.0, 0)
    except NotImplementedError:
        return False
    return True


needs_core = pytest.mark.skipif(not _core_ready(), reason="Chờ phần lõi features, stats, classifier")


def _blobs(n_per_class=60, num_classes=3, seed=0):
    """Ba cụm ảnh 4×4×1 tách biệt rõ, đã ở dạng số thực."""
    gen = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, 4, 4, 1, generator=gen) * 3
    labels = torch.arange(num_classes).repeat_interleave(n_per_class)
    images = centers[labels] + 0.3 * torch.randn(len(labels), 4, 4, 1, generator=gen)
    return images, labels


def test_resolve_gamma():
    x = torch.randn(50, 4, 4, 1, dtype=torch.float64) * 2
    assert resolve_gamma("scale", x) == pytest.approx(1 / (16 * x.var().item()))
    assert resolve_gamma("0.5", x) == 0.5
    assert resolve_gamma(2.0, x) == 2.0


@needs_core
@pytest.mark.parametrize("features", ["rff", "rp-relu"])
def test_pipeline_learns_separable_blobs(features, tmp_path):
    train_x, train_y = _blobs(seed=0)
    eval_x, eval_y = _blobs(n_per_class=20, seed=0)
    cfg = PipelineConfig(dim=200, gamma=resolve_gamma("scale", train_x), lam=1e-4, seed=0,
                         features=features, flip=True, batch_size=32)

    result = fit_and_evaluate(train_x, train_y, eval_x, eval_y, 3, cfg, stats_path=tmp_path / "stats.pt")

    assert result["accuracy"] > 90
    assert result["n_train_seen"] == 2 * len(train_y)  # tính cả ảnh lật
    assert len(result["per_class_accuracy"]) == 3
    assert 0 <= result["rho"] <= 1
    assert (tmp_path / "stats.pt").exists()


@needs_core
def test_pipeline_is_deterministic_and_order_invariant():
    train_x, train_y = _blobs(seed=1)
    eval_x, eval_y = _blobs(n_per_class=20, seed=2)
    cfg = PipelineConfig(dim=100, gamma=0.05, lam=1e-4, seed=3, batch_size=16)

    first = fit_and_evaluate(train_x, train_y, eval_x, eval_y, 3, cfg)
    perm = torch.randperm(len(train_y), generator=torch.Generator().manual_seed(0))
    shuffled = fit_and_evaluate(train_x[perm], train_y[perm], eval_x, eval_y, 3, cfg)

    assert first["accuracy"] == shuffled["accuracy"]


@needs_core
def test_pipeline_identity_features_use_input_dimension():
    train_x, train_y = _blobs(seed=4)
    eval_x, eval_y = _blobs(n_per_class=20, seed=4)
    flat_train, flat_eval = train_x.reshape(len(train_x), -1), eval_x.reshape(len(eval_x), -1)
    cfg = PipelineConfig(dim=0, gamma=0.0, lam=1e-4, seed=0, features="identity", batch_size=32)

    result = fit_and_evaluate(flat_train, train_y, flat_eval, eval_y, 3, cfg)

    assert result["accuracy"] > 90
    assert result["n_train_seen"] == len(train_y)


def test_pipeline_rejects_unknown_shrink():
    train_x, train_y = _blobs()
    cfg = PipelineConfig(dim=10, gamma=1.0, lam=0.0, seed=0, shrink="khong-co")
    with pytest.raises(ValueError):
        fit_and_evaluate(train_x, train_y, train_x, train_y, 3, cfg)
