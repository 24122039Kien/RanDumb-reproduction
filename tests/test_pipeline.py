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


def test_per_class_oas_matches_independent_reference(monkeypatch):
    from sklearn.covariance import OAS
    import randumb.pipeline as pipeline

    generator = torch.Generator().manual_seed(44)
    x0 = torch.randn(25, 5, generator=generator, dtype=torch.float64)
    x1 = torch.randn(65, 5, generator=generator, dtype=torch.float64) * 3 + 2
    x = torch.cat([x0, x1])
    y = torch.tensor([0] * len(x0) + [1] * len(x1))
    refs = [OAS(store_precision=False).fit(a.numpy()) for a in [x0, x1]]
    expected_cov = sum(torch.from_numpy(r.covariance_) * len(a) for r, a in zip(refs, [x0, x1])) / len(x)
    original_fit = pipeline.fit_lda
    captured = {}

    def capture(means, cov, lam, **kwargs):
        captured["cov"] = cov.clone()
        captured["means"] = means.clone()
        return original_fit(means, cov, lam, **kwargs)

    monkeypatch.setattr(pipeline, "fit_lda", capture)
    cfg = PipelineConfig(dim=5, gamma=1.0, lam=0.2, seed=0, features="identity",
                         feature_dtype=torch.float64, shrink="oas_per_class", lam_mode="relative", batch_size=9)
    result = fit_and_evaluate(x, y, x, y, 2, cfg)
    torch.testing.assert_close(captured["cov"], expected_cov)
    torch.testing.assert_close(captured["means"], torch.stack([x0.mean(0), x1.mean(0)]))
    assert result["rho_per_class"] == pytest.approx([r.shrinkage_ for r in refs])
    assert result["rho"] is None
    assert result["lambda_effective"] == pytest.approx(0.2 * expected_cov.trace().item() / 5)
    assert result["n_train_seen"] == len(x)


def test_relative_ridge_preserves_predictions_under_feature_rescaling():
    x, y = _blobs()
    cfg = PipelineConfig(dim=16, gamma=1.0, lam=0.1, seed=0, features="identity",
                         shrink="oas", lam_mode="relative", feature_dtype=torch.float64)
    first = fit_and_evaluate(x, y, x, y, 3, cfg)
    scaled = fit_and_evaluate(1e-5 * x, y, 1e-5 * x, y, 3, cfg)
    assert first["accuracy"] == scaled["accuracy"]
    assert first["rho"] == pytest.approx(scaled["rho"], rel=1e-5)
    assert scaled["lambda_effective"] == pytest.approx(first["lambda_effective"] * 1e-10, rel=1e-5, abs=0)
