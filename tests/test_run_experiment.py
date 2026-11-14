"""Kiểm tra cấu hình paper được truyền tới pipeline mà không tải dataset."""
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dataset,lam,flip", [
    ("mnist", 1e-6, False),
    ("cifar10", 1e-5, True),
    ("cifar100", 1e-5, True),
    ("tiny-imagenet", 1e-4, True),
    ("mini-imagenet", 1e-4, True),
])
def test_default_run_passes_paper_settings_to_pipeline(runner, monkeypatch, dataset, lam, flip):
    # Không cấp phát ma trận D=25K: bắt cấu hình tại ranh giới gọi pipeline.
    images = torch.arange(24, dtype=torch.uint8).reshape(2, 2, 2, 3)
    labels = torch.tensor([0, 1])
    monkeypatch.setattr(runner, "load_images", lambda *a, **kw: (images, labels))
    monkeypatch.setattr(runner, "peak_ram_gb", lambda: None)
    monkeypatch.setattr(runner, "peak_vram_gb", lambda: None)
    observed = {}

    def capture_run(train_x, train_y, eval_x, eval_y, num_classes, cfg, stats_path):
        observed["cfg"] = cfg
        return dict(accuracy=0.0, per_class_accuracy=[], rho=0.5, n_train_seen=2,
                    time_features_s=0, time_stats_s=0, time_solve_s=0, time_predict_s=0)

    # --no-wandb giữ bài test hoàn toàn offline.
    monkeypatch.setattr(runner, "fit_and_evaluate", capture_run)
    runner.main(["--dataset", dataset, "--group", "E1", "--no-wandb", "--device", "cpu"])
    cfg = observed["cfg"]
    assert cfg.dim == 25_000
    assert cfg.gamma == 1.0
    assert cfg.lam == lam
    assert cfg.flip is flip
    assert cfg.features == "rff"
    assert cfg.stats_dtype == cfg.feature_dtype == torch.float64


def test_explicit_sensitivity_settings_remain_available(runner):
    args = runner.parse_args(["--dataset", "cifar10", "--group", "E8", "--gamma", "scale",
                              "--lam", "0.0001", "--dim", "2000", "--feature-dtype", "float32"])
    assert args.gamma == "scale"
    assert args.lam == 1e-4
    assert args.dim == 2000
    assert args.feature_dtype == "float32"
