"""Kiểm thử scripts/run_e2.py trên file đặc trưng giả, không cần mạng hay W&B."""
import importlib.util
from pathlib import Path

import pytest
import torch

from randumb.vit_features import save_features


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_e2.py"
    spec = importlib.util.spec_from_file_location("run_e2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def features_file(tmp_path):
    """Hai lớp tách biệt trong không gian 16 chiều, 100 lớp khai báo như CIFAR-100."""
    gen = torch.Generator().manual_seed(0)
    centers = torch.randn(2, 16, generator=gen) * 4

    def sample(n):
        y = torch.arange(2).repeat(n)
        return centers[y] + 0.5 * torch.randn(len(y), 16, generator=gen), y

    train_x, train_y = sample(50)
    test_x, test_y = sample(20)
    path = tmp_path / "cifar100-in21k-norm.pt"
    save_features(path, {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y,
                         "meta": {"model_name": "fake", "normalize": True}})
    return path


@pytest.mark.parametrize("method", ["randumb", "ncm"])
def test_run_e2_learns_and_reports_paper_number(runner, features_file, method):
    metrics = runner.main(["--dataset", "cifar100", "--backbone", "in21k", "--method", method,
                           "--features-file", str(features_file), "--no-wandb", "--device", "cpu"])
    assert metrics["final_accuracy"] > 95
    assert metrics["paper_accuracy"] == (86.8 if method == "randumb" else 83.5)
    assert metrics["relative_deviation_pct"] is not None
    if method == "randumb":
        assert 0 <= metrics["rho"] <= 1


def test_run_e2_missing_features_file_is_explained(runner, tmp_path):
    with pytest.raises(FileNotFoundError, match="extract_vit_features"):
        runner.main(["--dataset", "cub", "--backbone", "in1k", "--method", "ncm",
                     "--features-file", str(tmp_path / "none.pt"), "--no-wandb"])
