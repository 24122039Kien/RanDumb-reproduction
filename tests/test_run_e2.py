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


def test_split_seed_resplits_self_split_dataset(runner, features_file):
    base = ["--dataset", "imagenet-a", "--backbone", "in1k", "--method", "randumb",
            "--features-file", str(features_file), "--no-wandb", "--device", "cpu"]
    first = runner.main(base + ["--split-seed", "1"])
    again = runner.main(base + ["--split-seed", "1"])
    # 140 mẫu, mỗi lớp 70: 14 mẫu mỗi lớp vào test.
    assert (first["n_train"], first["n_test"]) == (112, 28)
    assert first["final_accuracy"] == again["final_accuracy"]
    assert first["paper_accuracy"] == 49.5


def test_split_seed_rejected_when_dataset_has_its_own_test_split(runner, features_file):
    with pytest.raises(SystemExit):
        runner.main(["--dataset", "cifar100", "--backbone", "in21k", "--method", "ncm",
                     "--features-file", str(features_file), "--no-wandb", "--split-seed", "1"])


def test_default_name_separates_ncm_metric_and_split(runner):
    args = runner.parse_args(["--dataset", "imagenet-a", "--backbone", "in1k", "--method", "ncm",
                              "--ncm-metric", "cosine", "--split-seed", "2"])
    assert runner.default_name(args) == "E2-imagenet-a-in1k-ncm-cosine-split2-seed0"
    args = runner.parse_args(["--dataset", "cub", "--backbone", "in1k", "--method", "ncm"])
    assert runner.default_name(args) == "E2-cub-in1k-ncm-seed0"  # trùng tên các run đã có
