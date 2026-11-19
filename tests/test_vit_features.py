"""Kiểm thử phần trích đặc trưng E2 với dataset giả và mô hình giả, không cần mạng."""
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

from randumb import vit_features as vf


class FakeDataset:
    """Giả lập các thao tác của dataset Hugging Face mà module dùng tới."""

    def __init__(self, rows: dict):
        self.rows = rows
        self.column_names = list(rows)

    def __len__(self):
        return len(next(iter(self.rows.values())))

    def __getitem__(self, column):
        return self.rows[column]

    def select(self, indices):
        return FakeDataset({k: [v[i] for i in indices] for k, v in self.rows.items()})


def _images(n, size=6):
    return [Image.fromarray(np.full((size, size, 3), i * 10 % 256, dtype=np.uint8)) for i in range(n)]


def test_encode_labels_maps_strings_in_sorted_order():
    assert vf.encode_labels(["n02", "n01", "n02"]).tolist() == [1, 0, 1]
    assert vf.encode_labels([3, 1]).tolist() == [3, 1]


def test_preprocess_shape_range_and_normalization():
    raw = vf.preprocess(_images(2), image_size=8)
    assert raw.shape == (2, 3, 8, 8) and raw.dtype == torch.float32
    assert raw.min() >= 0 and raw.max() <= 1
    normed = vf.preprocess(_images(2), image_size=8, mean=(0.5,) * 3, std=(0.5,) * 3)
    assert torch.allclose(normed, (raw - 0.5) / 0.5)


def test_extract_features_keeps_order_and_batches():
    dataset = FakeDataset({"image": _images(5)})
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 4))
    indices = torch.tensor([4, 0, 2])
    feats = vf.extract_features(model, dataset, indices, "image", batch_size=2, image_size=8)
    expected = model(vf.preprocess([dataset["image"][i] for i in [4, 0, 2]], image_size=8, mean=(0.5,) * 3, std=(0.5,) * 3))
    assert feats.shape == (3, 4)
    assert torch.allclose(feats, expected, atol=1e-6)


def _install_fake_hf(monkeypatch, splits: dict):
    calls = []

    def load_dataset(hf_id, split, cache_dir=None):
        calls.append((hf_id, split))
        return splits[split]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    return calls


def test_load_e2_splits_self_splits_single_split_dataset(monkeypatch, tmp_path):
    labels = [f"n{i % 4:02d}" for i in range(40)]
    spec = vf.E2DatasetSpec("fake/r", 4, "image", "wnid", "test", 40, test_fraction=0.3)
    monkeypatch.setitem(vf.E2_DATASETS, "fake-r", spec)
    _install_fake_hf(monkeypatch, {"test": FakeDataset({"image": _images(40), "wnid": labels})})

    splits = vf.load_e2_splits("fake-r", hf_cache_dir=tmp_path)
    (_, tr_idx, tr_y), (_, te_idx, te_y) = splits["train"], splits["test"]
    assert len(tr_idx) == 28 and len(te_idx) == 12  # mỗi lớp 10 mẫu, 3 mẫu vào test
    assert set(tr_idx.tolist()).isdisjoint(te_idx.tolist())
    assert torch.bincount(te_y).tolist() == [3, 3, 3, 3]
    assert torch.bincount(tr_y).tolist() == [7, 7, 7, 7]
    again = vf.load_e2_splits("fake-r", hf_cache_dir=tmp_path)
    assert torch.equal(again["test"][1], te_idx)  # cố định nhờ SPLIT_SEED


def test_load_e2_splits_uses_given_test_split_and_checks_counts(monkeypatch, tmp_path):
    spec = vf.E2DatasetSpec("fake/c", 2, "img", "label", "train", 4, "test", 2)
    monkeypatch.setitem(vf.E2_DATASETS, "fake-c", spec)
    data = {
        "train": FakeDataset({"img": _images(4), "label": [0, 1, 0, 1]}),
        "test": FakeDataset({"img": _images(2), "label": [1, 0]}),
    }
    calls = _install_fake_hf(monkeypatch, data)
    splits = vf.load_e2_splits("fake-c", hf_cache_dir=tmp_path)
    assert calls == [("fake/c", "train"), ("fake/c", "test")]
    assert splits["test"][2].tolist() == [1, 0]

    monkeypatch.setitem(vf.E2_DATASETS, "fake-c", vf.E2DatasetSpec("fake/c", 2, "img", "label", "train", 5, "test", 2))
    with pytest.raises(AssertionError):
        vf.load_e2_splits("fake-c", hf_cache_dir=tmp_path)


def test_features_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("RANDUMB_DATA_DIR", str(tmp_path))
    path = vf.features_path("cub", "in21k", normalize=False)
    assert path == tmp_path / "features" / "cub-in21k-raw.pt"
    payload = {"train_x": torch.ones(2, 3), "train_y": torch.tensor([0, 1]),
               "test_x": torch.zeros(1, 3), "test_y": torch.tensor([1]),
               "meta": {"model_name": "x", "mean": (0.5, 0.5, 0.5), "normalize": False}}
    vf.save_features(path, payload)
    loaded = vf.load_features(path)
    assert torch.equal(loaded["train_x"], payload["train_x"]) and loaded["meta"]["mean"] == (0.5, 0.5, 0.5)
    with pytest.raises(FileNotFoundError):
        vf.load_features(tmp_path / "khong-co.pt")


def test_load_features_accepts_old_files_with_torch_version(tmp_path):
    # File trích trên Kaggle trước 16/11 lưu torch.__version__ nguyên kiểu TorchVersion.
    path = tmp_path / "old.pt"
    torch.save({"train_x": torch.ones(1, 2), "meta": {"torch_version": torch.__version__}}, path)
    assert vf.load_features(path)["meta"]["torch_version"] == torch.__version__


def test_extract_script_writes_a_file_run_e2_can_read(tmp_path, monkeypatch):
    """Chạy đúng đường đi của scripts/extract_vit_features.py với dataset và mô hình giả."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("extract", root / "scripts" / "extract_vit_features.py")
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)

    data = FakeDataset({"img": _images(6), "fine_label": [0, 1, 2, 0, 1, 2]})
    splits = {"train": (data, torch.arange(4), torch.tensor([0, 1, 2, 0])),
              "test": (data, torch.tensor([4, 5]), torch.tensor([1, 2]))}
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(2), torch.nn.Flatten())
    info = {"backbone": "in21k", "model_name": "fake", "pretrained_tag": "fake", "hf_hub_id": None,
            "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5), "timm_version": "0", "feature_dim": 12}
    monkeypatch.setattr(extract, "load_e2_splits", lambda *a, **k: splits)
    monkeypatch.setattr(extract, "create_backbone", lambda *a, **k: (model, info))

    out = extract.main(["--dataset", "cifar100", "--backbone", "in21k", "--device", "cpu",
                        "--out", str(tmp_path / "f.pt"), "--batch-size", "4"])
    loaded = vf.load_features(out)
    assert loaded["train_x"].shape == (4, 12) and loaded["test_x"].shape == (2, 12)
    assert isinstance(loaded["meta"]["torch_version"], str)


def test_resplit_features_keeps_all_samples_and_class_balance():
    y = torch.arange(4).repeat_interleave(10)
    x = torch.arange(40, dtype=torch.float32).unsqueeze(1)
    payload = {"train_x": x[:30], "train_y": y[:30], "test_x": x[30:], "test_y": y[30:], "meta": {"a": 1}}
    out = vf.resplit_features(payload, 0.2, seed=3)
    assert torch.bincount(out["test_y"]).tolist() == [2, 2, 2, 2]
    merged = torch.cat([out["train_x"], out["test_x"]]).flatten().sort().values
    assert torch.equal(merged, x.flatten())  # không mất, không trùng mẫu
    assert torch.equal(out["train_y"], y[out["train_x"].flatten().long()])  # nhãn đi đúng với mẫu
    assert out["meta"] == {"a": 1, "resplit_seed": 3}
    other = vf.resplit_features(payload, 0.2, seed=4)
    assert not torch.equal(out["test_x"], other["test_x"])
