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
