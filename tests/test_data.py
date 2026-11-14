"""Kiểm thử các tiện ích dữ liệu trên dữ liệu giả, không cần mạng."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from randumb import data
from randumb.data import (
    DatasetSpec,
    channel_stats,
    flatten,
    get_spec,
    image_to_array,
    iter_batches,
    load_images,
    normalize,
    train_val_split,
)


def _fake_images(n=40, h=4, w=5, c=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, h, w, c), generator=gen, dtype=torch.uint8)


def test_get_spec_rejects_unknown_name():
    with pytest.raises(KeyError):
        get_spec("khong-co")


def test_image_to_array_handles_gray_rgb_and_resize():
    gray_spec = DatasetSpec("x", 1, 1, 2, False, channels=1)
    rgb_spec = DatasetSpec("x", 1, 1, 2, True, resize_to=32)
    gray = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    big_gray = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))  # ảnh xám trong dataset màu

    assert image_to_array(gray, gray_spec).shape == (28, 28, 1)
    assert image_to_array(big_gray, rgb_spec).shape == (32, 32, 3)


def test_train_val_split_is_stratified_disjoint_and_seeded():
    images = torch.arange(100).view(100, 1, 1, 1)
    labels = torch.arange(100) % 4
    (tr_x, tr_y), (va_x, va_y) = train_val_split(images, labels, 0.2, seed=0)

    assert len(va_y) == 20 and len(tr_y) == 80
    assert torch.bincount(va_y).tolist() == [5, 5, 5, 5]
    assert set(tr_x.flatten().tolist()).isdisjoint(va_x.flatten().tolist())
    (_, _), (va_x2, _) = train_val_split(images, labels, 0.2, seed=0)
    assert torch.equal(va_x, va_x2)


def test_split_indices_is_stratified_and_complete():
    labels = torch.arange(30) % 3
    first, second = data.split_indices(labels, 0.2, seed=1)
    assert torch.bincount(labels[second]).tolist() == [2, 2, 2]
    assert sorted(first.tolist() + second.tolist()) == list(range(30))


def test_train_val_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        train_val_split(torch.zeros(4, 1, 1, 1), torch.zeros(4, dtype=torch.long), 1.0, seed=0)


def test_channel_stats_matches_direct_computation():
    images = _fake_images()
    mean, std = channel_stats(images, chunk_size=7)
    x = images.double().div(255).reshape(-1, 3)
    assert torch.allclose(mean.double(), x.mean(0), atol=1e-6)
    assert torch.allclose(std.double(), x.std(0, unbiased=False), atol=1e-6)


def test_normalize_modes():
    images = _fake_images()
    z = normalize(images, "zscore", channel_stats(images)).reshape(-1, 3)
    assert torch.allclose(z.mean(0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(z.std(0, unbiased=False), torch.ones(3), atol=1e-4)

    l2 = normalize(images, "l2")
    assert torch.allclose(flatten(l2).norm(dim=1), torch.ones(len(images)), atol=1e-5)

    unit = normalize(images, "unit")
    assert unit.min() >= 0 and unit.max() <= 1
    assert unit.shape == images.shape and unit.dtype == torch.float32


def test_normalize_rejects_bad_input():
    images = _fake_images()
    with pytest.raises(ValueError):
        normalize(images, "khong-co")
    with pytest.raises(ValueError):
        normalize(images, "zscore")


def test_iter_batches_counts_flipped_samples():
    images = normalize(_fake_images(n=10), "unit")
    labels = torch.arange(10)
    batches = list(iter_batches(images, labels, batch_size=4, flip=True))

    assert sum(len(y) for _, y in batches) == 20
    assert batches[0][0].shape == (4, 4 * 5 * 3)
    flipped = torch.flip(images[:4], dims=[2]).reshape(4, -1)
    assert torch.equal(batches[1][0], flipped)
    assert sum(len(y) for _, y in iter_batches(images, labels, 4, flip=False)) == 10


def _install_fake_datasets(monkeypatch, n_rows, columns=("img", "label")):
    rows = {
        columns[0]: [Image.fromarray(np.full((4, 4, 3), i, dtype=np.uint8)) for i in range(n_rows)],
        columns[1]: [i % 2 for i in range(n_rows)],
    }

    class FakeDataset:
        column_names = list(columns)

        def __len__(self):
            return n_rows

        def __getitem__(self, key):
            return rows[key]

    calls = []

    def load_dataset(hf_id, split, cache_dir=None):
        calls.append((hf_id, split, cache_dir))
        return FakeDataset()

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    return calls


def test_data_dir_defaults_to_repo_and_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.delenv("RANDUMB_DATA_DIR", raising=False)
    assert data.data_dir() == Path(data.__file__).resolve().parents[2] / "data"
    monkeypatch.setenv("RANDUMB_DATA_DIR", str(tmp_path))
    assert data.data_dir() == tmp_path


def test_load_images_checks_counts_and_uses_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("RANDUMB_DATA_DIR", str(tmp_path))
    spec = DatasetSpec("fake/ds", 6, 3, 2, True, image_column="img", test_split="valid")
    monkeypatch.setitem(data.DATASETS, "fake", spec)

    calls = _install_fake_datasets(monkeypatch, n_rows=3)
    images, labels = load_images("fake", "test")
    assert images.shape == (3, 4, 4, 3) and images.dtype == torch.uint8
    assert labels.tolist() == [0, 1, 0]
    assert calls == [("fake/ds", "valid", str(tmp_path / "hf"))]
    assert (tmp_path / "cache" / "fake-test.npz").exists()

    load_images("fake", "test")  # lần hai đọc từ cache
    assert len(calls) == 1
    load_images("fake", "test", use_cache=False)  # bỏ qua cache thì tải lại
    assert len(calls) == 2

    with pytest.raises(AssertionError):
        load_images("fake", "train")  # có 3 mẫu, mong đợi 6


def test_load_images_reports_missing_column(monkeypatch, tmp_path):
    monkeypatch.setenv("RANDUMB_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(data.DATASETS, "fake", DatasetSpec("fake/ds", 3, 3, 2, True))
    _install_fake_datasets(monkeypatch, n_rows=3, columns=("img", "label"))
    with pytest.raises(AssertionError, match="image"):
        load_images("fake", "train")
