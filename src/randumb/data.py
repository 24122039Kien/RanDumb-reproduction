"""Tải dữ liệu từ Hugging Face, chuẩn hóa, lật ảnh và tách validation.

Xem Docs/Instruction.md, mục 4.2. Tên cột, tên split và số mẫu dưới đây đã đối chiếu
với Hugging Face datasets-server ngày 08/11/2026.

Quy ước dữ liệu trong module này:
- Ảnh thô là tensor ``uint8`` kích thước ``(N, H, W, C)``.
- Sau ``normalize`` là tensor ``float32`` cùng kích thước, chưa duỗi.
- ``flatten`` duỗi theo thứ tự ``H, W, C`` thành ``(N, d)`` ngay trước khi chiếu RFF,
  để có thể lật ảnh trên dữ liệu đã chuẩn hóa.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import torch

NORMALIZATIONS = ("zscore", "l2", "unit")


@dataclass(frozen=True)
class DatasetSpec:
    """Thông tin của một dataset, dùng để tải và kiểm tra ngay sau khi tải.

    Attributes:
        hf_id: Tên dataset trên Hugging Face.
        num_train: Số mẫu train mong đợi.
        num_test: Số mẫu của split dùng làm test.
        num_classes: Số lớp.
        use_flip: Có lật ảnh khi train hay không.
        image_column: Tên cột ảnh.
        label_column: Tên cột nhãn.
        test_split: Tên split trên Hugging Face dùng làm tập test.
        channels: 1 cho ảnh xám, 3 cho ảnh màu.
        resize_to: Cạnh ảnh sau khi thu nhỏ, ``None`` nếu giữ nguyên.
    """

    hf_id: str
    num_train: int
    num_test: int
    num_classes: int
    use_flip: bool
    image_column: str = "image"
    label_column: str = "label"
    test_split: str = "test"
    channels: int = 3
    resize_to: Optional[int] = None


DATASETS = {
    "mnist": DatasetSpec("ylecun/mnist", 60_000, 10_000, 10, False, channels=1),
    "cifar10": DatasetSpec("uoft-cs/cifar10", 50_000, 10_000, 10, True, image_column="img"),
    "cifar100": DatasetSpec(
        "uoft-cs/cifar100", 50_000, 10_000, 100, True, image_column="img", label_column="fine_label"
    ),
    # Maysee/tiny-imagenet đã đổi tên thành zh-plus/tiny-imagenet; ảnh gốc 64×64.
    "tiny-imagenet": DatasetSpec(
        "zh-plus/tiny-imagenet", 100_000, 10_000, 200, True, test_split="valid", resize_to=32
    ),
    # Bản của timm có split test 5.000 ảnh; dùng validation 10.000 ảnh như Instruction.md.
    # Nặng khoảng 7,5 GB và ảnh có kích thước khác nhau.
    "mini-imagenet": DatasetSpec(
        "timm/mini-imagenet", 50_000, 10_000, 100, True, test_split="validation", resize_to=32
    ),
}


def data_dir() -> Path:
    """Thư mục dữ liệu của project.

    Mặc định là ``<repo>/data``, không phụ thuộc thư mục đang đứng khi chạy lệnh.
    Đặt biến môi trường ``RANDUMB_DATA_DIR`` để đổi sang chỗ khác, ví dụ
    ``/kaggle/working/data`` trên Kaggle.

    Returns:
        Đường dẫn thư mục dữ liệu.
    """
    override = os.environ.get("RANDUMB_DATA_DIR")
    return Path(override) if override else Path(__file__).resolve().parents[2] / "data"


def get_spec(name: str) -> DatasetSpec:
    """Lấy ``DatasetSpec`` theo tên, báo lỗi rõ ràng nếu tên sai.

    Args:
        name: Khóa trong ``DATASETS``.

    Returns:
        Thông tin của dataset.

    Raises:
        KeyError: Nếu ``name`` không có trong ``DATASETS``.
    """
    if name not in DATASETS:
        raise KeyError(f"Dataset '{name}' không có. Các lựa chọn: {sorted(DATASETS)}")
    return DATASETS[name]


def image_to_array(image, spec: DatasetSpec) -> np.ndarray:
    """Chuyển một ảnh PIL về mảng ``uint8`` kích thước ``(H, W, C)``.

    Ảnh xám trong dataset màu (có trong TinyImageNet) được chuyển sang RGB.
    Thu nhỏ dùng bộ lọc bilinear; đây là lựa chọn của nhóm vì bài báo không nêu.

    Args:
        image: Ảnh ``PIL.Image``.
        spec: Thông tin dataset, quyết định số kênh và kích thước.

    Returns:
        Mảng ``uint8`` kích thước ``(H, W, C)``.
    """
    from PIL import Image

    image = image.convert("L" if spec.channels == 1 else "RGB")
    if spec.resize_to is not None and image.size != (spec.resize_to, spec.resize_to):
        image = image.resize((spec.resize_to, spec.resize_to), Image.BILINEAR)
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    return array


def load_images(
    name: str,
    split: str,
    cache_dir: Optional[Union[str, Path]] = None,
    hf_cache_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tải một split từ Hugging Face và kiểm tra số mẫu, tên cột, số lớp.

    Lần đầu tải sẽ giải mã toàn bộ ảnh rồi lưu ra file ``.npz`` để các lần sau đọc
    lại nhanh. Mặc định mọi thứ nằm trong ``data_dir()``: file gốc tải về ở
    ``data/hf``, ảnh đã giải mã ở ``data/cache``.

    Args:
        name: Khóa trong ``DATASETS``.
        split: ``"train"`` hoặc ``"test"``. ``"test"`` được ánh xạ sang
            ``DatasetSpec.test_split``.
        cache_dir: Thư mục lưu file ``.npz``, ``None`` để dùng ``data_dir() / "cache"``.
        hf_cache_dir: Thư mục lưu file gốc của Hugging Face, ``None`` để dùng
            ``data_dir() / "hf"``.
        use_cache: ``False`` để không đọc và không ghi file ``.npz``.

    Returns:
        Cặp ``(images, labels)``: ``uint8`` ``(N, H, W, C)`` và ``int64`` ``(N,)``.

    Raises:
        ValueError: Nếu ``split`` không hợp lệ.
        AssertionError: Nếu tên cột, số mẫu hoặc số lớp khác với ``DATASETS``.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split phải là 'train' hoặc 'test', nhận được '{split}'")
    spec = get_spec(name)
    expected = spec.num_train if split == "train" else spec.num_test

    cache_file = None
    if use_cache:
        cache_file = Path(cache_dir or data_dir() / "cache") / f"{name}-{split}.npz"
    if cache_file is not None and cache_file.exists():
        with np.load(cache_file) as cached:
            images, labels = cached["images"], cached["labels"]
    else:
        from datasets import load_dataset

        hf_split = spec.test_split if split == "test" else "train"
        hf_dir = Path(hf_cache_dir or data_dir() / "hf")
        dataset = load_dataset(spec.hf_id, split=hf_split, cache_dir=str(hf_dir))
        for column in (spec.image_column, spec.label_column):
            assert column in dataset.column_names, (
                f"{spec.hf_id}: không có cột '{column}', các cột hiện có: {dataset.column_names}"
            )
        assert len(dataset) == expected, f"{spec.hf_id}/{hf_split}: có {len(dataset)} mẫu, mong đợi {expected}"

        images = np.stack([image_to_array(image, spec) for image in dataset[spec.image_column]])
        labels = np.asarray(dataset[spec.label_column], dtype=np.int64)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, images=images, labels=labels)

    assert len(images) == expected, f"{name}/{split}: có {len(images)} mẫu, mong đợi {expected}"
    assert labels.min() >= 0 and labels.max() < spec.num_classes, (
        f"{name}/{split}: nhãn nằm ngoài [0, {spec.num_classes})"
    )
    return torch.from_numpy(images), torch.from_numpy(labels)


def train_val_split(
    images: torch.Tensor,
    labels: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Tách một phần train làm validation, giữ nguyên tỉ lệ giữa các lớp.

    Dùng để chọn γ và chuẩn hóa mà không nhìn tập test.

    Args:
        images: Ảnh train, chiều đầu là số mẫu.
        labels: Nhãn train ``int64``.
        val_fraction: Tỉ lệ mỗi lớp dành cho validation, trong khoảng ``(0, 1)``.
        seed: Seed của phép tách.

    Returns:
        Hai cặp ``(images, labels)`` cho train và validation.

    Raises:
        ValueError: Nếu ``val_fraction`` không nằm trong ``(0, 1)``.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction phải nằm trong (0, 1), nhận được {val_fraction}")
    generator = torch.Generator().manual_seed(seed)
    val_parts = []
    for cls in torch.unique(labels):
        idx = torch.nonzero(labels == cls).flatten()
        idx = idx[torch.randperm(len(idx), generator=generator)]
        val_parts.append(idx[: int(round(len(idx) * val_fraction))])
    val_idx = torch.cat(val_parts).sort().values
    mask = torch.ones(len(labels), dtype=torch.bool)
    mask[val_idx] = False
    train_idx = torch.nonzero(mask).flatten()
    return (images[train_idx], labels[train_idx]), (images[val_idx], labels[val_idx])


def channel_stats(images: torch.Tensor, chunk_size: int = 10_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Tính trung bình và độ lệch chuẩn theo kênh trên thang ``[0, 1]``.

    Cộng dồn theo từng khối bằng ``float64`` để không phải giữ cả tập ở dạng số thực.

    Args:
        images: Ảnh ``uint8`` ``(N, H, W, C)``, thường là tập train.
        chunk_size: Số ảnh mỗi khối.

    Returns:
        Cặp ``(mean, std)``, mỗi tensor ``float32`` kích thước ``(C,)``.
    """
    channels = images.shape[-1]
    total = torch.zeros(channels, dtype=torch.float64)
    total_sq = torch.zeros(channels, dtype=torch.float64)
    count = 0
    for start in range(0, len(images), chunk_size):
        block = images[start:start + chunk_size].to(torch.float64).div_(255.0).reshape(-1, channels)
        total += block.sum(dim=0)
        total_sq += (block * block).sum(dim=0)
        count += block.shape[0]
    mean = total / count
    std = (total_sq / count - mean * mean).clamp_min(0.0).sqrt()
    return mean.float(), std.float()


def normalize(
    images: torch.Tensor,
    mode: str,
    stats: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Chuẩn hóa ảnh, giữ nguyên kích thước ``(N, H, W, C)``.

    Bài báo chỉ viết "normalize all images", nên nhóm hỗ trợ ba cách để so sánh ở E8.

    Args:
        images: Ảnh ``uint8`` ``(N, H, W, C)``.
        mode: ``"zscore"`` (trừ mean, chia std theo kênh), ``"l2"`` (mỗi ảnh có chuẩn
            L2 bằng 1) hoặc ``"unit"`` (chỉ chia 255).
        stats: Kết quả ``channel_stats`` của tập train, bắt buộc khi ``mode="zscore"``.

    Returns:
        Tensor ``float32`` ``(N, H, W, C)``.

    Raises:
        ValueError: Nếu ``mode`` không hợp lệ hoặc thiếu ``stats`` cho ``"zscore"``.
    """
    if mode not in NORMALIZATIONS:
        raise ValueError(f"mode phải là một trong {NORMALIZATIONS}, nhận được '{mode}'")
    x = images.to(torch.float32).div_(255.0)
    if mode == "zscore":
        if stats is None:
            raise ValueError("mode='zscore' cần stats = channel_stats(train_images)")
        mean, std = stats
        x.sub_(mean).div_(std.clamp_min(1e-8))
    elif mode == "l2":
        norms = x.reshape(len(x), -1).norm(dim=1).clamp_min(1e-12)
        x.div_(norms.view(-1, *([1] * (x.dim() - 1))))
    return x


def flatten(images: torch.Tensor) -> torch.Tensor:
    """Duỗi ảnh ``(N, H, W, C)`` thành vector ``(N, H·W·C)``.

    Args:
        images: Ảnh đã chuẩn hóa.

    Returns:
        Tensor ``(N, d)``.
    """
    return images.reshape(len(images), -1)


def iter_batches(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    flip: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Duyệt dữ liệu theo lô và trả về vector đã duỗi.

    Khi ``flip=True``, mỗi lô ảnh gốc được theo sau bởi lô ảnh lật ngang tương ứng,
    nên cả ảnh gốc và ảnh lật đều được tính là mẫu (Docs/Instruction.md, mục 2.2).

    Args:
        images: Ảnh đã chuẩn hóa ``(N, H, W, C)``.
        labels: Nhãn ``(N,)``.
        batch_size: Số ảnh gốc mỗi lô.
        flip: Có thêm ảnh lật ngang hay không.

    Yields:
        Từng cặp ``(x, y)`` với ``x`` kích thước ``(B, d)``.
    """
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        yield flatten(batch), batch_labels
        if flip:
            yield flatten(torch.flip(batch, dims=[2])), batch_labels
