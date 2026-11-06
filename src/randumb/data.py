"""Tải dữ liệu từ Hugging Face, chuẩn hóa, lật ảnh và tách validation.

Xem Docs/Instruction.md, mục 4.2. Các nguồn dưới đây chưa được tải thử.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass(frozen=True)
class DatasetSpec:
    """Thông tin của một dataset, dùng để kiểm tra ngay sau khi tải.

    Attributes:
        hf_id: Tên dataset trên Hugging Face.
        num_train: Số mẫu train mong đợi.
        num_test: Số mẫu test mong đợi.
        num_classes: Số lớp.
        use_flip: Có lật ảnh khi train hay không.
        resize_to: Cạnh ảnh sau khi thu nhỏ, `None` nếu giữ nguyên.
    """

    hf_id: str
    num_train: int
    num_test: int
    num_classes: int
    use_flip: bool
    resize_to: Optional[int] = None


DATASETS = {
    "mnist": DatasetSpec("ylecun/mnist", 60_000, 10_000, 10, False),
    "cifar10": DatasetSpec("uoft-cs/cifar10", 50_000, 10_000, 10, True),
    "cifar100": DatasetSpec("uoft-cs/cifar100", 50_000, 10_000, 100, True),
    "tiny-imagenet": DatasetSpec("Maysee/tiny-imagenet", 100_000, 10_000, 200, True, 32),
    "mini-imagenet": DatasetSpec("timm/mini-imagenet", 50_000, 10_000, 100, True, 32),
}


def load_images(name: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Tải một split và kiểm tra số mẫu với `DATASETS`.

    Args:
        name: Khóa trong `DATASETS`.
        split: `"train"` hoặc `"test"`.

    Returns:
        Cặp `(images, labels)`.
    """
    raise NotImplementedError("Sprint 1")


def train_val_split(
    images: torch.Tensor,
    labels: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Tách một phần train làm validation để chọn γ và chuẩn hóa.

    Args:
        images: Ảnh train.
        labels: Nhãn train.
        val_fraction: Tỉ lệ dành cho validation.
        seed: Seed của phép tách.

    Returns:
        Hai cặp `(images, labels)` cho train và validation.
    """
    raise NotImplementedError("Sprint 1")


def normalize(images: torch.Tensor, mode: str, reference: torch.Tensor) -> torch.Tensor:
    """Chuẩn hóa ảnh và duỗi thành vector.

    Args:
        images: Ảnh cần chuẩn hóa.
        mode: `"zscore"`, `"l2"` hoặc `"unit"` (chỉ chia 255).
        reference: Ảnh train dùng để tính thống kê của `"zscore"`.

    Returns:
        Tensor kích thước `(N, d)`.
    """
    raise NotImplementedError("Sprint 1")


def iter_batches(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    flip: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Duyệt dữ liệu theo lô, tính cả ảnh gốc và ảnh lật khi `flip=True`.

    Args:
        images: Ảnh train.
        labels: Nhãn train.
        batch_size: Kích thước lô.
        flip: Có thêm ảnh lật ngang hay không.

    Yields:
        Từng cặp `(images, labels)`.
    """
    raise NotImplementedError("Sprint 1")
