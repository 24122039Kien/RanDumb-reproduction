"""Trích đặc trưng ViT-B/16 cho E2 (Benchmark F) và lưu lại để khỏi trích lại.

Xem Docs/Instruction.md, mục 4.4, và Docs/e2_datasets.md. Benchmark F không dùng RFF:
RanDumb chỉ học bộ phân loại tuyến tính trên đặc trưng 768 chiều của ViT.

Quy ước:
- Ảnh được đổi sang RGB, thu thẳng về 224×224 bằng bicubic, không cắt giữa.
- ``normalize=True`` trừ mean, chia std theo cấu hình của mô hình trong timm;
  ``False`` giữ ảnh ở thang ``[0, 1]`` như code gốc của tác giả (Docs/Instruction.md, mục 4.4).
- Đặc trưng là vector pre-logits (token CLS sau lớp norm cuối) của ``num_classes=0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import torch

from randumb.data import data_dir, split_indices

# Trọng số gần với timm 0.6 mà các bài cùng thời dùng; timm 1.x đổi mặc định của tên ngắn.
BACKBONES = {
    "in1k": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "in21k": "vit_base_patch16_224.augreg_in21k",
}
IMAGE_SIZE = 224
SPLIT_SEED = 0  # cố định, để mọi run dùng cùng cách chia ImageNet-R và ImageNet-A


@dataclass(frozen=True)
class E2DatasetSpec:
    """Thông tin một dataset của E2.

    Attributes:
        hf_id: Tên dataset trên Hugging Face.
        num_classes: Số lớp.
        image_column: Tên cột ảnh.
        label_column: Tên cột nhãn; nếu là chuỗi (ví dụ wnid) sẽ được đánh số theo thứ tự sắp xếp.
        train_split: Split dùng làm train, hoặc split duy nhất cần tự chia.
        num_train: Số mẫu mong đợi của ``train_split``.
        test_split: Split dùng làm test; ``None`` thì tự chia từ ``train_split``.
        num_test: Số mẫu mong đợi của ``test_split``.
        test_fraction: Tỉ lệ mỗi lớp đưa vào test khi tự chia.
    """

    hf_id: str
    num_classes: int
    image_column: str
    label_column: str
    train_split: str
    num_train: int
    test_split: Optional[str] = None
    num_test: Optional[int] = None
    test_fraction: float = 0.2


E2_DATASETS = {
    "cifar100": E2DatasetSpec("uoft-cs/cifar100", 100, "img", "fine_label", "train", 50_000, "test", 10_000),
    "cub": E2DatasetSpec("Donghyun99/CUB-200-2011", 200, "image", "label", "train", 5_994, "test", 5_794),
    "imagenet-r": E2DatasetSpec("axiong/imagenet-r", 200, "image", "wnid", "test", 30_000),
    "imagenet-a": E2DatasetSpec("barkermrl/imagenet-a", 200, "image", "label", "train", 7_500),
}


def get_e2_spec(name: str) -> E2DatasetSpec:
    """Lấy thông tin dataset E2 theo tên.

    Args:
        name: Khóa trong ``E2_DATASETS``.

    Returns:
        Thông tin của dataset.

    Raises:
        KeyError: Nếu tên không có trong ``E2_DATASETS``.
    """
    if name not in E2_DATASETS:
        raise KeyError(f"Dataset E2 '{name}' không có. Các lựa chọn: {sorted(E2_DATASETS)}")
    return E2_DATASETS[name]


def encode_labels(raw_labels: list) -> torch.Tensor:
    """Đổi nhãn sang số nguyên; nhãn dạng chuỗi được đánh số theo thứ tự sắp xếp.

    Args:
        raw_labels: Danh sách nhãn (số nguyên hoặc chuỗi).

    Returns:
        Tensor ``int64``.
    """
    if raw_labels and isinstance(raw_labels[0], str):
        vocab = {name: i for i, name in enumerate(sorted(set(raw_labels)))}
        return torch.tensor([vocab[name] for name in raw_labels], dtype=torch.int64)
    return torch.tensor(raw_labels, dtype=torch.int64)


def load_e2_splits(name: str, hf_cache_dir: Optional[Union[str, Path]] = None) -> dict[str, tuple[Any, torch.Tensor, torch.Tensor]]:
    """Tải dataset E2 và xác định chỉ số của tập train và test.

    Ảnh chưa được giải mã ở bước này; chỉ cột nhãn được đọc.

    Args:
        name: Khóa trong ``E2_DATASETS``.
        hf_cache_dir: Thư mục lưu file của Hugging Face, ``None`` để dùng ``data/hf``.

    Returns:
        Từ điển ``{"train": (dataset, indices, labels), "test": (...)}``.

    Raises:
        AssertionError: Nếu thiếu cột, sai số mẫu hoặc sai số lớp.
    """
    from datasets import load_dataset

    spec = get_e2_spec(name)
    cache = str(hf_cache_dir or data_dir() / "hf")

    def _load(split: str, expected: int):
        dataset = load_dataset(spec.hf_id, split=split, cache_dir=cache)
        for column in (spec.image_column, spec.label_column):
            assert column in dataset.column_names, (
                f"{spec.hf_id}: không có cột '{column}', các cột hiện có: {dataset.column_names}"
            )
        assert len(dataset) == expected, f"{spec.hf_id}/{split}: có {len(dataset)} mẫu, mong đợi {expected}"
        return dataset, encode_labels(list(dataset[spec.label_column]))

    train_set, train_labels = _load(spec.train_split, spec.num_train)
    if spec.test_split is None:
        train_idx, test_idx = split_indices(train_labels, spec.test_fraction, SPLIT_SEED)
        splits = {
            "train": (train_set, train_idx, train_labels[train_idx]),
            "test": (train_set, test_idx, train_labels[test_idx]),
        }
    else:
        test_set, test_labels = _load(spec.test_split, spec.num_test)
        splits = {
            "train": (train_set, torch.arange(len(train_set)), train_labels),
            "test": (test_set, torch.arange(len(test_set)), test_labels),
        }
    for split, (_, _, labels) in splits.items():
        assert int(labels.max()) < spec.num_classes, f"{name}/{split}: nhãn vượt quá {spec.num_classes} lớp"
    return splits


def preprocess(images: list, image_size: int = IMAGE_SIZE, mean=None, std=None) -> torch.Tensor:
    """Đổi một lô ảnh PIL thành tensor ``(B, 3, H, W)`` cho ViT.

    Args:
        images: Danh sách ảnh ``PIL.Image``.
        image_size: Cạnh ảnh sau khi thu về.
        mean: Trung bình theo kênh để chuẩn hóa, ``None`` để giữ thang ``[0, 1]``.
        std: Độ lệch chuẩn theo kênh, đi cùng ``mean``.

    Returns:
        Tensor ``float32``.
    """
    from PIL import Image

    arrays = [
        np.asarray(image.convert("RGB").resize((image_size, image_size), Image.BICUBIC), dtype=np.float32)
        for image in images
    ]
    batch = torch.from_numpy(np.stack(arrays)).div_(255.0).permute(0, 3, 1, 2).contiguous()
    if mean is not None:
        batch = (batch - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    return batch


def create_backbone(backbone: str, device: str = "cpu", model_name: Optional[str] = None):
    """Tạo ViT-B/16 pre-trained từ timm, bỏ đầu phân loại.

    Args:
        backbone: ``"in1k"`` hoặc ``"in21k"``.
        device: Thiết bị tính toán.
        model_name: Ghi đè tên mô hình trong ``BACKBONES``.

    Returns:
        Cặp ``(model, info)``; ``info`` chứa tên, tag trọng số, mean, std và phiên bản timm.
    """
    import timm

    name = model_name or BACKBONES[backbone]
    model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(device)
    cfg = model.pretrained_cfg
    info = {
        "backbone": backbone,
        "model_name": name,
        "pretrained_tag": cfg.get("tag"),
        "hf_hub_id": cfg.get("hf_hub_id"),
        "mean": tuple(cfg.get("mean", (0.5, 0.5, 0.5))),
        "std": tuple(cfg.get("std", (0.5, 0.5, 0.5))),
        "timm_version": timm.__version__,
        "feature_dim": model.num_features,
    }
    return model, info


def iter_image_batches(dataset, indices: torch.Tensor, image_column: str, batch_size: int) -> Iterator[list]:
    """Duyệt ảnh theo lô theo danh sách chỉ số, không giải mã cả tập một lúc.

    Args:
        dataset: Dataset Hugging Face.
        indices: Chỉ số các mẫu cần lấy.
        image_column: Tên cột ảnh.
        batch_size: Số ảnh mỗi lô.

    Yields:
        Danh sách ảnh ``PIL.Image``.
    """
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size].tolist()
        yield dataset.select(batch_idx)[image_column]


@torch.inference_mode()
def extract_features(
    model,
    dataset,
    indices: torch.Tensor,
    image_column: str,
    batch_size: int = 128,
    device: str = "cpu",
    normalize: bool = True,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Chạy ViT trên các mẫu đã chọn và trả về đặc trưng.

    Args:
        model: Mô hình trả về vector đặc trưng ``(B, d)``.
        dataset: Dataset Hugging Face.
        indices: Chỉ số các mẫu.
        image_column: Tên cột ảnh.
        batch_size: Số ảnh mỗi lô.
        device: Thiết bị tính toán.
        normalize: Có chuẩn hóa mean/std hay không.
        mean: Trung bình theo kênh khi ``normalize=True``.
        std: Độ lệch chuẩn theo kênh khi ``normalize=True``.
        image_size: Cạnh ảnh đưa vào mô hình.

    Returns:
        Tensor ``float32`` ``(N, d)`` trên CPU.
    """
    outputs = []
    for images in iter_image_batches(dataset, indices, image_column, batch_size):
        batch = preprocess(images, image_size, mean if normalize else None, std if normalize else None)
        outputs.append(model(batch.to(device)).float().cpu())
    return torch.cat(outputs)


def features_path(dataset: str, backbone: str, normalize: bool) -> Path:
    """Đường dẫn mặc định của file đặc trưng.

    Args:
        dataset: Tên dataset E2.
        backbone: ``"in1k"`` hoặc ``"in21k"``.
        normalize: Có chuẩn hóa mean/std hay không.

    Returns:
        ``data/features/<dataset>-<backbone>-<norm|raw>.pt``.
    """
    return data_dir() / "features" / f"{dataset}-{backbone}-{'norm' if normalize else 'raw'}.pt"


def save_features(path: Union[str, Path], payload: dict) -> None:
    """Lưu đặc trưng và thông tin trích ra file ``.pt``.

    Args:
        path: Đường dẫn file.
        payload: Từ điển gồm ``train_x``, ``train_y``, ``test_x``, ``test_y``, ``meta``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_features(path: Union[str, Path]) -> dict:
    """Nạp file đặc trưng đã lưu.

    Args:
        path: Đường dẫn file.

    Returns:
        Từ điển gồm ``train_x``, ``train_y``, ``test_x``, ``test_y``, ``meta``.

    Raises:
        FileNotFoundError: Nếu chưa trích đặc trưng.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Hãy chạy scripts/extract_vit_features.py trước.")
    return torch.load(path, map_location="cpu", weights_only=True)
