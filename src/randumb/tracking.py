"""Đặt seed, đo thời gian và bộ nhớ, và bọc W&B để mọi run được ghi lại giống nhau.

Xem Docs/Instruction.md, mục 2.7 và 4.1. `WANDB_API_KEY` chỉ lấy từ biến môi trường
hoặc Kaggle Secrets, không bao giờ ghi vào code.
"""
from __future__ import annotations

import os
import platform
import random
import time
from importlib import metadata
from typing import Any, Iterable, Optional

import numpy as np
import torch

WANDB_ENTITY = "24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm"
WANDB_PROJECT = "randumb-reproduction-24TNT"
TRACKED_PACKAGES = ("numpy", "scipy", "scikit-learn", "torch", "wandb", "datasets", "timm")


def set_seed(seed: int) -> None:
    """Đặt seed cho `random`, `numpy` và `torch` (CPU và mọi GPU).

    Args:
        seed: Giá trị seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_info() -> dict[str, Any]:
    """Thu thập phiên bản Python, thư viện và tên GPU để log vào cấu hình run.

    Returns:
        Từ điển thông tin môi trường; thư viện chưa cài có giá trị ``None``.
    """
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for name in TRACKED_PACKAGES:
        try:
            info[f"version_{name}"] = metadata.version(name)
        except metadata.PackageNotFoundError:
            info[f"version_{name}"] = None
    return info


def init_run(
    name: str,
    group: str,
    tags: Iterable[str],
    config: dict[str, Any],
    project: str = WANDB_PROJECT,
    entity: Optional[str] = None,
    enabled: bool = True,
) -> Any:
    """Khởi tạo một W&B run trong project chung của nhóm.

    Cấu hình được gộp thêm ``environment_info()``. Nếu ``enabled=False``, run chạy ở
    chế độ ``disabled`` để thử trên laptop mà không cần đăng nhập W&B.

    Args:
        name: Tên run, ví dụ ``E1-cifar100-D25000-seed0``.
        group: Nhóm thực nghiệm, ví dụ ``E1``.
        tags: Các thẻ, ví dụ ``lab:reproduction``.
        config: Cấu hình đầy đủ của run.
        project: Tên W&B project.
        entity: Tên team W&B; ``None`` thì lấy từ biến môi trường ``WANDB_ENTITY``,
            nếu không có thì dùng team của nhóm.
        enabled: ``False`` để không gửi gì lên W&B.

    Returns:
        Đối tượng run của W&B.
    """
    import wandb

    return wandb.init(
        project=project,
        entity=entity or os.environ.get("WANDB_ENTITY") or WANDB_ENTITY,
        name=name,
        group=group,
        tags=list(tags),
        config={**config, **environment_info()},
        mode="online" if enabled else "disabled",
    )


def relative_deviation(team: float, paper: float) -> float:
    """Tính sai lệch tương đối ``(team - paper) / |paper| × 100%``.

    Args:
        team: Kết quả của nhóm.
        paper: Kết quả trong bài báo, khác 0.

    Returns:
        Sai lệch tính theo phần trăm.

    Raises:
        ValueError: Nếu ``paper`` bằng 0.
    """
    if paper == 0:
        raise ValueError("paper bằng 0 nên không tính được sai lệch tương đối")
    return (team - paper) / abs(paper) * 100.0


class Timer:
    """Đo thời gian một khối lệnh, có đồng bộ GPU để số đo không bị sớm.

    Ví dụ::

        with Timer() as t:
            ...
        print(t.seconds)
    """

    def __init__(self) -> None:
        self.seconds = 0.0
        self._start = 0.0

    def __enter__(self) -> "Timer":
        _synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        _synchronize()
        self.seconds = time.perf_counter() - self._start


def _synchronize() -> None:
    """Chờ GPU chạy xong các lệnh đang xếp hàng, nếu có GPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_ram_gb() -> Optional[float]:
    """Đọc lượng RAM đỉnh mà tiến trình đã dùng.

    Dùng ``resource`` trên Linux (Kaggle) và ``psutil`` trên Windows.

    Returns:
        Số GB, hoặc ``None`` nếu hệ điều hành không hỗ trợ.
    """
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # Linux trả về KB
    except ImportError:
        pass
    try:
        import psutil

        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)  # chỉ có trên Windows
        return peak / 2**30 if peak is not None else None
    except ImportError:
        return None


def peak_vram_gb() -> Optional[float]:
    """Đọc lượng bộ nhớ GPU đỉnh mà torch đã cấp phát.

    Returns:
        Số GB, hoặc ``None`` nếu không có GPU.
    """
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**30
