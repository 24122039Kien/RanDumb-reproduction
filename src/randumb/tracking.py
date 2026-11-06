"""Đặt seed và bọc W&B để mọi run được ghi lại theo cùng một cách.

Xem Docs/Instruction.md, mục 2.7 và 4.1. `WANDB_API_KEY` chỉ lấy từ biến môi trường.
"""
from __future__ import annotations

from typing import Any, Iterable


def set_seed(seed: int) -> None:
    """Đặt seed cho `random`, `numpy` và `torch`.

    Args:
        seed: Giá trị seed.
    """
    raise NotImplementedError("Sprint 1")


def environment_info() -> dict[str, Any]:
    """Thu thập phiên bản thư viện và tên GPU để log vào cấu hình run.

    Returns:
        Từ điển thông tin môi trường.
    """
    raise NotImplementedError("Sprint 1")


def init_run(name: str, group: str, tags: Iterable[str], config: dict[str, Any]) -> Any:
    """Khởi tạo một W&B run trong project chung của nhóm.

    Args:
        name: Tên run, ví dụ `E1-cifar100-D25000-seed0`.
        group: Nhóm thực nghiệm, ví dụ `E1`.
        tags: Các thẻ, ví dụ `lab:reproduction`.
        config: Cấu hình đầy đủ của run.

    Returns:
        Đối tượng run của W&B.
    """
    raise NotImplementedError("Sprint 1")


def relative_deviation(team: float, paper: float) -> float:
    """Tính sai lệch tương đối `(team - paper) / |paper| × 100%`.

    Args:
        team: Kết quả của nhóm.
        paper: Kết quả trong bài báo.

    Returns:
        Sai lệch tính theo phần trăm.
    """
    raise NotImplementedError("Sprint 1")
