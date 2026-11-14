"""Bộ phân loại so sánh: Nearest Class Mean (NCM).

NCM chỉ dùng trung bình lớp, không khử tương quan. Nó là dòng "NCM" trong Bảng 1
và Bảng 7 của bài (Benchmark F), và cũng là "bỏ khử tương quan" ở E3.
Bài không nói NCM dùng khoảng cách nào, nên nhóm hỗ trợ ba cách và ghi lại lựa chọn.
"""
from __future__ import annotations

import torch

NCM_METRICS = ("euclidean", "cosine", "dot")


def class_means(x: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Tính trung bình của từng lớp.

    Args:
        x: Đặc trưng ``(N, d)``.
        y: Nhãn ``int64`` ``(N,)``.
        num_classes: Số lớp ``C``.

    Returns:
        Tensor ``(C, d)``; lớp không có mẫu có trung bình bằng 0.
    """
    sums = torch.zeros(num_classes, x.shape[1], dtype=x.dtype, device=x.device)
    sums.index_add_(0, y.to(x.device), x)
    counts = torch.bincount(y.to(x.device), minlength=num_classes).clamp_min(1).to(x.dtype)
    return sums / counts.unsqueeze(1)


def predict_ncm(x: torch.Tensor, means: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    """Chọn lớp có trung bình gần nhất.

    Args:
        x: Đặc trưng ``(B, d)``.
        means: Trung bình lớp ``(C, d)``.
        metric: ``"euclidean"`` (khoảng cách nhỏ nhất), ``"cosine"`` (cosine lớn nhất)
            hoặc ``"dot"`` (tích vô hướng lớn nhất, đúng công thức (1) của bài).

    Returns:
        Nhãn dự đoán ``(B,)``.

    Raises:
        ValueError: Nếu ``metric`` không hợp lệ.
    """
    if metric == "euclidean":
        return torch.cdist(x, means).argmin(dim=1)
    if metric == "cosine":
        x_unit = torch.nn.functional.normalize(x, dim=1)
        means_unit = torch.nn.functional.normalize(means, dim=1)
        return (x_unit @ means_unit.T).argmax(dim=1)
    if metric == "dot":
        return (x @ means.T).argmax(dim=1)
    raise ValueError(f"metric phải là một trong {NCM_METRICS}, nhận được '{metric}'")
