"""Ghép các khối của RanDumb thành một lượt học và đánh giá.

Luồng xử lý: ảnh đã chuẩn hóa → duỗi → biểu diễn ngẫu nhiên → tích lũy thống kê
→ co OAS → giải LDA → dự đoán. Module này không tải dữ liệu và không log W&B, để
có thể kiểm thử trên dữ liệu giả.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch

from randumb.classifier import fit_lda, oas_shrink, predict
from randumb.data import flatten, iter_batches
from randumb.features import RandomFourierFeatures, RPReLU
from randumb.stats import SufficientStats
from randumb.tracking import Timer

FEATURES = ("rff", "rp-relu", "identity")


class IdentityFeatures:
    """Giữ nguyên đầu vào, dùng khi đặc trưng đã có sẵn (ViT ở E2) hoặc bỏ RFF (E3).

    Args:
        device: Thiết bị tính toán.
        dtype: Kiểu số của đặc trưng.
    """

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = dtype

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Chuyển đầu vào sang đúng thiết bị và kiểu số.

        Args:
            x: Tensor ``(B, d)``.

        Returns:
            Tensor ``(B, d)``.
        """
        return x.to(device=self.device, dtype=self.dtype)
SHRINKS = ("oas", "oas_per_class", "none")


@dataclass
class PipelineConfig:
    """Cấu hình của một lượt học.

    Attributes:
        dim: Số chiều D của biểu diễn ngẫu nhiên (bỏ qua với ``identity``).
        gamma: Tham số γ của RFF (bỏ qua với ``rp-relu``).
        lam: Hệ số ridge λ cố định, hoặc alpha khi lam_mode="relative".
        lam_mode: "fixed" hoặc lambda = alpha * trace(cov) / D.
        seed: Seed của phép chiếu ngẫu nhiên.
        features: ``"rff"`` hoặc ``"rp-relu"``.
        shrink: ``"oas"`` (gộp), ``"oas_per_class"`` (co từng lớp rồi gộp), ``"none"``.
        flip: Có thêm ảnh lật ngang khi train hay không.
        batch_size: Số ảnh gốc mỗi lô.
        device: Thiết bị tính toán.
        stats_dtype: Kiểu số của thống kê tích lũy và phép giải.
        feature_dtype: Kiểu số của đặc trưng.
    """

    dim: int
    gamma: float
    lam: float
    seed: int
    features: str = "rff"
    shrink: str = "oas"
    flip: bool = False
    batch_size: int = 512
    device: str = "cpu"
    stats_dtype: torch.dtype = torch.float64
    feature_dtype: torch.dtype = torch.float32
    lam_mode: str = "fixed"


def resolve_gamma(gamma: Union[str, float], train_x: torch.Tensor) -> float:
    """Đổi γ dạng chữ sang số.

    ``"scale"`` cho ``1 / (d · Var(X))``, tính trên toàn bộ phần tử của tập train đã
    chuẩn hóa, với ``d`` là số chiều sau khi duỗi.

    Args:
        gamma: Số thực, chuỗi số, hoặc ``"scale"``.
        train_x: Ảnh train đã chuẩn hóa ``(N, H, W, C)``.

    Returns:
        Giá trị γ.
    """
    if isinstance(gamma, str) and gamma == "scale":
        x = flatten(train_x)
        return float(1.0 / (x.shape[1] * x.to(torch.float64).var().item()))
    return float(gamma)


def build_features(cfg: PipelineConfig, input_dim: int) -> Union[RandomFourierFeatures, RPReLU, IdentityFeatures]:
    """Tạo phép chiếu ngẫu nhiên theo cấu hình; ``"identity"`` thì giữ nguyên đầu vào.

    Args:
        cfg: Cấu hình của lượt học.
        input_dim: Số chiều của ảnh sau khi duỗi.

    Returns:
        Đối tượng có phương thức ``transform``.

    Raises:
        ValueError: Nếu ``cfg.features`` không hợp lệ.
    """
    if cfg.features == "rff":
        return RandomFourierFeatures(input_dim, cfg.dim, cfg.gamma, cfg.seed, cfg.device, cfg.feature_dtype)
    if cfg.features == "rp-relu":
        return RPReLU(input_dim, cfg.dim, cfg.seed, cfg.device, cfg.feature_dtype)
    if cfg.features == "identity":
        return IdentityFeatures(cfg.device, cfg.feature_dtype)
    raise ValueError(f"features phải là một trong {FEATURES}, nhận được '{cfg.features}'")


def fit_and_evaluate(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    num_classes: int,
    cfg: PipelineConfig,
    stats_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Học RanDumb trong một lượt qua tập train rồi đánh giá trên tập eval.

    Args:
        train_x: Ảnh train đã chuẩn hóa ``(N, H, W, C)``.
        train_y: Nhãn train ``(N,)``.
        eval_x: Ảnh đánh giá đã chuẩn hóa, cùng kích thước ảnh với train.
        eval_y: Nhãn đánh giá.
        num_classes: Số lớp.
        cfg: Cấu hình của lượt học.
        stats_path: Nếu có, lưu thống kê tích lũy ra file này để E5 dùng lại.

    Returns:
        Từ điển gồm ``accuracy`` (%), ``per_class_accuracy`` (%), ``rho``,
        ``n_train_seen`` và thời gian từng bước (giây).

    Raises:
        ValueError: Nếu ``cfg.shrink`` không hợp lệ.
    """
    if cfg.shrink not in SHRINKS:
        raise ValueError(f"shrink phải là một trong {SHRINKS}, nhận được '{cfg.shrink}'")
    if cfg.lam_mode not in ("fixed", "relative"):
        raise ValueError("lam_mode phải là fixed hoặc relative")
    if cfg.shrink == "oas_per_class" and stats_path is not None:
        raise ValueError("Chưa hỗ trợ lưu sufficient stats cho OAS từng lớp")

    input_dim = flatten(train_x[:1]).shape[1]
    features = build_features(cfg, input_dim)
    dim = input_dim if cfg.features == "identity" else cfg.dim
    time_features = time_stats = 0.0

    def accumulate(x, y, classes):
        nonlocal time_features, time_stats
        stats = SufficientStats(dim, classes, cfg.device, cfg.stats_dtype)
        seen = 0
        for x_batch, y_batch in iter_batches(x, y, cfg.batch_size, cfg.flip):
            with Timer() as timer:
                phi = features.transform(x_batch)
            time_features += timer.seconds
            with Timer() as timer:
                stats.update(phi, y_batch)
            time_stats += timer.seconds
            seen += len(y_batch)
        return stats, seen

    rho = None
    rho_per_class = None
    if cfg.shrink == "oas_per_class":
        # E8: xử lý các lớp lần lượt, mỗi ảnh chỉ dùng một lần. Không giữ C
        # covariance D×D cùng lúc; đây là đánh giá cuối luồng class-incremental.
        cov = torch.zeros(dim, dim, device=cfg.device, dtype=cfg.stats_dtype)
        means = torch.zeros(num_classes, dim, device=cfg.device, dtype=cfg.stats_dtype)
        rho_per_class = [None] * num_classes
        n_train_seen = 0
        for c in range(num_classes):
            indices = (train_y == c).nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                continue
            stats, seen = accumulate(train_x[indices], torch.zeros_like(train_y[indices]), 1)
            with Timer() as timer:
                class_means, class_cov = stats.finalize(consume=True)
                class_cov, rho_c = oas_shrink(class_cov, seen, inplace=True)
                cov.add_(class_cov, alpha=seen)
                means[c] = class_means[0]
                rho_per_class[c] = rho_c
                del class_cov, stats
            time_stats += timer.seconds
            n_train_seen += seen
        if n_train_seen == 0:
            raise ValueError("Tập train rỗng")
        cov.div_(n_train_seen)
    else:
        stats, n_train_seen = accumulate(train_x, train_y, num_classes)
        if stats_path is not None:
            stats.save(stats_path)

    with Timer() as solve_timer:
        if cfg.shrink != "oas_per_class":
            means, cov = stats.finalize(consume=True)
            del stats
            if cfg.shrink == "oas":
                cov, rho = oas_shrink(cov, n_train_seen, inplace=True)
        mean_variance = float(cov.trace() / dim)
        lambda_effective = cfg.lam * mean_variance if cfg.lam_mode == "relative" else cfg.lam
        weights = fit_lda(means, cov, lambda_effective, inplace=True)
        del cov

    predictions = []
    with Timer() as predict_timer:
        for x_batch, _ in iter_batches(eval_x, eval_y, cfg.batch_size, flip=False):
            phi = features.transform(x_batch).to(cfg.stats_dtype)
            predictions.append(predict(phi, weights, means).cpu())
    predictions = torch.cat(predictions)

    correct = predictions == eval_y
    per_class = [
        float(correct[eval_y == c].float().mean() * 100) if (eval_y == c).any() else float("nan")
        for c in range(num_classes)
    ]
    return {
        "accuracy": float(correct.float().mean() * 100),
        "per_class_accuracy": per_class,
        "rho": rho,
        "rho_per_class": rho_per_class,
        "mean_variance": mean_variance,
        "lambda_effective": lambda_effective,
        "n_train_seen": n_train_seen,
        "time_features_s": time_features,
        "time_stats_s": time_stats,
        "time_solve_s": solve_timer.seconds,
        "time_predict_s": predict_timer.seconds,
    }
