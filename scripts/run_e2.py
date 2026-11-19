"""Chạy E2 (Benchmark F): RanDumb hoặc NCM trên đặc trưng ViT đã trích, log lên W&B.

Cần chạy scripts/extract_vit_features.py trước. Ví dụ:

    python scripts/run_e2.py --dataset cifar100 --backbone in21k --method randumb --tag lab:reproduction
    python scripts/run_e2.py --dataset cifar100 --backbone in21k --method ncm --tag lab:reproduction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from randumb.baselines import NCM_METRICS, class_means, predict_ncm  # noqa: E402
from randumb.pipeline import PipelineConfig, fit_and_evaluate  # noqa: E402
from randumb.tracking import Timer, init_run, relative_deviation, set_seed  # noqa: E402
from randumb.vit_features import (  # noqa: E402
    BACKBONES, E2_DATASETS, features_path, load_features, resplit_features,
)

# Số của bài (Docs/Instruction.md, mục 4.4; Bảng 1 phải và Bảng 7).
PAPER_E2 = {
    ("in21k", "randumb"): {"cifar100": 86.8, "imagenet-a": 42.2, "imagenet-r": 64.9, "cub": 88.5},
    ("in21k", "ncm"): {"cifar100": 83.5, "imagenet-a": 41.4, "imagenet-r": 54.8, "cub": 86.5},
    ("in1k", "randumb"): {"cifar100": 84.5, "imagenet-a": 49.5, "imagenet-r": 66.9, "cub": 88.0},
    ("in1k", "ncm"): {"cifar100": 78.3, "imagenet-a": 44.3, "imagenet-r": 62.5, "cub": 84.8},
}


def parse_args(argv=None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh.

    Args:
        argv: Danh sách tham số, ``None`` để đọc từ ``sys.argv``.

    Returns:
        Các tham số của run.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(E2_DATASETS))
    parser.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    parser.add_argument("--method", required=True, choices=["randumb", "ncm"])
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="chọn file đặc trưng đã trích có hay không chuẩn hóa mean/std")
    parser.add_argument("--features-file", default=None, help="ghi đè đường dẫn file đặc trưng")
    parser.add_argument("--lam", type=float, default=0.0,
                        help="λ của RanDumb; bài không nêu cho Benchmark F, code gốc dùng 0")
    parser.add_argument("--shrink", choices=["oas", "none"], default="oas")
    parser.add_argument("--ncm-metric", choices=NCM_METRICS, default="euclidean")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="chỉ cho ImageNet-R/-A: gộp train+test rồi chia lại 80/20 theo seed này")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--group", default="E2")
    parser.add_argument("--tag", action="append", default=[], help="thẻ W&B, lặp lại được")
    parser.add_argument("--name", default=None, help="mặc định <group>-<dataset>-<backbone>-<method>-seed<seed>")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.split_seed is not None and E2_DATASETS[args.dataset].test_split is not None:
        parser.error(f"--split-seed chỉ dùng cho dataset tự chia; {args.dataset} đã có tập test riêng")
    return args


def default_name(args: argparse.Namespace) -> str:
    """Tên run W&B, phân biệt khoảng cách NCM và cách chia lại.

    Args:
        args: Tham số của run.

    Returns:
        Ví dụ ``E2-imagenet-a-in1k-ncm-cosine-split1-seed0``.
    """
    method = args.method
    if method == "ncm" and args.ncm_metric != "euclidean":
        method = f"ncm-{args.ncm_metric}"
    split = f"-split{args.split_seed}" if args.split_seed is not None else ""
    return f"{args.group}-{args.dataset}-{args.backbone}-{method}{split}-seed{args.seed}"


def run_ncm(train_x, train_y, test_x, test_y, num_classes: int, metric: str, device: str) -> dict:
    """Học và đánh giá NCM.

    Args:
        train_x: Đặc trưng train ``(N, d)``.
        train_y: Nhãn train.
        test_x: Đặc trưng test.
        test_y: Nhãn test.
        num_classes: Số lớp.
        metric: Khoảng cách dùng cho NCM.
        device: Thiết bị tính toán.

    Returns:
        Từ điển gồm ``accuracy`` (%) và thời gian.
    """
    with Timer() as fit_timer:
        means = class_means(train_x.to(device, torch.float64), train_y, num_classes)
    with Timer() as predict_timer:
        predictions = predict_ncm(test_x.to(device, torch.float64), means, metric).cpu()
    return {
        "accuracy": float((predictions == test_y).float().mean() * 100),
        "rho": None,
        "time_solve_s": fit_timer.seconds,
        "time_predict_s": predict_timer.seconds,
    }


def main(argv=None) -> dict:
    """Chạy một tổ hợp dataset × backbone × phương pháp, in JSON và log W&B.

    Args:
        argv: Danh sách tham số, ``None`` để đọc từ ``sys.argv``.

    Returns:
        Từ điển các chỉ số đã log.
    """
    args = parse_args(argv)
    set_seed(args.seed)
    path = Path(args.features_file) if args.features_file else features_path(args.dataset, args.backbone, args.normalize)
    data = load_features(path)
    spec = E2_DATASETS[args.dataset]
    if args.split_seed is not None:
        data = resplit_features(data, spec.test_fraction, args.split_seed)
    meta = data["meta"]
    num_classes = spec.num_classes

    if args.method == "randumb":
        cfg = PipelineConfig(dim=0, gamma=0.0, lam=args.lam, seed=args.seed, features="identity",
                             shrink=args.shrink, flip=False, batch_size=args.batch_size, device=args.device,
                             stats_dtype=torch.float64, feature_dtype=torch.float64)
        result = fit_and_evaluate(data["train_x"], data["train_y"], data["test_x"], data["test_y"], num_classes, cfg)
    else:
        result = run_ncm(data["train_x"], data["train_y"], data["test_x"], data["test_y"], num_classes,
                         args.ncm_metric, args.device)

    paper = PAPER_E2[(args.backbone, args.method)].get(args.dataset)
    metrics = {
        "final_accuracy": result["accuracy"],
        "paper_accuracy": paper,
        "relative_deviation_pct": relative_deviation(result["accuracy"], paper) if paper else None,
        "rho": result["rho"],
        "time_solve_s": result["time_solve_s"],
        "time_predict_s": result["time_predict_s"],
        "n_train": len(data["train_y"]),
        "n_test": len(data["test_y"]),
    }
    name = args.name or default_name(args)
    if args.wandb:
        config = {
            **{k: v for k, v in vars(args).items() if k not in ("wandb", "entity")},
            **{f"feat_{k}": v for k, v in meta.items()},
            "features_path": str(path),
        }
        run = init_run(name, args.group, args.tag or ["lab:reproduction"], config, entity=args.entity)
        run.log(metrics)
        metrics["wandb_url"] = run.url
        run.finish()
    print(json.dumps({"name": name, **metrics}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
