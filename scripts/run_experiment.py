"""Chạy một thực nghiệm RanDumb theo tham số dòng lệnh và log kết quả lên W&B.

Ví dụ chạy thử trên laptop, không log W&B:

    python scripts/run_experiment.py --dataset mnist --dim 1000 --lam 1e-6 \
        --group smoke --no-wandb --max-train 5000

Ví dụ trên Kaggle:

    python scripts/run_experiment.py --dataset cifar100 --dim 25000 --lam 1e-5 \
        --group E1 --tag lab:reproduction --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Terminal Windows mặc định cp1252 không in được tiếng Việt trong --help.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from randumb.data import NORMALIZATIONS, channel_stats, get_spec, load_images, normalize, train_val_split  # noqa: E402
from randumb.pipeline import FEATURES, PipelineConfig, fit_and_evaluate, resolve_gamma  # noqa: E402
from randumb.tracking import init_run, peak_ram_gb, peak_vram_gb, relative_deviation, set_seed  # noqa: E402

# Độ chính xác của RanDumb trong bài (Docs/Instruction.md, mục 4.3).
PAPER_E1 = {"mnist": 98.3, "cifar10": 55.6, "cifar100": 28.6, "tiny-imagenet": 11.6, "mini-imagenet": 17.7}
DTYPES = {"float32": torch.float32, "float64": torch.float64}


def parse_args(argv=None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh.

    Args:
        argv: Danh sách tham số, ``None`` để đọc từ ``sys.argv``.

    Returns:
        Các tham số của run.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="khóa trong randumb.data.DATASETS")
    parser.add_argument("--dim", type=int, default=25_000, help="số chiều D của đặc trưng")
    parser.add_argument("--features", choices=FEATURES, default="rff")
    parser.add_argument("--gamma", default="scale", help="số thực, hoặc 'scale' = 1/(d·Var(X))")
    parser.add_argument("--lam", type=float, required=True, help="hệ số ridge λ")
    parser.add_argument("--lam-mode", choices=["fixed", "relative"], default="fixed")
    parser.add_argument("--norm", choices=NORMALIZATIONS, default="zscore")
    parser.add_argument("--shrink", choices=["oas", "oas_per_class", "none"], default="oas")
    parser.add_argument("--flip", action=argparse.BooleanOptionalAction, default=None,
                        help="mặc định theo DatasetSpec.use_flip")
    parser.add_argument("--eval-split", choices=["test", "val"], default="test",
                        help="'val' tách validation từ train để chọn cấu hình (E8)")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=None, help="lấy ngẫu nhiên tối đa N ảnh train để chạy thử")
    parser.add_argument("--max-eval", type=int, default=None, help="lấy ngẫu nhiên tối đa N ảnh đánh giá để chạy thử")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float64", help="kiểu số của thống kê và phép giải")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cache-dir", default=None, help="nơi lưu ảnh đã giải mã, mặc định <repo>/data/cache")
    parser.add_argument("--hf-cache-dir", default=None, help="nơi lưu file tải từ HF, mặc định <repo>/data/hf")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                        help="--no-cache để không đọc/ghi file .npz")
    parser.add_argument("--save-stats", default=None, help="lưu A, tong, dem ra file .pt để E5 dùng lại")
    parser.add_argument("--group", required=True, help="nhóm thực nghiệm, ví dụ E1")
    parser.add_argument("--tag", action="append", default=[], help="thẻ W&B, lặp lại được")
    parser.add_argument("--name", default=None, help="tên run, mặc định <group>-<dataset>-D<dim>-seed<seed>")
    parser.add_argument("--entity", default=None, help="team W&B, mặc định lấy từ WANDB_ENTITY")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def random_subset(images: torch.Tensor, labels: torch.Tensor, limit, seed: int):
    """Lấy ngẫu nhiên tối đa ``limit`` mẫu để chạy thử nhanh.

    Args:
        images: Ảnh.
        labels: Nhãn.
        limit: Số mẫu tối đa, ``None`` để giữ nguyên.
        seed: Seed của phép chọn.

    Returns:
        Cặp ``(images, labels)`` đã rút gọn.
    """
    if limit is None or limit >= len(labels):
        return images, labels
    idx = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))[:limit].sort().values
    return images[idx], labels[idx]


def main(argv=None) -> dict:
    """Chạy toàn bộ pipeline, in kết quả dạng JSON và log lên W&B.

    Args:
        argv: Danh sách tham số, ``None`` để đọc từ ``sys.argv``.

    Returns:
        Từ điển các chỉ số đã log.
    """
    args = parse_args(argv)
    if args.shrink == "oas_per_class":
        raise NotImplementedError("--shrink oas_per_class thuộc Sprint 2 (E8 pha B), chưa cài trong phần lõi")
    if args.lam_mode == "relative":
        raise NotImplementedError("--lam-mode relative thuộc Sprint 2 (E8 pha B), chưa cài trong phần lõi")

    set_seed(args.seed)
    spec = get_spec(args.dataset)

    load_kwargs = {"cache_dir": args.cache_dir, "hf_cache_dir": args.hf_cache_dir, "use_cache": args.cache}
    train_images, train_labels = load_images(args.dataset, "train", **load_kwargs)
    if args.eval_split == "val":
        (train_images, train_labels), (eval_images, eval_labels) = train_val_split(
            train_images, train_labels, args.val_fraction, args.seed
        )
    else:
        eval_images, eval_labels = load_images(args.dataset, "test", **load_kwargs)
    train_images, train_labels = random_subset(train_images, train_labels, args.max_train, args.seed)
    eval_images, eval_labels = random_subset(eval_images, eval_labels, args.max_eval, args.seed)

    stats = channel_stats(train_images) if args.norm == "zscore" else None
    train_x = normalize(train_images, args.norm, stats)
    eval_x = normalize(eval_images, args.norm, stats)
    del train_images, eval_images

    cfg = PipelineConfig(
        dim=args.dim,
        gamma=resolve_gamma(args.gamma, train_x),
        lam=args.lam,
        seed=args.seed,
        features=args.features,
        shrink=args.shrink,
        flip=spec.use_flip if args.flip is None else args.flip,
        batch_size=args.batch_size,
        device=args.device,
        stats_dtype=DTYPES[args.dtype],
    )
    config = {
        **{k: v for k, v in vars(args).items() if k not in ("wandb", "entity")},
        "gamma_value": cfg.gamma,
        "flip": cfg.flip,
        "hf_id": spec.hf_id,
        "n_train": len(train_labels),
        "n_eval": len(eval_labels),
        "torch_version": torch.__version__,
    }
    name = args.name or f"{args.group}-{args.dataset}-D{args.dim}-seed{args.seed}"
    run = init_run(name, args.group, args.tag or ["lab:reproduction"], config, entity=args.entity) if args.wandb else None

    result = fit_and_evaluate(train_x, train_labels, eval_x, eval_labels, spec.num_classes, cfg, args.save_stats)

    paper = PAPER_E1[args.dataset]
    metrics = {
        "final_accuracy": result["accuracy"],
        "paper_accuracy": paper,
        "relative_deviation_pct": relative_deviation(result["accuracy"], paper),
        "rho": result["rho"],
        "n_train_seen": result["n_train_seen"],
        "time_features_s": result["time_features_s"],
        "time_stats_s": result["time_stats_s"],
        "time_solve_s": result["time_solve_s"],
        "time_predict_s": result["time_predict_s"],
        "peak_ram_gb": peak_ram_gb(),
        "peak_vram_gb": peak_vram_gb(),
    }
    if run is not None:
        import wandb

        run.log(metrics)
        run.log({"per_class_accuracy": wandb.Table(
            columns=["class", "accuracy"], data=[[c, a] for c, a in enumerate(result["per_class_accuracy"])]
        )})
        metrics["wandb_url"] = run.url
        run.finish()

    print(json.dumps({"name": name, "gamma": cfg.gamma, **metrics}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
