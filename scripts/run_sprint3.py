"""Chạy E1 và E8 (pha A, B) tuần tự, tiếp tục từ JSON đã hoàn tất và tổng hợp bảng kết quả.

Mặc định chỉ in kế hoạch. Dùng --execute trên máy đủ bộ nhớ, đã đăng nhập W&B.
E1: 15 run. E8-A: 32 run, chuẩn hóa × γ. E8-B: 24 run, shrink × λ, alpha hiệu chỉnh tại D=5000.
E8 chạy seed 0 trên CIFAR-10/100 với D = 5.000 và 25.000 (Docs/Instruction.md, mục 4.6).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from randumb.provenance import code_version

PAPER = {"mnist": (1e-6, 98.3), "cifar10": (1e-5, 55.6), "cifar100": (1e-5, 28.6),
         "tiny-imagenet": (1e-4, 11.6), "mini-imagenet": (1e-4, 17.7)}


GROUPS = ("E1", "E8", "E8-A", "E8-B", "all")
# E8 pha A (Docs/Instruction.md, mục 4.6): chuẩn hóa × γ. Lưới đầy đủ ở D = 5.000; ở D lớn chỉ
# giữ γ = 1,0 và `scale` để vừa giờ GPU. Bỏ tổ hợp L2 + γ = 1,0 vì E8-B (oas, fixed) đã chạy đúng nó.
E8A_NORMS = ("zscore", "l2", "unit")
E8A_GAMMAS = {5000: ("1.0", "scale*0.5", "scale", "scale*2"), 25000: ("1.0", "scale")}


def gamma_tag(gamma: str) -> str:
    """Tên ngắn của γ trong tên run, ví dụ ``scale*0.5`` thành ``scale0.5``."""
    return "1" if gamma == "1.0" else gamma.replace("*", "")


def make_jobs(group="all") -> list[dict]:
    """Lập các run đã chốt, fixed trước relative để có covariance tham chiếu."""
    jobs = []
    if group in ("all", "E1"):
        for dataset, (lam, _) in PAPER.items():
            for seed in (0, 1, 2):
                jobs.append(dict(group="E1", dataset=dataset, dim=25000, seed=seed,
                                 shrink="oas", lam_mode="fixed", lam=lam, norm="l2", gamma="1.0",
                                 name=f"E1-{dataset}-D25000-seed{seed}"))
    if group in ("all", "E8", "E8-A"):
        for dim, gammas in E8A_GAMMAS.items():
            for dataset in ("cifar10", "cifar100"):
                for norm in E8A_NORMS:
                    for gamma in gammas:
                        if (norm, gamma) == ("l2", "1.0"):
                            continue
                        jobs.append(dict(group="E8", dataset=dataset, dim=dim, seed=0,
                                         shrink="oas", lam_mode="fixed", lam=1e-5, norm=norm, gamma=gamma,
                                         name=f"E8-A-{dataset}-D{dim}-{norm}-g{gamma_tag(gamma)}-seed0"))
    if group in ("all", "E8", "E8-B"):
        for mode in ("fixed", "relative"):
            for dataset in ("cifar10", "cifar100"):
                for dim in (5000, 25000):
                    for shrink in ("oas", "oas_per_class", "none"):
                        jobs.append(dict(group="E8", dataset=dataset, dim=dim, seed=0,
                                         shrink=shrink, lam_mode=mode, lam=1e-5, norm="l2", gamma="1.0",
                                         name=f"E8-B-{dataset}-D{dim}-{shrink}-{mode}-seed0"))
    return jobs


def expected_config(job, source_hash, device, batch_size) -> dict:
    """Các trường cần khớp trước khi dùng lại kết quả; alpha phải được resolve trước."""
    gamma = job.get("gamma", "1.0")
    fixed_gamma = {} if gamma.startswith("scale") else {"gamma_value": float(gamma)}
    return {**{k: job[k] for k in ("dataset", "dim", "seed", "shrink", "lam_mode", "lam", "group")},
            "norm": job.get("norm", "l2"), "gamma": gamma, **fixed_gamma,
            "dtype": "float64", "feature_dtype": "float64",
            "features": "rff", "flip": job["dataset"] != "mnist", "eval_split": "test",
            "max_train": None, "max_eval": None, "batch_size": batch_size, "device": device,
            "source_sha256": source_hash, "calibration_run": job.get("calibration_run")}


def completed_record(path, job, source_hash, device, batch_size):
    """Chỉ công nhận JSON đúng cấu hình/source, metric hợp lệ và có URL W&B."""
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = expected_config(job, source_hash, device, batch_size)
    actual = record.get("config", {})
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches or record.get("name") != job["name"]:
        raise ValueError(f"{path}: kết quả khác cấu hình/source ({mismatches}); dùng thư mục output mới")
    accuracy = record.get("final_accuracy")
    if not isinstance(accuracy, (float, int)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
        raise ValueError(f"{path}: accuracy không hợp lệ")
    url = record.get("wandb_url", "")
    if not url.startswith("https://") or "/runs/" not in url:
        raise ValueError(f"{path}: thiếu URL W&B; chưa đủ bằng chứng hoàn tất run")
    return record


def resolve_job(job, output_dir, source_hash, device, batch_size):
    """Hiệu chỉnh alpha bằng trace covariance train tại D=5000, không dùng accuracy test."""
    job = dict(job)
    if job["lam_mode"] == "relative":
        ref = dict(job, dim=5000, lam_mode="fixed", lam=1e-5)
        ref["name"] = f"E8-B-{job['dataset']}-D5000-{job['shrink']}-fixed-seed0"
        record = completed_record(output_dir / (ref["name"] + ".json"), ref, source_hash, device, batch_size)
        if record is None:
            raise ValueError(f"Thiếu calibration run {ref['name']}")
        mean_variance = record.get("mean_variance")
        if mean_variance is None or not math.isfinite(mean_variance) or mean_variance <= 0:
            raise ValueError("Trace covariance tham chiếu phải dương và hữu hạn")
        job["lam"] = 1e-5 / mean_variance
        job["calibration_run"] = record["wandb_url"]
    return job


def build_command(job, output, device, batch_size):
    """Tạo argv rõ ràng, không dựa vào mặc định CLI hoặc nội suy qua shell."""
    argv = [sys.executable, str(ROOT / "scripts" / "run_experiment.py")]
    values = {"dataset": job["dataset"], "group": job["group"], "name": job["name"],
              "dim": job["dim"], "seed": job["seed"], "shrink": job["shrink"],
              "lam-mode": job["lam_mode"], "lam": job["lam"],
              "gamma": job.get("gamma", "1.0"), "norm": job.get("norm", "l2"),
              "features": "rff", "dtype": "float64", "feature-dtype": "float64",
              "eval-split": "test", "batch-size": batch_size, "device": device, "output": str(output)}
    for key, value in values.items():
        argv.extend(["--" + key, str(value)])
    argv.extend(["--no-flip" if job["dataset"] == "mnist" else "--flip", "--tag", "lab:reproduction"])
    if "calibration_run" in job:
        argv.extend(["--calibration-run", job["calibration_run"]])
    return argv


def write_summary(records, output_dir):
    """Bảng paper/team cho các run đã kiểm chứng; không điền số cho run còn thiếu."""
    rows = []
    for record in records:
        cfg = record["config"]
        paper = PAPER[cfg["dataset"]][1]
        rows.append(dict(name=record["name"], dataset=cfg["dataset"], seed=cfg["seed"],
                         group=cfg["group"], dim=cfg["dim"], norm=cfg["norm"], gamma=cfg["gamma"],
                         gamma_value=record.get("gamma"), shrink=cfg["shrink"], lam_mode=cfg["lam_mode"],
                         lambda_effective=record["lambda_effective"], paper_accuracy=paper,
                         team_accuracy=record["final_accuracy"],
                         difference_pp=record["final_accuracy"] - paper,
                         relative_deviation_pct=100 * (record["final_accuracy"] - paper) / paper,
                         wandb_url=record["wandb_url"]))
    if rows:
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    # Chỉ tính mean/std chính thức khi đủ cả 3 seed; không dùng run E8 thay E1.
    aggregates = []
    for dataset, (_, paper) in PAPER.items():
        e1 = [r for r in records if r["config"]["group"] == "E1" and r["config"]["dataset"] == dataset]
        seeds = {r["config"]["seed"] for r in e1}
        complete = len(e1) == 3 and seeds == {0, 1, 2}
        values = [r["final_accuracy"] for r in e1]
        mean = statistics.mean(values) if complete else None
        aggregates.append(dict(dataset=dataset, completed_seeds=len(seeds), expected_seeds=3,
                               status="complete" if complete else "incomplete", paper_accuracy=paper,
                               mean_accuracy=mean, sample_std=statistics.stdev(values) if complete else None,
                               difference_pp=mean - paper if complete else None,
                               relative_deviation_pct=100 * (mean - paper) / paper if complete else None,
                               wandb_urls=";".join(r["wandb_url"] for r in e1)))
    with (output_dir / "e1_aggregate.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)


def collect_records(output_dir, source_hash, device, batch_size):
    """Giữ bảng tổng hợp của các phiên trước khi tiếp tục theo bộ lọc dataset/seed."""
    records = []
    for job in make_jobs():
        path = output_dir / (job["name"] + ".json")
        if path.exists():
            resolved = resolve_job(job, output_dir, source_hash, device, batch_size)
            records.append(completed_record(path, resolved, source_hash, device, batch_size))
    return records


def main(argv=None):
    """In kế hoạch hoặc chạy tuần tự một suite, dừng ngay khi có lỗi."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, default="all")
    parser.add_argument("--dataset", choices=tuple(PAPER), help="chỉ chạy một dataset trong phiên này")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), help="chỉ chạy một seed")
    parser.add_argument("--dim", type=int, choices=(5000, 25000), help="chỉ chạy một D (E8)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "sprint3")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    jobs = [j for j in make_jobs(args.group) if (args.dataset is None or j["dataset"] == args.dataset)
            and (args.seed is None or j["seed"] == args.seed) and (args.dim is None or j["dim"] == args.dim)]
    if not jobs:
        parser.error("Bộ lọc không chọn run nào (E8-B dùng seed 0)")
    if not args.execute:
        print(json.dumps(jobs, indent=2))
        print(f"Planned runs: {len(jobs)}; relative alpha will be calibrated from the D5000 fixed run.")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_hash = code_version()["source_sha256"]
    records = []
    for pending in jobs:
        job = resolve_job(pending, args.out_dir, source_hash, args.device, args.batch_size)
        output = args.out_dir / (job["name"] + ".json")
        record = completed_record(output, job, source_hash, args.device, args.batch_size)
        if record is None:
            print(f"Running {job['name']}", flush=True)
            subprocess.run(build_command(job, output, args.device, args.batch_size), cwd=ROOT, check=True)
            record = completed_record(output, job, source_hash, args.device, args.batch_size)
            if record is None:
                raise RuntimeError(f"Run kết thúc nhưng không có kết quả: {job['name']}")
        else:
            print(f"Resume: verified {job['name']}", flush=True)
        records.append(record)
        write_summary(collect_records(args.out_dir, source_hash, args.device, args.batch_size), args.out_dir)
    print(f"Completed and verified: {len(records)}/{len(jobs)} selected runs")


if __name__ == "__main__":
    main()
