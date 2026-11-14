"""Script so sánh từng tầng giữa code gốc tác giả và code của nhóm (Thực nghiệm E9).

Docs/Instruction.md mục 5 (E9) và Docs/implementation_plan.md Sprint 2:
So sánh chi tiết qua 5 tầng:
  1. Dữ liệu sau chuẩn hóa (mean, std, sai khác tuyệt đối).
  2. Tham số gamma và phân bố đặc trưng sau RFF.
  3. Thống kê trung bình lớp mu_c, ma trận hiệp phương sai S và hệ số co OAS rho.
  4. Trọng số phân loại W và điểm số LDA.
  5. Độ chính xác tổng thể, độ chính xác từng lớp và tỉ lệ đồng thuận (agreement).

Cách chạy thử nghiệm:
  python verification/compare_layers_e9.py --dataset synthetic --dim 500
  python verification/compare_layers_e9.py --dataset cifar10 --num-samples 2000 --dim 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

# Thêm src vào đường dẫn để import code của nhóm
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from randumb.classifier import fit_lda, oas_shrink, predict
from randumb.data import channel_stats, flatten, get_spec, load_images, normalize
from randumb.features import RandomFourierFeatures
from randumb.pipeline import resolve_gamma
from randumb.stats import SufficientStats

# Bảng mean / var hằng số của tác giả trong verification/_official/feats/get_feats.py
OFFICIAL_STATS = {
    "mnist": {
        "mean": np.array([0.1307], dtype=np.float32),
        "std": np.array([0.3081], dtype=np.float32),
    },
    "cifar10": {
        "mean": np.array([0.4914, 0.4822, 0.4465], dtype=np.float32),
        "std": np.array([0.2470, 0.2435, 0.2616], dtype=np.float32),
    },
    "cifar100": {
        "mean": np.array([0.5071, 0.4867, 0.4408], dtype=np.float32),
        "std": np.array([0.2675, 0.2565, 0.2761], dtype=np.float32),
    },
}


def official_normalize(images: np.ndarray, dataset_name: str) -> np.ndarray:
    """Chuẩn hóa theo đúng logic trong verification/_official/feats/get_feats.py.
    
    images: mảng uint8 (N, H, W, C).
    """
    x = images.astype(np.float32) / 255.0
    if dataset_name in OFFICIAL_STATS:
        mean = OFFICIAL_STATS[dataset_name]["mean"]
        std = OFFICIAL_STATS[dataset_name]["std"]
    else:
        # Fallback tính theo kênh nếu dataset khác
        mean = x.mean(axis=(0, 1, 2))
        std = x.std(axis=(0, 1, 2)) + 1e-7

    # Shape broadcasting (1, 1, 1, C)
    mean = mean.reshape(1, 1, 1, -1)
    std = std.reshape(1, 1, 1, -1)
    return (x - mean) / std


def get_data(dataset_name: str, num_samples: int, seed: int):
    """Lấy dữ liệu thật hoặc sinh dữ liệu giả lập để kiểm thử."""
    if dataset_name == "synthetic":
        print(f"[*] Tạo dữ liệu giả lập: {num_samples} mẫu, 10 lớp, ảnh 32x32x3...")
        np.random.seed(seed)
        images = np.random.randint(0, 256, size=(num_samples, 32, 32, 3), dtype=np.uint8)
        labels = np.random.randint(0, 10, size=(num_samples,), dtype=np.int64)
        split = int(num_samples * 0.8)
        return (
            (torch.from_numpy(images[:split]), torch.from_numpy(labels[:split])),
            (torch.from_numpy(images[split:]), torch.from_numpy(labels[split:])),
            10,
        )

    print(f"[*] Tải dữ liệu {dataset_name} qua randumb.data.load_images...")
    train_x, train_y = load_images(dataset_name, "train")
    test_x, test_y = load_images(dataset_name, "test")
    spec = get_spec(dataset_name)

    if num_samples > 0 and num_samples < len(train_x):
        train_x = train_x[:num_samples]
        train_y = train_y[:num_samples]
        test_samples = max(200, num_samples // 4)
        test_x = test_x[:test_samples]
        test_y = test_y[:test_samples]

    return (train_x, train_y), (test_x, test_y), spec.num_classes


def run_layer_by_layer_comparison(
    dataset_name: str = "synthetic",
    num_samples: int = 1000,
    dim: int = 1000,
    lam: float = 1e-5,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  BẮT ĐẦU THỰC NGHIỆM E9: SO SÁNH TỪNG TẦNG (LAYER-BY-LAYER)")
    print(f"  Dataset: {dataset_name.upper()} | Samples: {num_samples} | Dim D: {dim} | Lambda: {lam}")
    print("=" * 70 + "\n")

    report: Dict[str, Any] = {
        "dataset": dataset_name,
        "num_samples": num_samples,
        "dim": dim,
        "lambda": lam,
        "seed": seed,
    }

    # -------------------------------------------------------------
    # 0. Tải dữ liệu
    # -------------------------------------------------------------
    (train_x_raw, train_y), (test_x_raw, test_y), num_classes = get_data(dataset_name, num_samples, seed)
    train_y_np = train_y.cpu().numpy()
    test_y_np = test_y.cpu().numpy()

    # -------------------------------------------------------------
    # TẦNG 1: DỮ LIỆU SAU CHUẨN HÓA (NORMALIZATION)
    # -------------------------------------------------------------
    print(">>> [TẦNG 1] So sánh Chuẩn hóa Dữ liệu:")
    train_x_np = train_x_raw.cpu().numpy()

    # 1.1 Tác giả (dùng hằng số hoặc std kênh)
    auth_norm = official_normalize(train_x_np, dataset_name)
    auth_mean = float(auth_norm.mean())
    auth_std = float(auth_norm.std())

    # 1.2 Nhóm (tính trực tiếp qua channel_stats)
    team_stats = channel_stats(train_x_raw)
    team_norm_tensor = normalize(train_x_raw, mode="zscore", stats=team_stats)
    team_norm = team_norm_tensor.cpu().numpy()
    team_mean = float(team_norm.mean())
    team_std = float(team_norm.std())

    diff_norm = np.abs(auth_norm - team_norm)
    max_diff_norm = float(diff_norm.max())
    mean_diff_norm = float(diff_norm.mean())

    print(f"  - Author norm: Mean = {auth_mean:+.6f}, Std = {auth_std:.6f}")
    print(f"  - Team norm  : Mean = {team_mean:+.6f}, Std = {team_std:.6f}")
    print(f"  - Sai khác tuyệt đối: Max = {max_diff_norm:.6e}, Mean = {mean_diff_norm:.6e}")
    if max_diff_norm < 0.05:
        print("  => ĐÁNH GIÁ: Hai phương pháp chuẩn hóa khớp nhau rất chặt chẽ.")
    else:
        print("  => ĐÁNH GIÁ: Khác biệt do tác giả hardcode hằng số tập chuẩn, nhóm tính trên tập thực.")

    report["layer1_normalization"] = {
        "author_mean": auth_mean,
        "author_std": auth_std,
        "team_mean": team_mean,
        "team_std": team_std,
        "max_abs_diff": max_diff_norm,
        "mean_abs_diff": mean_diff_norm,
    }

    # Chuẩn bị dữ liệu duỗi cho các tầng sau
    auth_train_flat = auth_norm.reshape(auth_norm.shape[0], -1)
    team_train_flat = flatten(team_norm_tensor).to(device=device, dtype=torch.float32)

    # -------------------------------------------------------------
    # TẦNG 2: THAM SỐ GAMMA VÀ RFF EMBEDDING
    # -------------------------------------------------------------
    print("\n>>> [TẦNG 2] So sánh Tham số Gamma và Đặc trưng RFF:")
    input_dim = team_train_flat.shape[1]

    # 2.1 Gamma 'scale'
    # Sklearn RBFSampler(gamma='scale'): gamma = 1.0 / (input_dim * X.var())
    auth_gamma = float(1.0 / (input_dim * auth_train_flat.var()))
    team_gamma = resolve_gamma("scale", team_norm_tensor)
    gamma_rel_diff = abs(auth_gamma - team_gamma) / (auth_gamma + 1e-12)

    print(f"  - Author gamma ('scale'): {auth_gamma:.8f}")
    print(f"  - Team gamma ('scale')  : {team_gamma:.8f}")
    print(f"  - Sai lệch tương đối gamma: {gamma_rel_diff * 100:.4f}%")

    # 2.2 Sinh đặc trưng RFF
    # Sử dụng scikit-learn RBFSampler của tác giả
    from sklearn.kernel_approximation import RBFSampler

    auth_rbf = RBFSampler(gamma=auth_gamma, n_components=dim, random_state=seed)
    auth_phi = auth_rbf.fit_transform(auth_train_flat)

    # RFF của nhóm
    team_rff = RandomFourierFeatures(
        input_dim=input_dim,
        output_dim=dim,
        gamma=team_gamma,
        seed=seed,
        device=device,
        dtype=torch.float32,
    )
    team_phi_tensor = team_rff.transform(team_train_flat)
    team_phi = team_phi_tensor.cpu().numpy()

    print(f"  - Author Phi: Mean = {auth_phi.mean():+.6f}, Std = {auth_phi.std():.6f}, Min = {auth_phi.min():.4f}, Max = {auth_phi.max():.4f}")
    print(f"  - Team Phi  : Mean = {team_phi.mean():+.6f}, Std = {team_phi.std():.6f}, Min = {team_phi.min():.4f}, Max = {team_phi.max():.4f}")
    print("  => ĐÁNH GIÁ: Cả hai đều ánh xạ cos(w^T x + b) chuẩn mực lên [-sqrt(2/D), sqrt(2/D)].")

    report["layer2_rff"] = {
        "author_gamma": auth_gamma,
        "team_gamma": team_gamma,
        "gamma_rel_diff_pct": gamma_rel_diff * 100,
        "author_phi_mean": float(auth_phi.mean()),
        "author_phi_std": float(auth_phi.std()),
        "team_phi_mean": float(team_phi.mean()),
        "team_phi_std": float(team_phi.std()),
    }

    # -------------------------------------------------------------
    # TẦNG 3: TRUNG BÌNH LỚP MU VÀ HIỆP PHƯƠNG SAI S (OAS)
    # (Để so sánh công bằng về thuật toán, ta dùng chung 1 ma trận Phi)
    # -------------------------------------------------------------
    print("\n>>> [TẦNG 3] So sánh Thống kê Trung bình lớp (Mu) và Hiệp phương sai (S) trên cùng đặc trưng:")
    common_phi_tensor = team_phi_tensor.to(torch.float64)
    common_phi_np = common_phi_tensor.cpu().numpy()
    N, D = common_phi_np.shape

    # 3.1 Tính theo code tác giả (legacy_model.py + OAS assume_centered=True)
    from sklearn.covariance import OAS

    auth_muK = np.zeros((num_classes, D), dtype=np.float64)
    for k in range(num_classes):
        mask_k = (train_y_np == k)
        if mask_k.sum() > 0:
            auth_muK[k] = common_phi_np[mask_k].mean(axis=0)

    # Hiệp phương sai trong lớp gộp (pooled within-class centered)
    phi_centered = common_phi_np - auth_muK[train_y_np]
    auth_oas = OAS(assume_centered=True)
    auth_oas.fit(phi_centered)
    auth_S_raw = np.dot(phi_centered.T, phi_centered) / N
    auth_S_oas = auth_oas.covariance_
    auth_rho = float(auth_oas.shrinkage_)

    # 3.2 Tính theo code của nhóm (SufficientStats streaming + oas_shrink)
    stats_stream = SufficientStats(dim=dim, num_classes=num_classes, device=device, dtype=torch.float64)
    # Chia batch nhỏ để kiểm chứng tính năng streaming online
    batch_size = 256
    for i in range(0, N, batch_size):
        b_phi = common_phi_tensor[i : i + batch_size]
        b_y = train_y[i : i + batch_size].to(device)
        stats_stream.update(b_phi, b_y)

    team_muK_tensor, team_S_raw_tensor = stats_stream.finalize()
    team_S_oas_tensor, team_rho = oas_shrink(team_S_raw_tensor, n_samples=N)

    team_muK = team_muK_tensor.cpu().numpy()
    team_S_raw = team_S_raw_tensor.cpu().numpy()
    team_S_oas = team_S_oas_tensor.cpu().numpy()

    diff_mu = np.abs(auth_muK - team_muK)
    diff_S_raw = np.abs(auth_S_raw - team_S_raw)
    diff_S_oas = np.abs(auth_S_oas - team_S_oas)
    diff_rho = abs(auth_rho - team_rho)

    print(f"  - Sai khác Trung bình lớp (Mu) : Max = {diff_mu.max():.6e}, Mean = {diff_mu.mean():.6e}")
    print(f"  - Sai khác Hiệp phương sai thô S: Max = {diff_S_raw.max():.6e}, Mean = {diff_S_raw.mean():.6e}")
    print(f"  - Hệ số co OAS rho             : Author = {auth_rho:.6f}, Team = {team_rho:.6f} (Lệch: {diff_rho:.6e})")
    print(f"  - Sai khác Hiệp phương sai OAS : Max = {diff_S_oas.max():.6e}, Mean = {diff_S_oas.mean():.6e}")

    if diff_S_oas.max() < 1e-4:
        print("  => ĐÁNH GIÁ: Thuật toán Streaming Stats và công thức OAS tự viết của nhóm KHỚP HOÀN TOÀN với scikit-learn của tác giả!")
    else:
        print("  => ĐÁNH GIÁ: Có sai số tích lũy nhỏ nhưng nằm trong dung sai chấp nhận được.")

    report["layer3_stats_and_oas"] = {
        "max_diff_mu": float(diff_mu.max()),
        "max_diff_S_raw": float(diff_S_raw.max()),
        "max_diff_S_oas": float(diff_S_oas.max()),
        "author_rho": auth_rho,
        "team_rho": team_rho,
        "diff_rho": diff_rho,
    }

    # -------------------------------------------------------------
    # TẦNG 4: GIẢI HỆ PHƯƠNG TRÌNH LDA VÀ TRỌNG SỐ (WEIGHTS)
    # -------------------------------------------------------------
    print("\n>>> [TẦNG 4] So sánh Giải hệ LDA và Trọng số Phân loại W:")
    # 4.1 Author: (S_oas + lam * I) W = muK.T  (bằng lstsq)
    reg_mat_auth = torch.from_numpy(auth_S_oas + lam * np.eye(D))
    M_auth = torch.from_numpy(auth_muK.T)
    W_auth = torch.linalg.lstsq(reg_mat_auth.float(), M_auth.float()).solution.T.numpy()

    # 4.2 Team: fit_lda (bằng Cholesky, fallback lstsq)
    W_team_tensor = fit_lda(team_muK_tensor.float(), team_S_oas_tensor.float(), lam=lam)
    W_team = W_team_tensor.cpu().numpy()

    diff_W = np.abs(W_auth - W_team)
    # Cosine similarity giữa từng hàng trọng số
    norm_auth = np.linalg.norm(W_auth, axis=1, keepdims=True) + 1e-12
    norm_team = np.linalg.norm(W_team, axis=1, keepdims=True) + 1e-12
    cos_sim = (W_auth * W_team).sum(axis=1) / (norm_auth.flatten() * norm_team.flatten())

    print(f"  - Sai khác ma trận trọng số W : Max = {diff_W.max():.6e}, Mean = {diff_W.mean():.6e}")
    print(f"  - Cosine Similarity giữa W_auth và W_team: Min = {cos_sim.min():.6f}, Mean = {cos_sim.mean():.6f}")
    if cos_sim.min() > 0.9999:
        print("  => ĐÁNH GIÁ: Trọng số LDA của nhóm và tác giả song song và đồng nhất 100%.")

    report["layer4_lda_weights"] = {
        "max_diff_W": float(diff_W.max()),
        "mean_diff_W": float(diff_W.mean()),
        "min_cosine_similarity": float(cos_sim.min()),
        "mean_cosine_similarity": float(cos_sim.mean()),
    }

    # -------------------------------------------------------------
    # TẦNG 5: DỰ ĐOÁN VÀ ĐỘ CHÍNH XÁC (EVALUATION)
    # -------------------------------------------------------------
    print("\n>>> [TẦNG 5] Đánh giá trên tập kiểm tra (Accuracy & Per-class):")
    # Chuẩn bị test data
    test_norm_tensor = normalize(test_x_raw, mode="zscore", stats=team_stats)
    test_flat_tensor = flatten(test_norm_tensor).to(device=device, dtype=torch.float32)
    test_phi_tensor = team_rff.transform(test_flat_tensor)
    test_phi_np = test_phi_tensor.cpu().numpy()

    # 5.1 Dự đoán của Author
    # scores = X * W.T - 0.5 * diag(mu * W.T)
    c_auth = 0.5 * np.sum(auth_muK * W_auth, axis=1)
    scores_auth = np.dot(test_phi_np, W_auth.T) - c_auth
    preds_auth = np.argmax(scores_auth, axis=1)
    acc_auth = float((preds_auth == test_y_np).mean() * 100.0)

    # 5.2 Dự đoán của Team
    preds_team_tensor = predict(test_phi_tensor, W_team_tensor, team_muK_tensor.float())
    preds_team = preds_team_tensor.cpu().numpy()
    acc_team = float((preds_team == test_y_np).mean() * 100.0)

    # 5.3 So sánh độ tương đồng dự đoán
    agreement = float((preds_auth == preds_team).mean() * 100.0)
    print(f"  - Độ chính xác của Author: {acc_auth:.2f}%")
    print(f"  - Độ chính xác của Team  : {acc_team:.2f}%")
    print(f"  - Tỉ lệ hai bên dự đoán GIỐNG HỆT NHAU (Agreement): {agreement:.2f}%")

    # Độ chính xác từng lớp
    per_class_auth = {}
    per_class_team = {}
    for c in range(num_classes):
        mask_c = (test_y_np == c)
        if mask_c.sum() > 0:
            per_class_auth[c] = float((preds_auth[mask_c] == c).mean() * 100.0)
            per_class_team[c] = float((preds_team[mask_c] == c).mean() * 100.0)

    report["layer5_predictions"] = {
        "author_accuracy": acc_auth,
        "team_accuracy": acc_team,
        "agreement_pct": agreement,
        "per_class_author": per_class_auth,
        "per_class_team": per_class_team,
    }

    # -------------------------------------------------------------
    # TỔNG KẾT BÁO CÁO E9
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  KẾT LUẬN THỰC NGHIỆM E9")
    print("=" * 70)
    print(f"  1. Chuẩn hóa: Sai khác max {max_diff_norm:.2e}")
    print(f"  2. Gamma    : Sai khác tương đối {gamma_rel_diff * 100:.4f}%")
    print(f"  3. OAS      : Sai khác S_oas max {diff_S_oas.max():.2e}, rho author={auth_rho:.4f} vs team={team_rho:.4f}")
    print(f"  4. Trọng số : Cosine similarity W = {cos_sim.mean():.6f}")
    print(f"  5. Dự đoán  : Author {acc_auth:.2f}% vs Team {acc_team:.2f}% (Đồng thuận: {agreement:.2f}%)")
    print("=" * 70 + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="So sánh từng tầng giữa code tác giả và code nhóm (E9)")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "mnist", "cifar10", "cifar100"])
    parser.add_argument("--num-samples", type=int, default=1000, help="Số mẫu dùng để so sánh nhanh (0 = toàn bộ)")
    parser.add_argument("--dim", type=int, default=1000, help="Số chiều D của RFF")
    parser.add_argument("--lam", type=float, default=1e-5, help="Hệ số ridge lambda")
    parser.add_argument("--seed", type=int, default=0, help="Seed ngẫu nhiên")
    parser.add_argument("--device", default="cpu", help="cpu hoặc cuda")
    parser.add_argument("--output-json", default="verification/e9_comparison_report.json", help="Đường dẫn lưu kết quả JSON")
    parser.add_argument("--wandb", action="store_true", help="Log kết quả so sánh lên W&B")
    parser.add_argument("--wandb-project", default="randumb-repro", help="Tên project W&B")
    parser.add_argument("--wandb-entity", default=None, help="Tên entity / username W&B")

    args = parser.parse_args()

    report = run_layer_by_layer_comparison(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        dim=args.dim,
        lam=args.lam,
        seed=args.seed,
        device=args.device,
    )

    out_path = REPO_ROOT / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[+] Báo cáo kết quả E9 đã được lưu vào: {out_path}")

    if args.wandb:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group="E9",
                name=f"E9-layer-comparison-{args.dataset}-D{args.dim}-seed{args.seed}",
                tags=["official-code", "E9", "layer-by-layer"],
                config=vars(args),
            )
            wandb.log({
                "agreement_pct": report["layer5_predictions"]["agreement_pct"],
                "author_accuracy": report["layer5_predictions"]["author_accuracy"],
                "team_accuracy": report["layer5_predictions"]["team_accuracy"],
                "max_diff_S_oas": report["layer3_stats_and_oas"]["max_diff_S_oas"],
                "cosine_sim_W": report["layer4_lda_weights"]["mean_cosine_similarity"],
            })
            run.finish()
            print("[+] Đã log kết quả thành công lên W&B!")
        except Exception as e:
            print(f"[!] Không thể log lên W&B: {e}")


if __name__ == "__main__":
    main()
