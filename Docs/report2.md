# Kết quả chạy notebook kiểm tra môi trường trên Kaggle

**Người làm:** Kiên
**Ngày chạy:** 11/11/2026
**Notebook:** `notebooks/00_kaggle_smoke.ipynb` trên Kaggle, GPU T4 x2
**W&B run:** [`smoke-kaggle-seed0` (6vddfz6r)](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/6vddfz6r)

---

## 1. Tóm tắt

- Notebook chạy hết cả 9 ô và log được run đầu tiên vào project `randumb-reproduction-24TNT`.
- **Lần chạy đầu bị lỗi** `FileNotFoundError: 'nvidia-smi'` vì quên bật GPU. Bật **Accelerator → GPU T4 x2** rồi chạy lại thì hết lỗi. Ô 2 đã được sửa để in hướng dẫn thay vì báo lỗi, nhưng sửa đổi này chưa được commit.
- **Phát hiện quan trọng nhất:** trên T4, tính toán `float64` chậm hơn `float32` khoảng **12-14 lần**. Đây là đầu vào chính để chốt kiểu số ở Sprint 2 (mục 3).
- **Rủi ro mới:** đọc code phần lõi hiện tại thì thấy ở D = 25.000 với `float64`, bộ nhớ đỉnh có thể vượt 16 GB VRAM của T4 (mục 3.3). Rủi ro này chưa được đo thật.

Ảnh chụp chỉ cho thấy output của ô 3, 6, 7 và 9. Output của ô 2, 4, 5 không có trong ảnh. Ô 5 có lệnh `assert` kiểm tra GPU, và notebook chạy được tới ô 9, nên GPU đã được nhận.

---

## 2. Môi trường Kaggle

| Thành phần | Phiên bản trên Kaggle | Laptop của Kiên | Laptop của Đăng (theo `dang.md`) |
|---|---|---|---|
| Python | 3.13 | 3.12.6 | 3.13.0 |
| torch | 2.11.0+cu130 (bản build cho CUDA 13.0) | 2.5.1+cpu | 2.13.0 (CPU) |
| numpy | 2.1.3 | | 2.3.5 |
| scipy | 1.16.3 | | |
| scikit-learn | 1.6.1 | | 1.8.0 |
| wandb | 0.28.1 | chưa cài | chưa cài |
| datasets | 4.8.5 | chưa cài | |
| timm | 1.0.30 | | |
| psutil | 5.9.5 | | |
| pytest | 8.4.2 | | 9.2.0 |

Các phiên bản trên Kaggle đã được ghim vào `requirements.txt`. Riêng `pillow` chưa ghim, vì notebook chưa in phiên bản của nó.

**Lưu ý:** phiên bản trên laptop của các thành viên khác nhau và khác Kaggle. Run chính thức nên chạy trên Kaggle để kết quả lặp lại được, còn laptop chỉ dùng để viết code và chạy test.

---

## 3. Tốc độ `float32` và `float64` trên T4

### 3.1. Số đo (ô 6, ma trận 4096 × 4096)

| Phép toán | `float32` | `float64` | `float64` chậm hơn |
|---|---|---|---|
| Nhân ma trận | 3,408 TFLOPS | 0,247 TFLOPS | khoảng 13,8 lần |
| Cholesky | 0,0094 giây | 0,1112 giây | khoảng 11,8 lần |

### 3.2. Ước lượng thời gian ở D = 25.000 (lý thuyết, chưa đo)

Tính từ tốc độ đo ở trên. Đây là **cận dưới**, chưa tính thời gian tính RFF, đọc dữ liệu và chi phí phụ.

| Bước | Số mẫu thực tế | `float64` | `float32` |
|---|---|---|---|
| Tích lũy `A += ΦᵀΦ`, MNIST (không flip) | 60.000 | khoảng 5 phút | khoảng 22 giây |
| Tích lũy `A`, CIFAR-10/100 hoặc miniImageNet (có flip) | 100.000 | khoảng 8,4 phút | khoảng 37 giây |
| Tích lũy `A`, TinyImageNet (có flip) | 200.000 | khoảng 17 phút | khoảng 73 giây |
| Cholesky D × D | | khoảng 25 giây | khoảng 2 giây |

Công thức: tích lũy `A` tốn khoảng `2·N·D²` phép tính, Cholesky tăng theo `D³` từ số đo ở 4096.

Với 5 dataset × 3 seed, riêng bước tích lũy của E1 mất **ít nhất khoảng 2,4 giờ GPU** ở `float64`, và khoảng 10 phút ở `float32`. Như vậy chạy được trong quota Kaggle, nhưng phải đo thật ở Sprint 2 để xác nhận.

### 3.3. Rủi ro bộ nhớ ở D = 25.000 (đọc code, chưa đo)

Mỗi ma trận D × D ở D = 25.000 chiếm khoảng **5 GB** với `float64`. Code phần lõi hiện tại tạo nhiều ma trận tạm cùng kích thước:

| Hàm | Ma trận D × D cùng tồn tại (ước lượng) |
|---|---|
| `SufficientStats.finalize` | `A`, `correction`, `A - correction`, `cov`: khoảng 4 ma trận |
| `oas_shrink` | `cov`, `cov ** 2`, `torch.eye`, `(1 - rho) * cov`, `rho * mu * eye`, kết quả: khoảng 4-6 ma trận |
| `fit_lda` | `cov`, `torch.eye`, `lam * eye`, `cov + lam * eye`, `L`: khoảng 4-5 ma trận |

Ở `float64`, 4 ma trận đã khoảng 20 GB, vượt 16 GB của một GPU T4. Ở D = 15.000 (khoảng 1,8 GB mỗi ma trận) thì vẫn vừa.

**Đề xuất cho Đăng ở Sprint 2**, trước khi chạy D = 25.000:
- Thay `torch.eye` và phép cộng bằng cách cộng trực tiếp vào đường chéo: `cov.diagonal().add_(lam)`.
- Co OAS tại chỗ: `cov.mul_(1 - rho)` rồi `cov.diagonal().add_(rho * mu)`.
- Trong `finalize`, tính phần hiệu chỉnh một lần bằng `(means.T * dem) @ means` thay cho vòng lặp, rồi trừ tại chỗ.
- Giải phóng `A` sau khi `finalize` nếu không cần lưu.

Nếu không kịp sửa, phương án dự phòng trong kế hoạch là dùng D = 15.000 hoặc `float32`, và ghi rõ là sai lệch.

---

## 4. Kiểm tra dataset (ô 7)

| Dataset | Kết quả | Split | Cột |
|---|---|---|---|
| MNIST | ok | `train` 60.000, `test` 10.000 | `image`, `label` |
| CIFAR-10 | ok | `train` 50.000, `test` 10.000 | `img`, `label` |
| CIFAR-100 | ok | `train` 50.000, `test` 10.000 | `img`, `fine_label`, `coarse_label` |
| TinyImageNet | ok | `train` 100.000, `valid` 10.000 | `image`, `label` |
| miniImageNet | ok | `train` 50.000, `validation` 10.000, `test` 5.000 | `image`, `label` |

Kết quả **khớp hoàn toàn** với `DATASETS` trên nhánh `feat/kien/data-pipeline`, nên không cần sửa `data.py`. Notebook gọi bảng cũ trên `main` có tên `Maysee/tiny-imagenet`, nhưng Hugging Face tự chuyển hướng sang `zh-plus/tiny-imagenet`.

**Cảnh báo phụ:** Hugging Face báo *"You are sending unauthenticated requests"*. Không ảnh hưởng tới kết quả, nhưng khi tải miniImageNet (7,5 GB) có thể bị giới hạn tốc độ. Nếu bị chậm, tạo token đọc trên Hugging Face và thêm vào Kaggle Secrets với tên `HF_TOKEN`.

---

## 5. Đã điền vào repo

| File | Nội dung |
|---|---|
| `requirements.txt` | Ghim phiên bản theo Kaggle |
| `README.md` | Thêm dòng run `6vddfz6r` vào bảng kết quả; cập nhật trạng thái `src/randumb/` |
| `Docs/implementation_plan.md` | Đánh dấu xong 4 việc của Kiên ở Sprint 0 |
| `Docs/report1.md` | Đánh dấu các bước đã làm |
| `Docs/e2_datasets.md` | Ghi nhận Kaggle xác nhận lại 5 dataset |

---

## 6. Tiến độ so với kế hoạch (tính đến 11/11)

Theo kế hoạch, 11/11 là ngày cuối Sprint 2, nhưng nhóm đang trễ khoảng một sprint.

| Mốc | Kế hoạch | Thực tế |
|---|---|---|
| 08/11: 6 test qua, có run đầu tiên | Trên `main` | Code của cả ba người đã xong nhưng **vẫn nằm trên ba nhánh riêng**, chưa merge |
| 09/11: đo giờ, chốt D và dataset E2 | | Mới có số đo nhanh ở mục 3; dataset E2 đang chờ hỏi trợ giảng |
| 11/11: chốt cấu hình (E8 pha A, E9 nhỏ) | | Chưa bắt đầu |

**Đề xuất thứ tự làm từ hôm nay:**
1. **Merge vào `main`** theo thứ tự: phần lõi của Đăng, rồi `feat/kien/data-pipeline`, rồi nhánh `verification/` của Huy. Mỗi nhánh tạo một Pull Request theo `branch_git.md`.
2. **Đo giờ thật trên Kaggle** bằng `scripts/run_experiment.py`: CIFAR-10, D = 1.000, 2.000, 5.000, cả `float32` và `float64`, với `--group timing`.
3. **Đăng sửa bộ nhớ** theo mục 3.3, rồi chốt D và kiểu số.
4. **Chạy E8 pha A** để chốt γ và chuẩn hóa, song song với **E9 nhỏ** của Huy.
5. Nếu tới 13/11 vẫn chưa chốt được cấu hình, cân nhắc lùi mốc khóa kết quả từ 15/11 sang 16/11, và áp dụng thứ tự cắt giảm trong kế hoạch.

---

## 7. Việc còn mở

- [ ] Mở link run `6vddfz6r` bằng cửa sổ ẩn danh để kiểm tra người ngoài xem được.
- [ ] Bấm **Save Version** trên Kaggle. Notebook hiện vẫn là bản nháp (Version 0), nên output có thể mất khi phiên kết thúc.
- [ ] Mời Đăng và Huy vào team W&B.
- [ ] Hoàn tất đơn Academic trên W&B trước khi hết dùng thử (khoảng 05/12).
- [ ] Xác nhận trên Moodle: form đã được duyệt chưa, giờ nộp, ngày poster workshop.
- [ ] Hỏi trợ giảng về nguồn dữ liệu E2 (xem `e2_datasets.md`, mục 3).
- [ ] (Không bắt buộc) Thêm `HF_TOKEN` vào Kaggle Secrets.
