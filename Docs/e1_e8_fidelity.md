# Đối chiếu E1 và E8 với bài báo và Instruction

**Người lập:** Kiên
**Ngày:** 19/11/2026
**Phạm vi:** những gì `notebooks/03_dang_sprint3.ipynb` thực sự chạy, qua `scripts/run_sprint3.py` và `scripts/run_experiment.py`.

**Nguồn đối chiếu:**
- Bài báo [Randumbofficial.pdf](../Randumbofficial.pdf): Mục 2, Mục 3 phần *Implementation Details* (trang 5-6), Bảng 2-6.
- [Instruction.md](Instruction.md): mục 2 (phương pháp), 4.3 (E1) và 4.6 (E8).
- [verification/README.md](../verification/README.md): phát hiện của Huy khi đọc code gốc.

---

## 1. Tóm tắt

- **E1 khớp mọi thông số bài báo nêu rõ.** Có 3 điểm bài không nói, nên nhóm phải tự chọn. Không cấu hình nào khớp "100%" được, vì chính bài báo và code gốc mâu thuẫn nhau ở γ (mục 3).
- **E8 không có trong bài báo.** Đây là phân tích độ nhạy do môn yêu cầu (Instruction, mục 4.6). Trước ngày 19/11, notebook chỉ chạy **nửa sau** (pha B: shrink × λ). **Đã bổ sung pha A** (chuẩn hóa × γ), nên giờ E8 phủ đủ 4 yếu tố của Instruction.
- **Sửa lỗi bộ nhớ ở bước giải hệ** (ngày 19/11). Lỗi này làm D = 25.000 hết VRAM trên T4; sau khi sửa vẫn giữ nguyên D = 25.000 và float64.

---

## 2. E1: từng thông số

| Thông số | Bài báo | Notebook chạy | Khớp |
|---|---|---|---|
| Dataset | MNIST, CIFAR-10, CIFAR-100, TinyImageNet200, miniImageNet100 | Đủ 5 | Có |
| Kích thước ảnh | Tiny/mini thu về 32×32; vector 784 (MNIST) và 3.072 chiều | Như bài | Có |
| RFF | "Scikit-Learn implementation of RFF", 25K chiều | `√(2/D)·cos(ωᵀx + b)`, `ω ~ N(0, 2γI)`, `b ~ U(0, 2π)`: đúng công thức `RBFSampler`, nhóm tự cài | Có |
| D | 25.000 | 25.000 | Có |
| γ | 1,0 | 1,0 | Có |
| λ | 10⁻⁶ MNIST; 10⁻⁵ CIFAR; 10⁻⁴ Tiny/mini | Như bài | Có |
| Flip | Mỗi ảnh kèm bản lật, trừ MNIST | Lật ngang (theo chiều rộng), cả hai đều là mẫu; MNIST không lật | Có |
| Shrink | OAS cho ma trận hiệp phương sai | OAS trên hiệp phương sai trong lớp gộp, `n` = số mẫu kể cả ảnh lật | Có |
| Giải hệ | "inverse obtained by least squares (S + λI)" | Cholesky của `S_oas + λI`, lỗi thì `lstsq`; cùng nghiệm khi ma trận xác định dương | Có (Instruction 2.4 yêu cầu Cholesky) |
| Dự đoán | Khoảng cách Mahalanobis tới trung bình lớp | Như bài; bỏ số hạng chung `φᵀS⁻¹φ` | Có (test 5) |
| Một lượt, không lưu mẫu | Có | Tích lũy theo lô, tương đương cập nhật từng mẫu | Có (test 2, 3) |
| Đánh giá | Tập test của mọi lớp sau khi học hết | Toàn bộ tập test, không giới hạn số mẫu | Có |
| Số lần chạy | 3 lần khởi tạo kernel (Checklist câu 7) | Seed 0, 1, 2 điều khiển `ω`, `b` | Có; bài không cho giá trị seed |
| Kiểu số | Không nêu; scikit-learn mặc định float64 | float64 cho RFF, thống kê, giải hệ | Có |
| Số để so | Bảng 2-4: 98,3 / 55,6 / 28,6 / 11,6 / 17,7 | Như bài; Tiny dùng 11,6 (ứng λ = 10⁻⁴) | Có |

### Các điểm bài không nói, nhóm phải tự chọn

| Điểm | Bài nói | Nhóm chọn | Ghi vào deck |
|---|---|---|---|
| Chuẩn hóa | Chỉ "normalize all images" | **L2 theo từng ảnh** (Đăng chốt ngày 14/11) | Là lựa chọn của nhóm, không phải của bài |
| Nguồn miniImageNet | Không nêu cách chia | `timm/mini-imagenet`, test là split `validation` (10.000 ảnh) | Dataset tương đương, có thể lệch số |
| Cách thu nhỏ Tiny/mini | Không nêu | PIL bilinear | Ảnh hưởng nhỏ |
| Dạng RFF | Mục 2 viết dạng cos + sin, 2D chiều | Dạng D chiều của scikit-learn, khớp "25K" | Mâu thuẫn nội tại của bài |

---

## 3. Vì sao không thể khớp "100%"

Bài ghi γ = 1,0, nhưng code gốc dùng `gamma='scale'` và chuẩn hóa z-score theo kênh (Huy kiểm tra trong `verification/README.md`). Hai nguồn này không thể cùng đúng:

| Cấu hình | Giữ đúng | Kết quả thử nhanh (CIFAR-10, D = 1.000, report3) |
|---|---|---|
| z-score + γ = 1,0 | γ của bài, chuẩn hóa của code gốc | **9,82%**, bằng đoán ngẫu nhiên |
| z-score + γ = `scale` | Code gốc | 45,30% |
| **L2 + γ = 1,0 (E1 hiện tại)** | **Mọi con số bài công bố** | 45,51% |

Lý do: với z-score, `ωᵀx` dao động khoảng √(2·3.072) ≈ 78 radian, nên `cos(ωᵀx + b)` gần như nhiễu. Chuẩn hóa L2 đưa khoảng cách giữa các ảnh về cỡ 1, khi đó γ = 1,0 mới có nghĩa.

**Kết luận:** E1 giữ đúng mọi thông số bài **viết ra**, và chỉ tự chọn ở chỗ bài **bỏ trống**. Cấu hình của code gốc (z-score + `scale`) được chạy trong E8 pha A, ở cả D = 25.000, để deck so được hai cách.

---

## 4. E8: đối chiếu với Instruction mục 4.6

Instruction yêu cầu chạy CIFAR-10 và CIFAR-100, với D = 5.000 và một D lớn, thay đổi từng yếu tố sau:

| Yếu tố | Instruction | Trước 19/11 | Sau 19/11 |
|---|---|---|---|
| Chuẩn hóa | z-score; L2; chỉ chia 255 | **Thiếu** | E8-A: đủ 3 cách |
| γ | 1,0; `scale`; vài bội số quanh `scale` | **Thiếu** | E8-A: 1,0; 0,5·`scale`; `scale`; 2·`scale` |
| Shrink | OAS gộp; OAS từng lớp; không OAS | E8-B: đủ | Giữ nguyên |
| λ | Cố định; tỉ lệ `trace(S)/D` | E8-B: đủ | Giữ nguyên |

### E8 pha A: 32 run, seed 0

Các thông số khác giữ như E1 cho CIFAR: OAS gộp, λ = 10⁻⁵ cố định, flip, float64.

| D | Chuẩn hóa | γ | Số run |
|---|---|---|---|
| 5.000 | z-score, L2, chỉ chia 255 | 1,0; 0,5·`scale`; `scale`; 2·`scale` | 11 mỗi dataset |
| 25.000 | z-score, L2, chỉ chia 255 | 1,0; `scale` | 5 mỗi dataset |

- Bỏ tổ hợp L2 + γ = 1,0 vì E8-B (`oas`, `fixed`) đã chạy đúng tổ hợp đó ở cả hai D.
- Ở D = 25.000 chỉ giữ hai giá trị γ để vừa giờ GPU; lưới đầy đủ đã có ở D = 5.000.
- `scale` = `1/(d·Var(X))`, tính trên tập train sau khi chuẩn hóa, giống `gamma='scale'` của scikit-learn.

### E8 pha B: 24 run, seed 0 (thiết kế của Đăng, không đổi)

CIFAR-10/100 × D ∈ {5.000, 25.000} × shrink ∈ {OAS gộp, OAS từng lớp, không OAS} × λ ∈ {cố định, tương đối}. Chi tiết ở [sprint3_Dang.md](sprint3_Dang.md).

---

## 5. Cách chạy trên Kaggle

Mỗi thành viên chạy một phiên trên tài khoản Kaggle của mình, cùng lúc. Chỉ sửa ô 1 của notebook 03, giữ `DATASET = None`, `SEED = None`, `DIM = None`:

| Người | Notebook | GROUP | Số run | Ước lượng GPU |
|---|---|---|---|---|
| Đăng | 03 | `'E1'` | 15 (5 dataset × 3 seed, D = 25.000) | 2,5-3 giờ, cộng thời gian tải Tiny/mini |
| Huy | 03 | `'E8-B'` | 24 (shrink × λ) | Khoảng 2 giờ |
| Kiên | 03 | `'E8-A'` | 32 (chuẩn hóa × γ) | Khoảng 2 giờ |
| Kiên | 04 | | 40 (E2 bổ sung) | Khoảng 10 phút, cần Input là output notebook E2 của Kiên |

- Chia theo nhóm, không chia theo D: λ tương đối của E8-B lấy hệ số từ run E8-B D = 5.000 `fixed` **trong cùng thư mục kết quả**. Chạy cả E8-B trong một phiên thì script tự xếp thứ tự đúng.
- E8-A không phụ thuộc run nào khác, nên chạy riêng được.
- Notebook 04 chỉ Kiên chạy được, vì file đặc trưng ViT nằm trong output notebook E2 của Kiên.
- Ô 5 (kiểm tra bộ nhớ ở D = 25.000) chạy trước khi tải dữ liệu. Nếu lỗi ở đó thì dừng và báo nhóm, không tự giảm D hoặc đổi dtype.
- Chạy xong: **Save Version**, tải `summary.csv`, `e1_aggregate.csv` (phiên E1) và các file JSON trong `/kaggle/working/sprint3-results` rồi gửi Kiên.
- Thời gian là ước lượng từ tốc độ float64 đo trên T4 (report2), chưa đo thật.
