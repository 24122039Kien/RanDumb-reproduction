# Báo cáo 14/11: merge code, kiểm tra cấu hình E1, cài đặt E2

**Người lập:** Kiên
**Ngày:** 14/11/2026
**Trạng thái `main`:** commit `426ae8a`, 54 test qua

---

## 1. Tóm tắt

- **Toàn bộ code đã nằm trên `main`.** Cả 6 nhánh trong repo đã được merge.
- **Cấu hình E1 mặc định hiện tại cho kết quả ngang đoán ngẫu nhiên.** γ = 1,0 kèm z-score chỉ đạt khoảng 10% trên CIFAR-10. Phải chốt lại chuẩn hóa hoặc γ **trước khi chạy E1** (mục 3).
- **Lịch được lập lại** từ 14/11 đến hạn deck: khóa kết quả tối 17/11, nộp 20/11, bỏ E3-E7 (mục 4).
- **Code E2 đã xong** và đã kiểm tra trên dữ liệu và trọng số ViT thật. Còn chờ chạy trên Kaggle (mục 5 và 6).

---

## 2. Merge các nhánh

| Nhánh | Người làm | Merge vào `main` |
|---|---|---|
| `dang/sprint-0-1` | Đăng | Đăng merge ngày 13/11 |
| `feat/kien/data-pipeline` | Kiên | Đăng merge ngày 13/11 |
| `feat/huy-sprint1` | Huy | Đăng merge ngày 13/11 |
| `dang/sprint-2-config` | Đăng | Merge 14/11 |
| `docs/kien/replan-14-11` | Kiên | Merge 14/11 |
| `feat/kien/e2-vit-features` | Kiên | Merge 14/11 |

- Không có xung đột. Sau mỗi lần merge, toàn bộ test đều qua: 37 test sau khi merge nhánh của Đăng, 54 test sau khi merge nhánh E2.
- Một lần chạy test đầu tiên trên laptop bị Python dừng đột ngột; chạy lại 3 lần đều qua, nên đó là lỗi thoáng qua của máy, không phải lỗi code.
- Đã kiểm tra: không có API key bị ghi cứng, không có file key hay code của tác giả bị commit, không có dòng ghi công Claude trong commit.

---

## 3. Phát hiện: γ = 1,0 chỉ dùng được với chuẩn hóa L2

Nhánh `dang/sprint-2-config` đặt mặc định E1 là γ = 1,0 (đúng giá trị bài báo) và giữ chuẩn hóa z-score. Thử nhanh trên CIFAR-10:

| γ | Chuẩn hóa | Độ chính xác |
|---|---|---|
| **1,0** | **z-score (mặc định hiện tại của E1)** | **9,82%** |
| 1,0 | chỉ chia 255 | 10,01% |
| 1,0 | L2 theo từng ảnh | 45,51% |
| `scale` | z-score | 45,30% |
| `scale` | chỉ chia 255 | 44,96% |
| `scale` | L2 theo từng ảnh | 43,86% |

**Điều kiện thử:** laptop, CPU, D = 1.000, seed 0, λ = 10⁻⁵, có flip, **không log W&B**. Mốc tham khảo suy từ bài ở D = 1.000 là khoảng 45,2%. Các số này **chưa dùng được trong deck**; phải chạy lại trên Kaggle như một phần của E8 pha A.

**Giải thích:** với z-score, mỗi ảnh có 3.072 chiều, mỗi chiều có phương sai khoảng 1, nên `ωᵀx` dao động khoảng √(2·3.072) ≈ 78 radian và `cos(ωᵀx)` gần như nhiễu. Chuẩn hóa L2 đưa khoảng cách giữa hai ảnh về cỡ 1, nên γ = 1,0 trở nên hợp lý.

**Khớp với kiểm chứng của Huy:** theo `verification/README.md`, code gốc dùng `gamma='scale'` cùng z-score theo kênh, không dùng γ = 1,0.

**Quyết định cần chốt (Đăng chủ trì, cả nhóm đồng ý):**

| Phương án | Ưu điểm | Nhược điểm |
|---|---|---|
| A. z-score + γ = `scale` | Giống code gốc mà Huy đã kiểm tra | Khác giá trị γ ghi trong bài |
| B. L2 + γ = 1,0 | Giữ đúng γ của bài | Bài không nói dùng L2; khác code gốc |

Đề xuất của Kiên: **phương án A** làm cấu hình chính của E1, đưa phương án B vào E8 làm phân tích sai lệch. Dù chọn phương án nào, phải đổi mặc định trong `scripts/run_experiment.py` hoặc truyền rõ `--norm` và `--gamma` khi chạy.

---

## 4. Lịch mới từ 14/11

Chi tiết ở [implementation_plan.md](implementation_plan.md), mục 0.

| Ngày | Đăng | Kiên | Huy |
|---|---|---|---|
| T7 14/11 | Chốt chuẩn hóa và γ; sửa bộ nhớ ở D = 25.000 | Code E2 (xong) | E9 nhỏ; phần đọc bài của deck |
| CN 15/11 | E8 pha A; đo giờ và bộ nhớ ở D = 25.000; E1 với MNIST, CIFAR-10 | Chạy E2 trên Kaggle | E9 đầy đủ |
| T2 16/11 | E1 với CIFAR-100, TinyImageNet, miniImageNet | Bổ sung E2 nếu còn thiếu | Phần phương pháp và đánh giá của deck |
| T3 17/11 | **Khóa kết quả** | **Khóa kết quả** | Phân tích E9 |
| T4 18/11 | Phân tích sai lệch, slide E1, E8 | Phân tích sai lệch, README, slide E2 | Ghép deck |
| T5 19/11 | Cả nhóm rà soát chéo | | |
| T6 20/11 | **Nộp deck và link repo** | | |

---

## 5. Code E2 đã cài

| File | Nội dung |
|---|---|
| `src/randumb/vit_features.py` | Tải 4 dataset từ HF, tự chia 80/20 theo lớp cho ImageNet-R và ImageNet-A (`SPLIT_SEED = 0`), tạo ViT-B/16, trích đặc trưng theo lô |
| `src/randumb/baselines.py` | NCM: Euclid, cosine, tích vô hướng |
| `src/randumb/pipeline.py` | Thêm chế độ `identity`: chạy RanDumb thẳng trên đặc trưng có sẵn, không RFF |
| `src/randumb/data.py` | Tách hàm `split_indices` dùng chung |
| `scripts/extract_vit_features.py` | Trích đặc trưng, lưu `data/features/<dataset>-<backbone>-<norm\|raw>.pt` |
| `scripts/run_e2.py` | RanDumb hoặc NCM, đối chiếu số của bài, log W&B |
| `notebooks/02_e2_vit.ipynb` | Chạy cả E2 trên Kaggle |

**Giao thức** (chi tiết và lý do ở [e2_datasets.md](e2_datasets.md), mục 4):

| Thành phần | Lựa chọn |
|---|---|
| Dataset | CIFAR-100, CUB, ImageNet-R, ImageNet-A; bỏ VTAB, OmniBenchmark, Cars |
| Backbone | IN-1K: `vit_base_patch16_224.augreg_in21k_ft_in1k`; IN-21K: `vit_base_patch16_224.augreg_in21k` |
| Tiền xử lý | Thu về 224×224 bicubic; mặc định chuẩn hóa mean/std của timm |
| RanDumb | LDA + OAS, λ = 0, không RFF |
| NCM | Khoảng cách Euclid |
| Số run | 4 dataset × 2 backbone × 2 phương pháp = 16 run, mỗi run 1 seed |

**Đã kiểm tra:**
- 54 test qua, gồm test cho dataset giả, mô hình giả và đường đi đầu cuối của `run_e2.py`.
- Trên laptop: tải CIFAR-100 thật (50.000 / 10.000, nhãn 0-99), tải trọng số `augreg_in21k` thật, đặc trưng 768 chiều hợp lệ.
- Trên tập con chỉ có 1 ảnh mỗi lớp: NCM Euclid 43,7%, NCM cosine 45,3%, RanDumb 43,7% trên 100 lớp (đoán ngẫu nhiên: 1%). Chỉ để kiểm tra code, không đối chiếu được với bài.

**Chưa kiểm tra:**
- Chưa chạy trên Kaggle; thời gian ước lượng 30-60 phút chưa được đo.
- Chưa tải thật CUB, ImageNet-R, ImageNet-A; mới chỉ kiểm tra tên cột và số mẫu qua API của Hugging Face.
- Trên laptop, trích đặc trưng chỉ đạt khoảng 8,5 ảnh/giây, nên không chạy E2 đầy đủ ở đây được.

---

## 6. Cách chạy E2 trên Kaggle

1. Tạo notebook mới, **File → Import Notebook**, chọn `notebooks/02_e2_vit.ipynb`.
2. Bật **GPU T4 x2**, **Internet On**, Secret `WANDB_API_KEY`. Nên thêm Secret `HF_TOKEN` để tải dataset nhanh hơn.
3. Giữ `REPO_BRANCH = "main"` ở ô 1, vì code E2 đã có trên `main`.
4. Chọn **Run All**. Nếu phiên bị ngắt, chạy lại ô 5: file đặc trưng đã có sẽ được bỏ qua.
5. Bấm **Save Version**, rồi gửi bảng kết quả ở ô 6 để cập nhật README và deck.

---

## 7. Việc còn mở

| Việc | Người | Hạn |
|---|---|---|
| Chốt chuẩn hóa và γ cho E1 (mục 3) | Đăng, cả nhóm đồng ý | 14/11 |
| Sửa bộ nhớ ở D = 25.000 hoặc chốt phương án dự phòng | Đăng | 15/11 |
| Chạy E2 trên Kaggle | Kiên | 15/11 |
| E9 và phần đọc bài của deck | Huy | 15/11 |
| Mời Đăng và Huy vào team W&B; hoàn tất đơn Academic | Kiên | Sớm nhất có thể |
| Xác nhận trên Moodle: giờ nộp, form đã duyệt | Cả nhóm | Sớm nhất có thể |
| Mở link run `6vddfz6r` ở cửa sổ ẩn danh; Save Version notebook smoke | Kiên | Sớm nhất có thể |
