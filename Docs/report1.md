# Việc tiếp theo sau khi tạo W&B project public

**Người làm:** Kiên
**Ngày:** 06/11/2026
**Bối cảnh:** W&B project [`randumb-reproduction-24TNT`](https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT) đã ở chế độ Public. Sprint 0 đang trễ 1 ngày: còn kiểm tra project, mời thành viên và chạy notebook Kaggle.

Làm theo thứ tự. Mỗi bước xong thì đánh dấu trong [implementation_plan.md](implementation_plan.md).

---

## 1. Kiểm tra project W&B (5 phút)

- [x] **Mở bằng cửa sổ ẩn danh:** đã xem được khi chưa đăng nhập (09/11).
- [ ] **Mời Đăng và Huy vào team:** vào **Team → Settings → Members**, mời bằng email.
- [x] **Tên entity:** `24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm`, project `randumb-reproduction-24TNT`. Đã điền sẵn trong `src/randumb/tracking.py` và ô 1 của notebook.
- [ ] **Hoàn tất đơn Academic:** bấm **Complete application** ở góc trái dưới. Bản dùng thử hết hạn khoảng 05/12, trùng hạn nộp poster.

## 2. Chuẩn bị Kaggle (10 phút)

- [x] **Lấy API key W&B:** trong cửa sổ đã đăng nhập, mở trang project và bấm **Generate** ở mục "Generate your API key", hoặc vào [wandb.ai/authorize](https://wandb.ai/authorize).
- [x] **Xác minh số điện thoại trên Kaggle:** cần có bước này mới bật được Internet.
- [x] **Import notebook:** vào **Create → New Notebook → File → Import Notebook**, tải lên `notebooks/00_kaggle_smoke.ipynb` (có trong repo trên GitHub).
- [x] **Chỉnh Settings:** Accelerator chọn **GPU T4 x2**, Internet chọn **On**. Lần chạy đầu quên bật GPU nên ô 2 báo lỗi `nvidia-smi`.
- [x] **Thêm Secret:** vào **Add-ons → Secrets**, thêm `WANDB_API_KEY` và bật cho notebook này. Không dán key vào code.

## 3. Chạy notebook (khoảng 10-15 phút, chưa đo thật)

- [x] **Kiểm tra ô 1:** `WANDB_ENTITY` và `WANDB_PROJECT` đã điền sẵn, không cần sửa.
- [x] **Chạy toàn bộ:** chọn **Run All** (11/11). Kết quả xem [report2.md](report2.md).
- [ ] **Kiểm tra run:** mở link run ở ô 9 bằng cửa sổ ẩn danh.

Notebook này chưa từng chạy trên Kaggle. Nếu ô nào báo lỗi, copy thông báo lỗi gửi Claude.

## 4. Gửi kết quả cho Claude

Gửi 3 thứ sau để Claude cập nhật repo và push:

| Từ đâu | Dùng để làm gì |
|---|---|
| Danh sách `pkg==version` ở **ô 3** | Ghim phiên bản trong `requirements.txt` |
| Bảng dataset ở **ô 7** | Sửa `DATASETS` trong `src/randumb/data.py` cho đúng tên cột và số mẫu |
| Link project W&B | Điền vào dòng "W&B project" trong README |

## 5. Sau đó: việc Sprint 1 của Kiên

Phụ trách dữ liệu và hạ tầng chạy:

1. [x] Viết `data.py` để tải 5 dataset, có `assert` kiểm tra số mẫu.
2. [x] Viết `tracking.py`.
3. [x] Lập bảng dữ liệu cho E2: [e2_datasets.md](e2_datasets.md).
4. [ ] Ngày 08/11: viết `run_experiment.py` và ghép pipeline với Đăng. Script đã viết trên nhánh `feat/kien/data-pipeline`; còn chờ merge phần lõi của Đăng vào `main`.

Bảng ở ô 7 là đầu vào trực tiếp cho `data.py`, nên chạy xong notebook là bắt tay vào được ngay.

## 6. Nhắc nhóm

- [ ] **Xác nhận trên Moodle:** form đã được duyệt chưa, giờ nộp chính xác của deck và poster, ngày poster workshop.
- [ ] **Nhắc Đăng:** clone repo và commit lần đầu bằng tài khoản của mình.
