# Nguồn dữ liệu cho E1 và E2

**Người lập:** Kiên
**Ngày tra cứu:** 08/11/2026
**Cách kiểm tra:** gọi API `datasets-server.huggingface.co/info` và `huggingface.co/api/datasets` để lấy tên cột, số mẫu, giấy phép; đọc README của [RevisitingCIL](https://github.com/zhoudw-zdw/RevisitingCIL). Chưa tải ảnh của dataset E2 nào.

---

## 1. E1: đã kiểm chứng, khác với Instruction.md ở 3 điểm

Năm dataset đều đã có trong `src/randumb/data.py`. MNIST và CIFAR-10 đã chạy thật trên laptop ở D = 1.000. Ngày 11/11, ô 7 của notebook Kaggle (W&B run `6vddfz6r`) xác nhận lại cả 5 dataset có đúng tên cột và số mẫu như bảng dưới, kể cả khi gọi tên cũ `Maysee/tiny-imagenet` (Hugging Face tự chuyển hướng).

| Dataset | Nguồn HF | Cột ảnh / nhãn | Split train / test | Ghi chú |
|---|---|---|---|---|
| MNIST | `ylecun/mnist` | `image` / `label` | `train` 60.000 / `test` 10.000 | Ảnh xám 28×28 |
| CIFAR-10 | `uoft-cs/cifar10` | **`img`** / `label` | `train` 50.000 / `test` 10.000 | |
| CIFAR-100 | `uoft-cs/cifar100` | **`img`** / `fine_label` | `train` 50.000 / `test` 10.000 | |
| TinyImageNet | **`zh-plus/tiny-imagenet`** | `image` / `label` | `train` 100.000 / **`valid`** 10.000 | Ảnh gốc 64×64, thu về 32×32 |
| miniImageNet | `timm/mini-imagenet` | `image` / `label` | `train` 50.000 / **`validation`** 10.000 | Nặng **7,5 GB**; ảnh nhiều kích thước; có thêm split `test` 5.000 ảnh không dùng |

**Khác với Instruction.md:**
1. `Maysee/tiny-imagenet` đã đổi tên thành `zh-plus/tiny-imagenet`.
2. CIFAR-10 và CIFAR-100 dùng cột `img`, không phải `image`.
3. miniImageNet nặng 7,5 GB, nên tải một lần rồi dùng cache `.npz` của `load_images`.

**Kết quả chạy thử trên laptop (CPU, D = 1.000, seed 0, z-score, γ = `scale`, không log W&B):**

| Dataset | λ | Độ chính xác | Mốc tham khảo suy từ bài | Thời gian tổng |
|---|---|---|---|---|
| MNIST | 10⁻⁶ | 94,52% | khoảng 94,7 | khoảng 2 giây |
| CIFAR-10 (có flip) | 10⁻⁵ | 44,35% | khoảng 45,2 | khoảng 4 giây |

Mốc tham khảo lấy số của bài ở D = 25K trừ mức tăng từ 1K lên 25K (Instruction.md, mục 4.5). Đây chỉ là kiểm tra pipeline, chưa phải run chính thức trên W&B.

---

## 2. E2: chưa có nguồn nào khớp đúng cách chia của bài

Bài báo dùng cách chia của RevisitingCIL / LAMDA-PILOT. README của RevisitingCIL cho thấy:
- Các tập con đã chia được phát hành qua **Google Drive / OneDrive**, không có trên Hugging Face hay Kaggle.
- Tác giả ghi: *"I do not have the right to distribute these datasets"*, nên nhóm **không được upload lại** các file này lên HF hay Kaggle.
- Không bản nào trên HF giữ tên file gốc, nên không dựng lại được đúng cách chia từ bản HF.

| Dataset | Bài dùng ở | Nguồn của RevisitingCIL | Bản trên HF tốt nhất tìm được | Cột / split trên HF | Giấy phép trên HF | Tình trạng |
|---|---|---|---|---|---|---|
| CIFAR-100 | Bảng 1, 7 | Tự tải bản chuẩn | `uoft-cs/cifar100` | `img`, `fine_label`; 50.000 / 10.000 | không ghi | **Khớp**: RevisitingCIL dùng bản chuẩn |
| CUB-200 | Bảng 1, 7 | Google Drive | `Donghyun99/CUB-200-2011` | `image`, `label` (200 lớp); 5.994 / 5.794 | không ghi | **Tương đương**: split chuẩn, chưa biết có giống RevisitingCIL không |
| ImageNet-R | Bảng 1, 7 | Google Drive | `axiong/imagenet-r` | `image`, `wnid`, `class_name`; chỉ 1 split 30.000 | MIT | **Tương đương**: nhóm phải tự chia |
| ImageNet-A | Bảng 1, 7 | Google Drive | `barkermrl/imagenet-a` | `image`, `label` (200 lớp); chỉ 1 split 7.500 | MIT | **Tương đương**: nhóm phải tự chia |
| VTAB | Bảng 1, 7 | Google Drive | Chỉ có từng tác vụ riêng lẻ (`clip-benchmark/wds_vtab-*`) | | | **Chưa có**: chưa xác minh tập con của RevisitingCIL gồm những tác vụ nào |
| OmniBenchmark | Bảng 1 (chỉ IN-21K) | Google Drive | `ZhangYuanhan/OmniBenchmark` | **trống** | CC BY-NC-ND 4.0 | **Chưa có**: giấy phép ND không cho phân phối bản đã chỉnh sửa |
| Stanford Cars | Bảng 1 (chỉ IN-21K) | Không có trong README của RevisitingCIL | `tanganke/stanford_cars` | `image`, `label` (196 lớp); 8.144 / 8.041 | không ghi | **Tương đương**: split chuẩn |

Chưa xác minh được: trang GitHub của LAMDA-PILOT (bị lỗi 504 khi tra cứu), số mẫu train/test trong các tập con của RevisitingCIL, và giấy phép gốc của CUB, Cars, ImageNet-R, ImageNet-A.

---

## 3. Đề xuất cho E2

1. **Chạy ngay** CIFAR-100, CUB, ImageNet-R, ImageNet-A từ HF, với cả hai backbone.
   - ImageNet-R và ImageNet-A: tự chia 80/20 theo từng lớp với seed cố định. Lưu danh sách chỉ số vào repo để ai cũng tái tạo được.
   - Ghi cả 4 dataset là "dataset tương đương" trong phân tích sai lệch, trừ CIFAR-100.
2. **Hỏi trợ giảng** trước ngày 09/11: có chấp nhận tải file của RevisitingCIL từ Google Drive bằng code (ví dụ `gdown`) trên Kaggle không, khi nhóm không upload lại. Nếu được, dùng cách này cho VTAB và có thể cho cả 4 dataset trên, để khớp đúng cách chia của bài.
3. **Nếu trợ giảng không đồng ý:** bỏ VTAB và OmniBenchmark, ghi rõ lý do. Thứ tự cắt giảm trong kế hoạch vốn đã bỏ VTAB đầu tiên.
4. **Cars** chỉ làm khi còn thời gian, vì bài chỉ báo cáo cho backbone IN-21K.
5. **Không upload** các file của RevisitingCIL lên HF hay Kaggle, vì tác giả không có quyền phân phối, và OmniBenchmark có giấy phép ND.
