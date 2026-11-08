# Thư mục dữ liệu

Mọi dữ liệu của project được lưu ở đây khi chạy code. **Chỉ file README này được commit**; phần còn lại nằm trong `.gitignore`, vì:
- GitHub từ chối file lớn hơn 100 MB (miniImageNet nặng khoảng 7,5 GB).
- Môn yêu cầu dữ liệu được tải bằng code từ Hugging Face hoặc Kaggle.
- Giấy phép của một số dataset không cho phân phối lại.

## Cấu trúc

```
data/
  hf/       file gốc tải từ Hugging Face (do thư viện datasets quản lý)
  cache/    ảnh đã giải mã và thu nhỏ, mỗi split một file <dataset>-<split>.npz
```

Hai thư mục con được tạo tự động ở lần chạy đầu, ví dụ:

```bash
python scripts/run_experiment.py --dataset mnist --dim 1000 --lam 1e-6 --group smoke --no-wandb
```

## Dung lượng dự kiến

| Dataset | `hf/` | `cache/` |
|---|---|---|
| MNIST | 20 MB (đã đo) | 53 MB (đã đo) |
| CIFAR-10, CIFAR-100, TinyImageNet | chưa đo | chưa đo |
| miniImageNet | khoảng 7,5 GB trở lên, theo thông tin trên Hugging Face | chưa đo |

## Đổi vị trí

- **Cả thư mục:** đặt biến môi trường `RANDUMB_DATA_DIR`, ví dụ trên Kaggle `RANDUMB_DATA_DIR=/kaggle/working/data`.
- **Từng phần:** dùng `--cache-dir` và `--hf-cache-dir` của `scripts/run_experiment.py`.
- **Không dùng cache `.npz`:** thêm `--no-cache`.

Xóa thư mục này lúc nào cũng được; code sẽ tải lại khi cần.
