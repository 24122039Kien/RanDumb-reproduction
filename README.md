# Tái lập và phản biện RanDumb

Term project của môn **Nhập môn Học máy (Introduction to Machine Learning)**, chủ đề *never-ending learning*. Nhóm đọc, tự cài lại, chạy lại thực nghiệm và phản biện một bài báo đã công bố trong suốt học kỳ.

> **Trạng thái:** Đang ở giai đoạn chuẩn bị. Repo chưa có phần cài đặt phương pháp và chưa có kết quả thực nghiệm nào.

**Thành viên:** *(điền tên và MSSV)*
**W&B project:** [randumb-reproduction-24TNT](https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT) (public)

## Bài báo

A. Prabhu, S. Sinha, P. Kumaraguru, P. H. S. Torr, O. Sener, and P. K. Dokania, "RanDumb: Random Representations Outperform Online Continually Learned Representations," in *Advances in Neural Information Processing Systems*, vol. 37, 2024.

- [Trang chính thức tại NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4309616aaed8e848009bc4a7ef73b493-Abstract-Conference.html)
- [Code của tác giả](https://github.com/drimpossible/RanDumb) (chỉ dùng để kiểm chứng và so sánh)

## Bài báo nói gì?

Trong online continual learning, mô hình nhận dữ liệu theo luồng, chỉ được xem mỗi mẫu một lần, và các lớp xuất hiện lần lượt. Hướng phổ biến là vừa học bộ phân loại, vừa học biểu diễn bằng mạng sâu.

RanDumb đặt câu hỏi: biểu diễn học được trong điều kiện đó có thực sự tốt hơn một biểu diễn ngẫu nhiên không hề được học không? Theo các tác giả, câu trả lời là không, ít nhất trong các benchmark online và ít lưu mẫu mà họ đánh giá.

## Phương pháp

```
Ảnh → duỗi thành vector → Random Fourier Features → khử tương quan → lớp có trung bình gần nhất
                          (cố định, không học)      (Mahalanobis)
```

1. **Biểu diễn ngẫu nhiên:** Chiếu dữ liệu lên không gian nhiều chiều bằng Random Fourier Features, xấp xỉ RBF kernel. Phép chiếu được cố định trước khi thấy dữ liệu.
2. **Thống kê trực tuyến:** Mỗi mẫu chỉ cập nhật trung bình của lớp và một ma trận hiệp phương sai dùng chung. Không lưu lại mẫu nào.
3. **Phân loại:** Chọn lớp gần nhất theo khoảng cách Mahalanobis, tương đương LDA khi các lớp có xác suất tiên nghiệm bằng nhau.

Với mô hình pre-trained (Benchmark F), đặc trưng ViT thay cho phép chiếu ngẫu nhiên, còn bộ phân loại giữ nguyên.

Bài phù hợp với môn vì được xây từ các khối quen thuộc: kernel, LDA/GDA, ước lượng hiệp phương sai và regularization.

## Kế hoạch thực hiện

Nhóm dự kiến chọn **Core scope** (chưa chốt). Chi tiết từng bước nằm trong [Docs/Instruction.md](Docs/Instruction.md), lịch và phân công nằm trong [Docs/implementation_plan.md](Docs/implementation_plan.md).

| Mã | Nội dung | Vai trò |
|---|---|---|
| E1 | RanDumb trên pixel với MNIST, CIFAR-10/100, TinyImageNet, miniImageNet | So sánh chính |
| E2 | RanDumb và NCM trên đặc trưng ViT-B/16 (Benchmark F) | Thực nghiệm chính thứ hai |
| E3-E7 | Bỏ từng thành phần, flip, λ, RP+ReLU, số chiều D | Làm thêm nếu đủ thời gian |
| E8 | Độ nhạy với chuẩn hóa, γ và cách shrink | Phân tích do nhóm thiết kế |
| E9 | So sánh với code của tác giả | Bắt buộc vì bài có code |

**Sản phẩm của môn:** một slide deck (đọc bài và tái lập) và một poster A2 (phản biện và thực nghiệm chứng minh), được bảo vệ tại poster workshop.

## Nội dung repo

| File / thư mục | Mô tả |
|---|---|
| [README.md](README.md) | Giới thiệu project, cài đặt và cách chạy |
| [Docs/Instruction.md](Docs/Instruction.md) | Hướng dẫn chi tiết các bước chạy lại thực nghiệm theo yêu cầu của môn |
| [Docs/implementation_plan.md](Docs/implementation_plan.md) | Kế hoạch theo sprint, phân công và lịch làm việc |
| [Randumbofficial.pdf](Randumbofficial.pdf) | Bản arXiv của bài báo |
| [src/randumb/](src/randumb/) | Cài đặt phương pháp: RFF, thống kê, bộ phân loại, dữ liệu, W&B |
| [scripts/](scripts/) | Chạy một thực nghiệm, trích đặc trưng ViT |
| [notebooks/](notebooks/) | Notebook Kaggle cho từng thực nghiệm |
| [tests/](tests/) | Kiểm thử bắt buộc trước khi chạy thực nghiệm |
| [verification/](verification/) | Script kiểm chứng với code gốc, không chứa code gốc |
| [data/](data/) | Nơi lưu dữ liệu tải về và cache khi chạy; nội dung không được commit |

Các hàm trong `src/randumb/` hiện mới có chữ ký và docstring; phần cài đặt làm ở Sprint 1.

## Cài đặt và chạy

*Sẽ cập nhật khi có phần cài đặt.* Môn yêu cầu Python 3.9 trở lên, và mục này phải hướng dẫn đủ để chạy lại mọi thứ từ đầu đến cuối.

## Bảng kết quả

*Chưa có run nào.* Mỗi thực nghiệm sẽ có một dòng theo mẫu:

| Thực nghiệm | Notebook/script | W&B run | Kết quả cho thấy |
|---|---|---|---|
| *(ví dụ: E1, CIFAR-100)* | | | |

## Tính toàn vẹn học thuật

Phần cốt lõi của phương pháp được nhóm tự cài từ mô tả trong bài. Code của tác giả chỉ dùng để kiểm chứng và so sánh, không được chép vào repo. Mọi thư viện và phần tái sử dụng sẽ được liệt kê và trích dẫn tại đây.
