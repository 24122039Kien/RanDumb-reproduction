# Tái lập và phản biện RanDumb

Term project của môn **Nhập môn Học máy (Introduction to Machine Learning)**, chủ đề *never-ending learning*. Nhóm đọc, tự cài lại, chạy lại thực nghiệm và phản biện một bài báo đã công bố trong suốt học kỳ.

> **Trạng thái (19/11):** Toàn bộ code đã có trên `main`. E2 đã có kết quả; E1 và E8 đã sẵn sàng chạy, chưa có kết quả.

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

Trạng thái `src/randumb/`: toàn bộ module đã có trên `main`. Đặc trưng ViT cho E2 nằm ở `vit_features.py`, NCM ở `baselines.py`. Mốc cập nhật xem [Docs/report3.md](Docs/report3.md) và [Docs/report4.md](Docs/report4.md).

## Cài đặt và chạy

### Môi trường

Các run có số liệu trong bảng kết quả đều chạy trên **Kaggle** (GPU T4, Python 3.13). Laptop chỉ dùng để viết code, chạy test và thử với D nhỏ. Phiên bản thư viện trong [requirements.txt](requirements.txt) được ghim theo đúng môi trường Kaggle; các bản này cần Python 3.11 trở lên.

```bash
git clone https://github.com/24122039Kien/RanDumb-reproduction.git
cd RanDumb-reproduction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # cài package randumb từ src/
python -m pytest -q                # phải qua hết trước khi chạy thực nghiệm
```

Dữ liệu được tải tự động từ Hugging Face trong lần chạy đầu, không cần chuẩn bị tay.

| Biến môi trường | Bắt buộc | Ý nghĩa |
|---|---|---|
| `WANDB_API_KEY` | Khi log W&B | Khóa W&B. Trên Kaggle đặt trong **Add-ons → Secrets**, không ghi vào code |
| `WANDB_ENTITY` | Không | Team W&B, mặc định là team của nhóm |
| `HF_TOKEN` | Không | Token đọc của Hugging Face, giúp tải nhanh hơn |
| `RANDUMB_DATA_DIR` | Không | Nơi lưu dữ liệu, cache và đặc trưng; mặc định [data/](data/) |

### Chạy thử nhanh trên máy cá nhân

Không cần GPU hay W&B, khoảng một phút:

```bash
python scripts/run_experiment.py --group smoke --dataset mnist --dim 500 \
    --max-train 3000 --max-eval 1000 --no-wandb --device cpu
```

Kết quả in ra dạng JSON. Với cấu hình mặc định (γ = 1,0, chuẩn hóa L2), lệnh trên cho khoảng 92%, thấp hơn 98,3% của bài vì chỉ dùng 3.000 ảnh và D = 500.

### Chạy lại từng thực nghiệm

Cách dễ nhất là import notebook tương ứng vào Kaggle, bật GPU, Internet và Secret `WANDB_API_KEY`, rồi **Run All**. Mỗi notebook tự clone repo, cài thư viện và chạy test trước.

| Thực nghiệm | Notebook Kaggle | Lệnh tương đương | Ghi chú |
|---|---|---|---|
| Kiểm tra môi trường | [00_kaggle_smoke.ipynb](notebooks/00_kaggle_smoke.ipynb) | | GPU, phiên bản thư viện, tốc độ `float32`/`float64` |
| E1 và E8 pha B | [03_dang_sprint3.ipynb](notebooks/03_dang_sprint3.ipynb) | `python scripts/run_sprint3.py --group E1 --device cuda --out-dir outputs/sprint3 --execute` | Cấu hình đã chốt: γ = 1,0, chuẩn hóa L2, D = 25.000, λ theo bài. Bỏ `--execute` để chỉ xem danh sách run; thêm `--dataset mnist` để chia phiên. Chi tiết: [Docs/sprint3_Dang.md](Docs/sprint3_Dang.md) |
| E2 | [02_e2_vit.ipynb](notebooks/02_e2_vit.ipynb) | `python scripts/extract_vit_features.py --dataset cifar100 --backbone in21k`, rồi `python scripts/run_e2.py --dataset cifar100 --backbone in21k --method randumb` | Khoảng 1 giờ, phần lớn là trích đặc trưng ViT. Giao thức: [Docs/e2_datasets.md](Docs/e2_datasets.md) |
| E2 bổ sung | [04_e2_followup.ipynb](notebooks/04_e2_followup.ipynb) | `python scripts/run_e2.py ... --method ncm --ncm-metric cosine` và `... --split-seed 1` | Dùng lại file đặc trưng của E2 |
| E9 | | `python verification/compare_layers_e9.py --dataset cifar10 --num-samples 2000` | So từng tầng với cách làm của code gốc. Muốn chạy chính code gốc: `python verification/setup_official.py` clone vào `verification/_official/` (không commit). Xem [verification/README.md](verification/README.md) |

Mỗi script có `--help` liệt kê đủ tham số. Mọi run đều log seed, cấu hình, phiên bản thư viện và GPU lên W&B.

### Thư viện ngoài nhóm chuẩn của môn

Ngoài numpy, scipy, scikit-learn và torch, repo dùng:

| Thư viện | Lý do |
|---|---|
| `wandb` | Theo dõi thực nghiệm, môn yêu cầu |
| `datasets` | Tải dữ liệu từ Hugging Face bằng code |
| `timm` | Mô hình ViT-B/16 pre-trained cho E2 (Benchmark F) |
| `psutil` | Đo RAM đỉnh khi đo giờ |
| `pillow` | Giải mã, đổi kênh và thu nhỏ ảnh |
| `pytest` | Chạy kiểm thử |

`scikit-learn` chỉ dùng để đối chiếu trong test và trong `verification/`; RFF, OAS và LDA do nhóm tự cài trong [src/randumb/](src/randumb/).

## Bảng kết quả

Mỗi thực nghiệm có một dòng. Số liệu chi tiết từng run và phân tích sai lệch nằm trong file báo cáo được dẫn ở cột cuối.

| Thực nghiệm | Notebook/script | W&B run | Kết quả cho thấy |
|---|---|---|---|
| Kiểm tra môi trường Kaggle (Sprint 0) | [notebooks/00_kaggle_smoke.ipynb](notebooks/00_kaggle_smoke.ipynb) | [6vddfz6r](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/6vddfz6r) | Trên T4, nhân ma trận `float64` chậm hơn `float32` khoảng 14 lần; 5 dataset E1 đọc được từ Hugging Face |
| E2: RanDumb và NCM trên đặc trưng ViT-B/16 (4 dataset × 2 backbone) | [notebooks/02_e2_vit.ipynb](notebooks/02_e2_vit.ipynb) | nhóm `E2`, 16 run, link từng run trong [Docs/report4.md](Docs/report4.md) | RanDumb hơn NCM ở cả 8 cặp, giống bài. RanDumb lệch dưới ±4% so với bài trên CIFAR-100, CUB, ImageNet-R; ImageNet-A cao hơn bài 11-17%, có thể do nhóm tự chia train/test |
| E9: So sánh từng tầng với code gốc tác giả (Verification) | [notebooks/09_e9_verification.ipynb](notebooks/09_e9_verification.ipynb) | [icae4r9o](https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/icae4r9o) | Hai pipeline đồng nhất 100% về mặt giải tích (Agreement 100.0%, Cosine sim $W=1.000000$, $S_{\text{oas}}$ max diff $10^{-17}$). Phát hiện 6 điểm sai khác mấu chốt trong code tác giả so với bài báo (xem notebook và [verification/README.md](verification/README.md)). |

## Tính toàn vẹn học thuật

Phần cốt lõi của phương pháp được nhóm tự cài từ mô tả trong bài. Code của tác giả chỉ dùng để kiểm chứng và so sánh, không được chép vào repo. Mọi thư viện và phần tái sử dụng sẽ được liệt kê và trích dẫn tại đây.
