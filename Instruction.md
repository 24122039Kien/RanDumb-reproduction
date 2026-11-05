# Hướng dẫn chạy lại thực nghiệm bài RanDumb

**Bài báo:** A. Prabhu, S. Sinha, P. Kumaraguru, P. H. S. Torr, O. Sener, P. K. Dokania, "RanDumb: Random Representations Outperform Online Continually Learned Representations," NeurIPS 2024.
[Trang chính thức](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4309616aaed8e848009bc4a7ef73b493-Abstract-Conference.html) · [Code của tác giả](https://github.com/drimpossible/RanDumb)

**Yêu cầu của môn:** [Overview](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement) · [Selection](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/selection) · [Reading](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reading) · [Reproduction](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction) · [Critique](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/analysis) · [Empirics](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/empirics)

> **Trạng thái của tài liệu này**
> - Chưa có lệnh, đoạn mã giả hay cấu hình nào ở đây được chạy thử trên dữ liệu thật hoặc trên Kaggle.
> - Mọi con số về thời gian đều là ước lượng. Bước 3 sẽ thay chúng bằng số đo thật.
> - Số liệu "paper" lấy từ bản NeurIPS, không lấy từ file arXiv v1 của nhóm. Hãy đối chiếu lại với PDF khi đưa vào deck.

---

## 1. Các quy định bắt buộc

| Quy định của môn | Việc nhóm phải làm | Hậu quả nếu sai |
|---|---|---|
| Tự cài lại từ mô tả trong bài. Code gốc chỉ để kiểm chứng và so sánh | Không chép dòng nào từ repo tác giả vào code của nhóm | **Cả project bị 0 điểm** |
| Không chép chữ, hình của bài và bài của nhóm khác | Tự vẽ lại toàn bộ hình | **Cả project bị 0 điểm** |
| Một repo code **public** suốt học kỳ | Commit sớm, commit đều, lịch sử commit sẽ được xem | Thiếu link repo: checkpoint xem như chưa nộp |
| Mọi run nằm trong **một** W&B project **public** | Mỗi con số trong deck và poster dẫn link tới run của nó | Số không truy được nguồn bị xem là "unbacked" |
| Dữ liệu tải bằng code từ Hugging Face hoặc Kaggle | Không chuẩn bị tay. Không có bản public thì tự upload lên HF/Kaggle của nhóm | Run không chạy được từ đầu đến cuối |
| Python 3.9+, numpy, scipy, scikit-learn, torch | Thư viện ngoài nhóm này (`wandb`, `datasets`, `timm`...) phải giải thích trong README | Mất điểm repo |
| Seed cố định và ghi lại, run lặp lại được | Xem [Bước 2](#bước-2-cài-phương-pháp-từ-mô-tả-trong-bài) | |
| Hàm quan trọng có docstring (mục đích, tham số, giá trị trả về) | README hướng dẫn cài đặt và chạy mọi thứ | Mất điểm repo |

**Lưu ý về AI:** Môn cho dùng AI, nhưng mọi thành viên phải giải thích được mọi phần ở buổi poster workshop.

**Lưu ý về file `randumb_benchmark.py`:** File này lặp lại pipeline của code tác giả chỉ để đo thời gian. **Không commit nó vào repo như code của nhóm.**

**Thứ tự công việc:**

```
Form được chấp nhận → Bước 0-1 chuẩn bị → Bước 2 tự cài + test → Bước 3 chạy thử đo giờ
→ Bước 4 chạy thực nghiệm → Bước 5 so với code gốc → Bước 6 phân tích sai lệch
→ Bước 7 README + deck → Bước 8 Critique + Empirics + poster
```

---

## Bước 0. Chuẩn bị tài khoản và repo

1. Đảm bảo form đăng ký đã được giảng viên chấp nhận. Từ lúc đó, đổi bài cần giảng viên đồng ý.
2. Tạo repo GitHub public.
3. Tạo W&B project **public**, ví dụ `randumb-repro`. Vào phần cài đặt project để kiểm tra chế độ hiển thị.
4. Trên Kaggle, tạo notebook và chọn:
   - **Accelerator: GPU T4 x2.** Không có lựa chọn "TPU T4". Không dùng TPU cho bài này.
   - **Internet: On.** Kaggle có thể yêu cầu xác minh số điện thoại.
5. Lưu `WANDB_API_KEY` trong **Add-ons → Secrets** của Kaggle. **Tuyệt đối không ghi key vào code hay commit lên repo.**

Thông số Kaggle mình tìm được từ các blog, không phải tài liệu chính thức: 4 CPU, khoảng 29-30 GB RAM, mỗi T4 16 GB VRAM, tối đa 12 giờ mỗi phiên, khoảng 30 giờ GPU mỗi tuần. Hãy kiểm tra lại trong tài khoản của nhóm.

**Cấu trúc repo đề xuất:**

```
README.md                  # cài đặt, cách chạy, bảng mapping, trích dẫn phần tái sử dụng
requirements.txt           # ghim phiên bản
src/randumb/
  features.py              # RFF, RP+ReLU
  stats.py                 # tích lũy trung bình lớp và hiệp phương sai
  classifier.py            # OAS, ridge, dự đoán
  data.py                  # tải dữ liệu từ HF/Kaggle, chuẩn hóa, flip
  tracking.py              # bọc W&B
scripts/
  run_experiment.py        # một run theo tham số dòng lệnh
  extract_vit_features.py  # trích đặc trưng cho Benchmark F
notebooks/                 # notebook Kaggle cho từng thực nghiệm
tests/                     # kiểm thử ở Bước 2
verification/              # script clone và chạy code gốc, không chứa code gốc
```

Repo của tác giả dùng giấy phép GPL-3.0. Chỉ `git clone` vào thư mục tạm lúc chạy, ghim theo commit cụ thể, **không đưa code đó vào repo của nhóm**.

---

## Bước 1. Chọn phạm vi (Scope)

Nhóm phải nêu lựa chọn và lý do trong deck.

| Scope | Yêu cầu | Với bài này |
|---|---|---|
| **Full** | Tất cả thực nghiệm chính, trên cùng dataset hoặc dataset tương đương | Rất nặng. Benchmark F có khoảng 15 phương pháp deep learning do tác giả tự chạy bằng LAMDA-PILOT, trên 2 backbone |
| **Core** | Ít nhất 2 thực nghiệm quan trọng nhất, gồm so sánh chính, cộng 1 phân tích do nhóm tự thiết kế | Khả thi, và phân tích tự thiết kế có sẵn chủ đề tốt |

**Khuyến nghị: chọn Core, và làm thêm các phân tích rẻ của Bảng 6.**

Lý do có thể ghi vào deck: số liệu đối thủ ở Benchmark A-E được tác giả lấy từ các bài khác, nên phần lõi là số của RanDumb. Phần tốn kém nhất là chạy lại các phương pháp prompt-tuning ở Benchmark F, vốn không phải đóng góp chính của bài.

| Mã | Thực nghiệm | Vị trí trong bài | Vai trò |
|---|---|---|---|
| **E1** | RanDumb trên pixel, 5 dataset | Bảng 1 trái, Bảng 2-5 | **Bắt buộc**, so sánh chính |
| **E2** | RanDumb và NCM trên đặc trưng ViT | Bảng 1 phải, Bảng 7 | **Bắt buộc** |
| E3 | Bỏ từng thành phần | Bảng 6 | Làm thêm |
| E4 | Có và không có flip | Bảng 6 | Làm thêm |
| E5 | Thay đổi λ | Bảng 6 | Làm thêm |
| E6 | So với RP+ReLU | Bảng 6 | Làm thêm |
| E7 | Thay đổi số chiều D | Hình 3 | Làm thêm |
| **E8** | Độ nhạy với chuẩn hóa, γ, cách shrink | Nhóm tự thiết kế | **Bắt buộc với Core** |
| **E9** | So với code gốc | Yêu cầu của môn | **Bắt buộc** vì bài có code |

**Không chạy lại, chỉ trích dẫn:** số của các phương pháp đối thủ (A-E), các dòng Joint, và bảng kiến trúc mạng (Bảng 6 phải, vì đó là thiết lập task-incremental khác hẳn). Hãy ghi rõ điều này trong deck.

---

## Bước 2. Cài phương pháp từ mô tả trong bài

Phần này tóm tắt công thức ở Mục 2 và 3 của bài, cộng các điểm bài không nói rõ. Nhóm tự viết code từ đây.

### 2.1. Biểu diễn ngẫu nhiên (Random Fourier Features)

Lấy mẫu `ω₁..ω_D ~ N(0, 2γI)` **một lần** trước khi thấy dữ liệu, rồi giữ nguyên.

Bài viết công thức `φ(x) = (1/√D) [cos(ωᵢᵀx), sin(ωᵢᵀx)]`, có **2D chiều**. Nhưng bài cũng nói dùng RFF của scikit-learn với kích thước 25K, tức dạng `√(2/D) cos(ωᵢᵀx + bᵢ)` với `bᵢ ~ U(0, 2π)`, có **D chiều**.

Hãy cài dạng D chiều để khớp "25K embedding", và ghi mâu thuẫn này vào phần phân tích sai lệch.

RFF là một phần cốt lõi của phương pháp, nên nên tự cài. `sklearn.kernel_approximation.RBFSampler` chỉ dùng để đối chiếu trong test.

### 2.2. Tích lũy thống kê

Bài xử lý từng mẫu một, chỉ cập nhật trung bình lớp và hiệp phương sai. Không lưu mẫu nào. Tích lũy theo lô cho kết quả **giống hệt** về toán học, và nhanh hơn nhiều:

```text
Khởi tạo: A = 0 (D×D), tong[c] = 0, dem[c] = 0, N = 0
Với mỗi lô Φ (B×D) có nhãn y:
    A += ΦᵀΦ
    với mỗi lớp c trong lô: tong[c] += tổng φ của lớp c; dem[c] += số mẫu
    N += B
μ_c = tong[c] / dem[c]
S   = (A - Σ_c dem[c] μ_c μ_cᵀ) / N
```

Không giữ ma trận N×D trong bộ nhớ, chỉ giữ `A`. Với flip, cả ảnh gốc và ảnh lật đều được tính là mẫu. MNIST và Benchmark F không dùng flip.

Dùng `float64` cho `A` nếu đủ bộ nhớ, vì ta cộng dồn nhiều số rất nhỏ. Lưu `A`, `tong`, `dem` ra file để E5 không phải tính lại.

### 2.3. Co hệ số (OAS) và ridge

Bài dùng OAS rồi lấy nghịch đảo của `(S + λI)`. Với `p = D`, `n = N`:

```text
mu    = trace(S) / p
alpha = trung bình của S_ij²
rho   = min(1, (alpha + mu²) / ((n + 1) (alpha - mu²/p)))
S_oas = (1 - rho) S + rho mu I
```

| Dataset | λ (theo bài) |
|---|---|
| MNIST | 10⁻⁶ |
| CIFAR-10, CIFAR-100 | 10⁻⁵ |
| TinyImageNet, miniImageNet | 10⁻⁴ |
| Benchmark F | **Bài không nêu**, phải ghi là sai lệch tiềm năng |

### 2.4. Dự đoán

Công thức (3) của bài là một **khoảng cách** Mahalanobis, trong khi công thức (1) viết `arg max`. Hãy lấy lớp có khoảng cách **nhỏ nhất**. Số hạng `φᵀS⁻¹φ` giống nhau ở mọi lớp nên có thể bỏ:

```text
Giải (S_oas + λI) w_c = μ_c bằng Cholesky (nếu lỗi thì dùng lstsq)
Điểm của lớp c = φᵀw_c - ½ μ_cᵀw_c
Lớp dự đoán   = lớp có điểm lớn nhất
```

Giải hệ phương trình như trên, **không** tính nghịch đảo tường minh.

### 2.5. Những điểm bài không nói rõ

Mỗi điểm phải được nhóm tự quyết, ghi lại, và đưa vào phân tích sai lệch:

| Điểm chưa rõ | Bài nói | Code gốc làm (theo bản tóm tắt khi đọc repo) |
|---|---|---|
| Chuẩn hóa ảnh | Chỉ viết "normalize all images" | Đưa về [0,1], trừ mean rồi chia std theo từng kênh |
| γ | 1.0 | `gamma='scale'` |
| Vị trí OAS | Viết chung là hiệp phương sai đã shrink | OAS theo **từng lớp** rồi gộp lại |
| λ | Theo dataset | README ghi λ = 0 |
| Cách học | Online từng mẫu | Fit cả lô một lần |
| Flip ở Benchmark F | Không flip | `runall.sh` có cả lượt `--augment` trên đặc trưng ViT |

**Cảnh báo về γ:** Theo tính toán của mình (chưa chạy), nếu chuẩn hóa từng kênh với γ = 1 thì `ωᵀx` có độ lệch chuẩn khoảng √(2·3072) ≈ 78 radian, nên đặc trưng gần như nhiễu và độ chính xác có thể rất thấp. Vì vậy hãy chạy E8 **trước** E1 ở D nhỏ để chọn cấu hình mặc định, rồi ghi rõ lý do chọn.

### 2.6. Kiểm thử bắt buộc (`tests/`)

Các test này phải qua trước khi chạy thực nghiệm:

1. **RFF xấp xỉ kernel:** `φ(x)ᵀφ(y)` tiến gần `exp(-γ‖x-y‖²)` khi D tăng.
2. **Online bằng batch:** cập nhật từng mẫu và theo lô cho cùng `μ_c`, `S` trên tập nhỏ.
3. **Không phụ thuộc thứ tự:** xáo thứ tự mẫu, kết quả không đổi. Đây là cơ sở để Benchmark A-E dùng chung một kết quả.
4. **OAS:** khớp `sklearn.covariance.OAS` trên dữ liệu nhỏ. So sánh như vậy được phép, vì thư viện chỉ dùng để kiểm chứng.
5. **Dự đoán:** khớp phép tính khoảng cách Mahalanobis đầy đủ, viết chậm, trên tập nhỏ.
6. **Seed:** cùng seed cho kết quả giống hệt.

### 2.7. Seed và độ lặp lại

- Đặt seed cho `random`, `numpy`, `torch` ở đầu mỗi run và log vào W&B.
- Dùng seed `0, 1, 2` cho E1 và E2. Bài báo cho biết đổi seed của kernel làm lệch khoảng ±0,2.
- Ghim phiên bản trong `requirements.txt` và log `torch.__version__`, tên GPU vào config của run.

---

## Bước 3. Chạy thử để đo giờ (làm trước tiên trên Kaggle)

Đây là bước thay các con số ước lượng bằng số đo thật.

1. Chạy CIFAR-10 với `D = 1000, 2000, 5000`, seed 0, trên GPU T4. Log `time_features_s`, `time_stats_s`, `time_solve_s`, `time_predict_s`, `peak_ram_gb`, `peak_vram_gb`.
2. Ngoại suy theo độ phức tạp:
   - Tích lũy `A`: tỉ lệ `N·D²`
   - Giải hệ: tỉ lệ `D³`
   - Tính RFF: tỉ lệ `N·d·D`
3. Quyết định D cuối cùng và cách chia phiên trước khi chạy thật.

**Bộ nhớ (tính lý thuyết cho `A` và `S`, D×D):**

| D | float32 | float64 |
|---|---|---|
| 5.000 | 0,1 GB | 0,2 GB |
| 10.000 | 0,4 GB | 0,8 GB |
| 15.000 | 0,9 GB | 1,8 GB |
| 25.000 | 2,5 GB | 5 GB |

Lúc giải hệ cần khoảng gấp đôi, vì phải giữ thêm bản sao của ma trận.

**Dự trù thời gian (ước lượng, chưa đo):**

| Tác vụ | Dự trù trên T4 | Ghi chú |
|---|---|---|
| Một lượt CIFAR có flip, D = 25.000 | Vài phút | Tính từ số phép toán |
| Một lượt TinyImageNet có flip, D = 25.000 | Khoảng gấp đôi CIFAR | 200K mẫu |
| Trích đặc trưng ViT cho Benchmark F | 30-60 phút | Chưa đo |

Tác giả chạy trên CPU 48 nhân, 392 GB RAM, mỗi thí nghiệm dưới 30 phút. Mình từng đo pipeline **của code gốc** trên dữ liệu giả lập (laptop, 4 luồng): khoảng 2,1 phút ở D = 1000 và 3,6 phút ở D = 2000. Số đó không đại diện cho code của nhóm.

Nếu không đủ tài nguyên cho D = 25.000, hãy dùng D = 15.000 và **nêu rõ đây là sai lệch**. Bài báo cho biết từ 15K lên 25K, độ chính xác tăng thêm 0,1 / 1,4 / 1,1 / 0,2 điểm trên MNIST / CIFAR-10 / CIFAR-100 / TinyImageNet.

---

## Bước 4. Chạy thực nghiệm

### 4.1. Quy tắc W&B cho mọi run

```python
import wandb

run = wandb.init(
    project="randumb-repro",
    group="E1",
    name="E1-cifar100-D25000-seed0",
    tags=["lab:reproduction", "E1"],
    config=config,  # dataset, D, gamma, lam, normalization, flip, seed, dtype, device
)
wandb.log({"final_accuracy": accuracy, "paper_accuracy": paper, "relative_deviation_pct": deviation})
run.finish()
```

- **Metric chính:** độ chính xác trên tập test của mọi lớp sau khi đi hết một lượt dữ liệu.
- **Sai lệch tương đối:** `(team - paper) / |paper| × 100%`.
- **Thực nghiệm phần Empirics** dùng **cùng project** nhưng gắn thẻ `lab:empirics`.

### 4.2. Dữ liệu

Tải bằng code trong script, ví dụ `load_dataset("uoft-cs/cifar10")`.

| Dataset | Nguồn đề xuất | Train / Test | Lưu ý |
|---|---|---|---|
| MNIST | HF `ylecun/mnist` | 60K / 10K | 784 chiều, không flip |
| CIFAR-10 | HF `uoft-cs/cifar10` | 50K / 10K | 3072 chiều |
| CIFAR-100 | HF `uoft-cs/cifar100` | 50K / 10K | Dùng nhãn `fine_label` |
| TinyImageNet | HF `Maysee/tiny-imagenet` | 100K / 10K | Thu về 32×32. Code gốc dùng nguồn này |
| miniImageNet | HF `timm/mini-imagenet` | 50K / 10K (val) | Thu về 32×32 |

**Về miniImageNet:** Code gốc tự tạo từ ImageNet theo danh sách lớp riêng, còn trang dataset của timm nói cách chia của họ khác bản thường dùng. Đây là "dataset tương đương" và phải ghi là nguồn sai lệch. Bản này giữ ảnh gốc nên có thể nặng, và cần kiểm tra điều khoản sử dụng.

Mình chưa tải thử nguồn nào ở trên. Hãy kiểm tra tên cột và số mẫu ngay khi tải: `assert len(train) == ...`.

### 4.3. E1: RanDumb trên pixel (thực nghiệm chính)

**Cấu hình:** D = 25.000, λ theo bảng 2.3, flip (trừ MNIST), 3 seed.

| Dataset | Paper | Team (TB ± ĐLC) | Sai lệch % | W&B |
|---|---|---|---|---|
| MNIST | 98,3 | | | |
| CIFAR-10 | 55,6 | | | |
| CIFAR-100 | 28,6 | | | |
| TinyImageNet | 11,6* | | | |
| miniImageNet | 17,7 | | | |

\* Bài báo không nhất quán: Bảng 3-5 ghi 11,6 còn dòng RanDumb ở Bảng 6 ghi 11,1. Theo dòng λ của Bảng 6, 11,6 ứng với λ = 10⁻⁴ và 11,1 ứng với λ = 10⁻⁵. Đây là chi tiết đáng nêu trong deck.

Vì RanDumb không phụ thuộc thứ tự và không lưu mẫu, một kết quả này được dùng để so ở Benchmark A, B.1, B.2, C, D, E. Nhóm đặt số của mình cạnh số đối thủ được trích từ bài (ghi rõ nguồn), không cần chạy lại đối thủ.

Vẫn phải tự kiểm chứng tiền đề này bằng test 3 ở mục 2.6.

### 4.4. E2: Benchmark F trên đặc trưng ViT

Benchmark F **không dùng RFF**: bài chỉ học bộ phân loại tuyến tính trên đặc trưng 768 chiều.

**Trích đặc trưng** (`scripts/extract_vit_features.py`, cần GPU):
- Hai mô hình ViT-B/16 từ `timm`, tiền huấn luyện trên ImageNet-1K và ImageNet-21K.
- Lưu đặc trưng thành artifact của W&B hoặc Kaggle dataset để khỏi trích lại.

**Bẫy cần ghi lại khi đối chiếu với code gốc:**
- Code gốc dùng tên `vit_base_patch16_224` và `vit_base_patch16_224_in21k`. Trọng số mặc định ứng với các tên này có thể khác nhau giữa các phiên bản `timm`, nên phải log tên và phiên bản chính xác.
- Code gốc **không** chuẩn hóa mean/std trước khi đưa vào ViT. Bài không nói gì về điểm này. Hãy thử cả hai cách.
- Theo bản tóm tắt mình đọc, `get_feats_vit.py` có vẻ có lỗi cú pháp và biến chưa định nghĩa. Nếu chạy thật mà đúng vậy, đây là phát hiện đáng ghi cho E9.

**Dữ liệu:** ImageNet-R, ImageNet-A, CUB, OmniBenchmark, VTAB, Cars. Phải dùng cách chia của LAMDA-PILOT / RevisitingCIL. Mình chưa xác minh có bản nào trên HF hay Kaggle đúng cách chia này. Nếu không có, nhóm tự tải về rồi upload lên HF hoặc Kaggle dataset của nhóm, đúng như quy định của môn. Tôn trọng giấy phép của từng dataset khi upload lại.

| Backbone | Phương pháp | CIFAR | IN-A | IN-R | CUB | OB | VTAB | Cars |
|---|---|---|---|---|---|---|---|---|
| IN-21K | RanDumb (paper) | 86,8 | 42,2 | 64,9 | 88,5 | 75,3 | 92,4 | 67,1 |
| IN-21K | NCM (paper) | 83,5 | 41,4 | 54,8 | 86,5 | - | 88,5 | - |
| IN-1K | RanDumb (paper) | 84,5 | 49,5 | 66,9 | 88,0 | - | 93,6 | - |
| IN-1K | NCM (paper) | 78,3 | 44,3 | 62,5 | 84,8 | - | 88,2 | - |

Bảng 7 của bản NeurIPS chỉ có 5 cột. OB và Cars chỉ xuất hiện ở Bảng 1 cho IN-21K, dù bài nói dùng 7 dataset. Hãy nêu điểm này.

Nếu thiếu thời gian, ưu tiên CIFAR, CUB, IN-R, IN-A, VTAB với cả hai backbone, và ghi rõ phần đã bỏ.

### 4.5. E3-E7: các phân tích của Bảng 6 và Hình 3

Chạy với 1 seed, cấu hình như E1 trừ yếu tố đang xét.

| Thực nghiệm | Cấu hình | MNIST | CIFAR-10 | CIFAR-100 | T-ImNet | m-ImNet |
|---|---|---|---|---|---|---|
| Đầy đủ (Bảng 6) | | 98,3 | 55,6 | 28,6 | 11,1 | 17,7 |
| **E3** | Bỏ khử tương quan (Kernel-NCM) | 83,8 | 30,0 | 12,0 | 4,7 | 8,9 |
| **E3** | Bỏ RFF (SLDA trên pixel) | 88,0 | 41,6 | 19,0 | 8,0 | 12,9 |
| **E3** | Bỏ cả hai (NCM) | 82,1 | 28,5 | 10,4 | 4,1 | 7,28 |
| **E4** | Không flip | 98,3 | 52,5 | 26,9 | 10,7 | 16,6 |
| **E5** | λ = 10⁻⁶ | 98,3 | 53,9 | 27,8 | 10,3 | 15,8 |
| **E5** | λ = 10⁻⁵ | - | 55,6 | 28,6 | 11,1 | 15,9 |
| **E5** | λ = 10⁻⁴ | 96,6 | 52,6 | 26,1 | 11,6 | 17,7 |
| **E6** | RP+ReLU (`max(0, xᵀW)`) | 95,2 | 48,8 | 23,1 | 9,7 | 15,7 |

- **E5:** dùng lại `A`, `tong`, `dem` đã lưu, chỉ giải lại hệ phương trình.
- **E6:** bài dùng "best embedding size" mà không nêu cụ thể. Hãy thử vài D và ghi D đã chọn.
- **E7:** chạy `D ∈ {1K, 2K, 5K, 10K, 15K, 20K, 25K}` trên MNIST, CIFAR-10, CIFAR-100, TinyImageNet. Vẽ lại Hình 3 bằng biểu đồ của nhóm. Mốc đối chiếu: từ 1K lên 25K tăng 3,6 / 10,4 / 7,0 / 2,5 điểm.

### 4.6. E8: phân tích độ nhạy do nhóm thiết kế

Đây là phân tích bắt buộc của Core scope. Nó nhắm đúng những gì bài bỏ ngỏ ở mục 2.5.

Chạy trên CIFAR-10 và CIFAR-100 với D = 5.000 và một D lớn, thay đổi từng yếu tố:

| Yếu tố | Các giá trị |
|---|---|
| Chuẩn hóa | z-score theo kênh; L2 theo ảnh; chỉ chia 255 |
| γ | 1,0; `1/(d·Var(X))`; vài bội số quanh giá trị đó |
| Shrink | OAS gộp chung; OAS theo từng lớp rồi gộp; không OAS |
| λ tương đối | λ cố định; λ tỉ lệ `trace(S)/D` |

**Giả thuyết đáng kiểm tra (theo tính toán, chưa chạy):** với RFF dạng `√(2/D)cos`, trị riêng trung bình của `S` vào cỡ `1/D`. Như vậy cùng một λ cố định có tác dụng rất khác nhau khi D thay đổi, nên đường trong Hình 3 có thể đang trộn lẫn hiệu ứng của D và của λ.

---

## Bước 5. E9: so sánh với code gốc

Môn yêu cầu chạy code gốc, so đầu ra, và chỉ ra hai pipeline khác nhau ở đâu.

1. Trong `verification/`, viết script `git clone` repo tác giả vào thư mục tạm và ghim commit.
2. Chạy đúng nguyên bản trước, trên MNIST và CIFAR-10 ở D nhỏ (1.000-5.000). Log thẻ `official-code`.
3. So từng tầng, không chỉ so độ chính xác cuối:
   - Dữ liệu sau chuẩn hóa (mean, std, vài mẫu đầu)
   - γ và phân bố của đặc trưng
   - Trung bình lớp và `S`
   - Độ chính xác từng lớp
4. Ghi lại các khác biệt của mục 2.5, và kiểm tra tuyên bố trong README của tác giả rằng code rút gọn lệch trong khoảng ±0,8% so với bài.

**Về tốc độ của code gốc:**
- `main.py` chỉ dùng NumPy và scikit-learn nên **không dùng GPU**.
- Mình đo trên dữ liệu giả lập: `OAS` mặc định có `store_precision=True`, nên tính thêm một ma trận nghịch đảo cho **từng lớp**. Trên CIFAR-100 ở D lớn, việc này rất chậm (ở D = 8.000, mỗi lớp 1000 mẫu: 88,6 giây so với 2,1 giây khi tắt).
- Nếu cần chạy D lớn, hãy tạo **bản sao kiểm chứng** với `store_precision=False`, và ghi rõ đã sửa gì. Trên một tập nhỏ mình thử, dự đoán không đổi. Hãy xác nhận lại trên dữ liệu thật.
- Đây là ví dụ đúng loại "phát hiện từ code gốc" mà môn đánh giá cao.

---

## Bước 6. Phân tích sai lệch

Với mỗi khoảng cách đáng chú ý, trả lời đủ 4 câu:

1. **Khoảng cách bao nhiêu?** Số của nhóm, số của bài, sai lệch %, kèm link W&B.
2. **Nguyên nhân giả định là gì?** Chọn trong: cấu hình bài không nêu; khác thư viện và phiên bản; khởi tạo ngẫu nhiên; phần cứng và độ chính xác số; khác dataset.
3. **Đã kiểm chứng thế nào?** Chỉ đổi đúng một yếu tố rồi chạy lại, hoặc đối chiếu với code gốc.
4. **Kết luận:** Đã giải thích được, giải thích một phần, hay chưa giải thích được. Được phép trả lời "chưa giải thích được", nhưng phải nói đã thử gì.

Môn không yêu cầu khớp số tuyệt đối. Điểm nằm ở việc **nói chính xác lệch bao xa và vì sao**.

Các nguồn sai lệch nên kiểm tra trước: mục 2.5; D thực dùng; float32/float64; miniImageNet khác bản gốc; phiên bản `timm` và trọng số ViT; λ của Benchmark F.

---

## Bước 7. README và slide deck (45% điểm)

### README phải có

- Cách cài đặt và chạy từ đầu đến cuối.
- Lý do dùng các thư viện ngoài nhóm chuẩn, và trích dẫn mọi phần tái sử dụng.
- Seed đã dùng.
- **Bảng mapping** theo mẫu:

| Thực nghiệm | Notebook/script | W&B run | Kết quả cho thấy |
|---|---|---|---|
| E1, CIFAR-100, D = 25K | `notebooks/e1.ipynb` | [link] | Một dòng nhận xét |

### Yêu cầu của deck

- [ ] Một file **pptx**, **20-40 slide** (tính cả phụ lục), **dưới 20 MB**.
- [ ] **Mỗi slide** có speaker notes, **tối đa 500 từ**. Không có thuyết trình, nên deck và notes là toàn bộ bài nộp.
- [ ] Trích dẫn kiểu **IEEE** có đánh số, đặt **cuối chính slide** đó. **Không** có slide References riêng.
- [ ] **Mọi hình do nhóm tự vẽ.** Không dán hình từ bài, không dán code thô lên slide.
- [ ] Viết bằng lời của nhóm, không dùng nguyên câu trong bài.
- [ ] Nộp link repo public thành **mục riêng** trên Moodle.

### Nội dung của deck

| Phần | Phải có |
|---|---|
| Bối cảnh | Bài toán, ví dụ thực tế **của nhóm**, ít nhất 2 phương pháp tiền nhiệm (ý chính, hạn chế, hệ quả của hạn chế). Gợi ý: SLDA, RanPAC, GDumb |
| Đóng góp | Theo lời của bài, rồi xếp lại thành lý thuyết / kỹ thuật / thực nghiệm. Nêu cái gì trước đây chưa làm được |
| Phương pháp | Ký hiệu thống nhất và bài toán hình thức; sơ đồ pipeline tự vẽ; từng thành phần (làm gì, vì sao cần, hoạt động thế nào); tính chất lý thuyết; độ phức tạp thời gian và bộ nhớ so với các tiền nhiệm |
| Phương pháp đánh giá | Dataset, metric, giao thức, thể hiện bằng hình và bảng |
| Tái lập | Lựa chọn scope và lý do; bảng paper và team cạnh nhau kèm sai lệch %; link W&B cho từng số; phân tích sai lệch; so sánh với code gốc |

---

## Bước 8. Critique, Empirics và poster (50% điểm)

### Critique: chỉ phê bình lý thuyết

Cần **ít nhất**: 3 điểm mạnh có bằng chứng; 2 hạn chế lý thuyết, nêu đủ rõ để thiết kế được thí nghiệm; 2 hướng phát triển, mỗi hướng gắn với một hạn chế.

### Empirics: thí nghiệm nhắm đúng từng hạn chế

Quy tắc giống hệt phần Reproduction, gắn thẻ `lab:empirics`. Các gợi ý dưới đây là **giả thuyết của mình, chưa được kiểm chứng**:

| Hạn chế cần lập luận | Dạng thí nghiệm | Cách thiết kế |
|---|---|---|
| Thống kê tích lũy coi mọi dữ liệu cũ như nhau, không có cơ chế quên, nên có thể tệ khi có **concept drift** | Dữ liệu vi phạm giả định | Cho phân bố của lớp đổi dần trong luồng. Vẽ độ chính xác theo mức drift |
| LDA giả định các lớp có **chung hiệp phương sai** | Dữ liệu vi phạm giả định | Tăng dần độ khác biệt hiệp phương sai giữa các lớp, so với mô hình dùng hiệp phương sai riêng |
| RFF trên pixel **không bất biến** với dịch chuyển ảnh | Ngoài phân bố | Dịch, xoay, đổi sáng ảnh test, vẽ đường suy giảm |
| Chi phí **`D²` bộ nhớ, `D³` thời gian** | Scaling | Đo giờ thật khi D tăng, so với cận lý thuyết |
| Fixed λ tương tác với D | Độ nhạy | Dùng kết quả E8 |

**Một phần mở rộng:** Áp RFF lên đặc trưng ViT. RanDumb không làm điều này ở Benchmark F, nhưng đó là cách làm của KLDA (AAAI 2025). Thí nghiệm này kiểm tra xem kernel còn cần thiết không khi đã có đặc trưng mạnh.

### Poster

- [ ] Một file **pptx**, **một slide duy nhất**, khổ **A2 dọc**.
- [ ] Mỗi hạn chế: lập luận lý thuyết đặt **ngay cạnh** bằng chứng đo được.
- [ ] Nếu chứng minh được hơn 2 hạn chế: chỉ đưa **2 hạn chế nổi bật nhất** lên poster, phần còn lại ghi trong `README.md` kèm bằng chứng.
- [ ] Mọi số thực nghiệm dẫn tới W&B run.
- [ ] Mọi thành viên trả lời được câu hỏi của giám khảo tại poster workshop.

---

## Danh sách kiểm tra trước khi nộp

**Đạo đức học thuật**
- [ ] Không có dòng nào chép từ repo tác giả. Repo gốc không nằm trong repo của nhóm.
- [ ] Không có chữ hay hình chép từ bài.

**Repo (5% điểm, cùng với W&B và định dạng)**
- [ ] Repo public, lịch sử commit đều đặn từ đầu học kỳ.
- [ ] `WANDB_API_KEY` và mọi token **không** nằm trong repo.
- [ ] Docstring cho các hàm quan trọng. Test ở mục 2.6 đều qua.
- [ ] README có bảng mapping. Chạy lại được từ đầu chỉ bằng README.

**W&B**
- [ ] Một project public duy nhất, mở được khi **chưa đăng nhập**.
- [ ] Thẻ `lab:reproduction` và `lab:empirics` tách biệt.
- [ ] Mỗi con số trong deck và poster có link tới đúng run.

**Moodle**
- [ ] File deck pptx dưới 20 MB. Link repo nộp riêng.
- [ ] File poster pptx.
- [ ] Theo dõi hạn nộp trên Moodle. Tài liệu này không ghi ngày vì trang môn học cũng không ghi.
