# Giải thích dễ hiểu: các thực nghiệm E1-E9 và tiến độ của repo

**Viết cho:** thành viên cần nắm nhanh đồ án mà chưa đọc kỹ bài báo
**Cập nhật:** 14/11/2026

Tài liệu này không có công thức. Muốn xem chi tiết kỹ thuật, đọc [Instruction.md](Instruction.md); muốn xem lịch, đọc [implementation_plan.md](implementation_plan.md).

---

## 1. Trước hết: RanDumb làm gì?

**Bài toán.** Máy nhận ảnh theo một dòng chảy: mỗi ảnh chỉ được xem **một lần**, các lớp mới xuất hiện dần (hôm nay học "mèo", "chó", mai mới gặp "ngựa"), và **không được lưu lại ảnh cũ**. Bài toán này gọi là *online continual learning*.

**Cách làm phổ biến.** Huấn luyện một mạng nơ-ron sâu để vừa tự học cách "nhìn" ảnh (học biểu diễn), vừa học cách phân loại.

**Câu hỏi của bài báo.** Trong điều kiện khắt khe như vậy, mạng sâu có thật sự học được cách nhìn tốt hơn một cách nhìn **ngẫu nhiên, không học gì** không?

**RanDumb** trả lời bằng một phương pháp rất đơn giản, gồm 3 bước:

```
Ảnh  →  [1] Chiếu ngẫu nhiên  →  [2] Cập nhật thống kê  →  [3] Chọn lớp gần nhất
         (cố định, không học)       (trung bình, hiệp         (có tính đến tương
                                    phương sai)               quan giữa các chiều)
```

1. **Chiếu ngẫu nhiên (Random Fourier Features).** Biến mỗi ảnh thành một vector rất dài, ví dụ **D = 25.000 chiều**, bằng một phép biến đổi ngẫu nhiên chọn sẵn từ đầu và không bao giờ thay đổi. Hãy hình dung nhìn bức ảnh qua 25.000 "lăng kính" ngẫu nhiên.
2. **Cập nhật thống kê.** Mỗi ảnh mới chỉ dùng để cập nhật **trung bình của lớp** và **một ma trận hiệp phương sai dùng chung**, rồi bỏ đi. Không lưu ảnh nào.
3. **Phân loại.** Ảnh mới thuộc lớp có trung bình **gần nó nhất**, đo theo khoảng cách Mahalanobis, tức là khoảng cách có tính tới việc các chiều tương quan với nhau.

**Kết luận của bài:** phương pháp "ngớ ngẩn" này **thắng** nhiều phương pháp học sâu phức tạp trên các benchmark online. Việc của nhóm là kiểm tra lại kết luận đó.

### Một số từ hay gặp

| Từ | Nghĩa đơn giản |
|---|---|
| **D** | Số chiều sau khi chiếu ngẫu nhiên. Bài dùng 25.000 |
| **γ (gamma)** | Độ "nhạy" của phép chiếu. Quá lớn thì kết quả thành nhiễu |
| **λ (lambda)** | Một lượng nhỏ cộng thêm để phép tính ổn định, mỗi dataset một giá trị |
| **Chuẩn hóa** | Đưa giá trị pixel về cùng thang đo trước khi chiếu |
| **Flip** | Thêm bản lật ngang của ảnh khi học, để có thêm dữ liệu |
| **OAS** | Cách làm ma trận hiệp phương sai bớt nhiễu khi ít dữ liệu |
| **NCM** | Chọn lớp có trung bình gần nhất, không khử tương quan. Là bản "thô" của RanDumb |
| **Backbone ViT** | Mạng Vision Transformer đã được người khác huấn luyện sẵn, dùng để lấy đặc trưng ảnh |
| **Seed** | Số khởi tạo ngẫu nhiên; cùng seed thì cho cùng kết quả |
| **W&B run** | Một lần chạy được ghi lại trên Weights & Biases, có link riêng để dẫn chứng |

---

## 2. Các thực nghiệm E1-E9

Mỗi thực nghiệm trả lời **một câu hỏi**. Môn yêu cầu tái lập kết quả của bài, nên phần lớn các E là "chạy lại một bảng trong bài, xem số của nhóm có giống không".

### Tổng quan

| Mã | Câu hỏi | Ứng với chỗ nào trong bài | Bắt buộc? | Người làm |
|---|---|---|---|---|
| **E1** | RanDumb trên ảnh thô có đạt đúng độ chính xác bài công bố không? | Bảng 1 trái, Bảng 2-5 | **Có** | Đăng |
| **E2** | Thay phép chiếu ngẫu nhiên bằng ViT đã huấn luyện sẵn thì sao? | Bảng 1 phải, Bảng 7 | **Có** | Kiên |
| E3 | Bỏ từng bước của RanDumb thì kém đi bao nhiêu? | Bảng 6 | Không, đã bỏ cho deck | |
| E4 | Có flip và không flip khác nhau bao nhiêu? | Bảng 6 | Không, đã bỏ cho deck | |
| E5 | λ ảnh hưởng thế nào? | Bảng 6 | Không, đã bỏ cho deck | |
| E6 | Thay RFF bằng một phép chiếu ngẫu nhiên khác (RP+ReLU) thì sao? | Bảng 6 | Không, đã bỏ cho deck | |
| E7 | Tăng số chiều D thì tốt lên bao nhiêu? | Hình 3 | Không, đã bỏ cho deck | |
| **E8** | Những chỗ bài **không nói rõ** (chuẩn hóa, γ, OAS) ảnh hưởng bao nhiêu? | Nhóm tự thiết kế | **Có** | Đăng |
| **E9** | Code của nhóm có cho kết quả giống code gốc của tác giả không? | Yêu cầu của môn | **Có** | Huy |

E3-E7 không bắt buộc với Core scope. Lý do bỏ khỏi giai đoạn deck và khả năng dùng lại cho poster nằm ở mục 2.3.

### 2.1. Bốn thực nghiệm bắt buộc

#### E1: so sánh chính

- **Làm gì:** chạy RanDumb trên 5 bộ ảnh: MNIST (chữ số viết tay), CIFAR-10, CIFAR-100 (ảnh nhỏ 32×32), TinyImageNet, miniImageNet. Dùng D = 25.000, mỗi bộ chạy 3 seed, tổng **15 run**.
- **So với:** số của bài, ví dụ MNIST 98,3%, CIFAR-10 55,6%, CIFAR-100 28,6%, TinyImageNet 11,6%, miniImageNet 17,7%.
- **Vì sao quan trọng nhất:** đây là kết quả trung tâm của bài. Vì RanDumb không phụ thuộc thứ tự dữ liệu, một con số này dùng được cho cả 6 benchmark A-E của bài, không cần chạy lại các phương pháp đối thủ.
- **Code:** `scripts/run_experiment.py`.

#### E2: dùng ViT thay phép chiếu ngẫu nhiên

- **Làm gì:** lấy đặc trưng ảnh từ ViT đã huấn luyện sẵn (2 phiên bản: huấn luyện trên ImageNet-1K và ImageNet-21K), rồi chạy hai bộ phân loại: **RanDumb** và **NCM**.
- **Dataset:** CIFAR-100, CUB (200 loài chim), ImageNet-R (tranh vẽ, hoạt hình), ImageNet-A (ảnh "khó"). Tổng 4 × 2 × 2 = **16 run**.
- **So với:** số của bài, ví dụ CIFAR-100 với ViT IN-21K: RanDumb 86,8%, NCM 83,5%.
- **Vì sao quan trọng:** kiểm tra tuyên bố thứ hai của bài, rằng chỉ học một bộ phân loại đơn giản trên đặc trưng có sẵn đã thắng nhiều cách tinh chỉnh mạng phức tạp.
- **Code:** `scripts/extract_vit_features.py`, `scripts/run_e2.py`, notebook `notebooks/02_e2_vit.ipynb`.

#### E8: phân tích độ nhạy do nhóm tự thiết kế

- **Làm gì:** thay đổi **từng chi tiết mà bài bỏ ngỏ**, xem độ chính xác thay đổi thế nào:
  - Chuẩn hóa ảnh: z-score, L2, chỉ chia 255.
  - γ: 1,0 như bài ghi, hoặc `scale` như code gốc.
  - Cách co OAS, cách đặt λ.
- **Vì sao bắt buộc:** Core scope yêu cầu ít nhất một phân tích do nhóm tự thiết kế.
- **Đã có kết quả thử sơ bộ rất đáng chú ý** (mục 3.2): γ = 1,0 như bài ghi chỉ dùng được khi chuẩn hóa L2.
- **Code:** dùng lại `scripts/run_experiment.py` với tham số `--norm`, `--gamma`, `--eval-split val`.

#### E9: so sánh với code gốc

- **Làm gì:** chạy code của tác giả, rồi so **từng bước** với code của nhóm: dữ liệu sau chuẩn hóa, γ, thống kê, trọng số, dự đoán.
- **Vì sao bắt buộc:** môn yêu cầu khi bài có công bố code. Những lỗi tìm thấy trong code gốc được đánh giá cao.
- **Quy tắc:** chỉ Huy đọc và chạy code gốc. Code gốc **không được** đưa vào repo của nhóm.
- **Code:** thư mục `verification/`.

### 2.2. Năm thực nghiệm làm thêm (E3-E7)

| Mã | Thử gì | Bài báo cho thấy |
|---|---|---|
| E3 | Bỏ phép chiếu RFF; bỏ bước khử tương quan; bỏ cả hai | Bỏ bước nào cũng giảm mạnh. Ví dụ CIFAR-10 giảm từ 55,6% xuống 28,5% khi bỏ cả hai |
| E4 | Không dùng flip | Giảm nhẹ, ví dụ CIFAR-10 còn 52,5% |
| E5 | Đổi λ giữa 10⁻⁶, 10⁻⁵, 10⁻⁴ | Mỗi dataset hợp với một λ khác nhau |
| E6 | Thay RFF bằng RP+ReLU | Kém hơn RFF, ví dụ CIFAR-10 còn 48,8% |
| E7 | Đổi D từ 1.000 đến 25.000 | D càng lớn càng tốt, ví dụ CIFAR-10 tăng 10,4 điểm |

### 2.3. Vì sao E3-E7 bị bỏ khỏi giai đoạn deck?

1. **Không bắt buộc.** Core scope chỉ cần E1, E2, cộng E8 do nhóm tự thiết kế; E9 bắt buộc vì bài có code.
2. **Không đủ thời gian.** Đến 14/11 còn 7 ngày tới hạn deck, deck cần 2-3 ngày, mà E1, E2, E8, E9 chưa có run chính thức nào. E3-E7 cộng lại khoảng 60 run nữa, cùng phân tích và slide.
3. **Dời sang poster.** Phần Empirics của poster (22/11-05/12) có các dạng "bỏ từng thành phần" và "thay đổi quy mô", rất hợp với E3 và E7. E3 giờ cũng rẻ hơn vì code E2 đã có chế độ bỏ RFF và bộ phân loại NCM.

---

## 3. Repo đang ở giai đoạn nào (14/11)?

### 3.1. Bức tranh chung

```
[Xong]  Dựng repo, W&B, Kaggle
[Xong]  Viết code: phần lõi (Đăng), dữ liệu + pipeline (Kiên), kiểm chứng (Huy)
[Xong]  Ghép toàn bộ code vào main, 54 test qua
[Đang]  Chốt cấu hình và bắt đầu chạy thực nghiệm   ← đang ở đây
[Chưa]  Phân tích sai lệch, viết deck                (18-19/11)
[Chưa]  Nộp deck                                     (20/11, hạn 21/11)
[Chưa]  Critique, Empirics, poster                   (22/11-05/12)
```

Nói gọn: **phần viết code đã xong, phần chạy thực nghiệm vừa bắt đầu.** Nhóm đang trễ khoảng một sprint so với kế hoạch ban đầu, nên lịch đã được lập lại.

### 3.2. Tiến độ từng thực nghiệm

| Mã | Code | Đã chạy | Việc tiếp theo |
|---|---|---|---|
| **E1** | Xong | Chưa có run chính thức. Thử trên laptop ở D = 1.000: MNIST 94,5%, CIFAR-10 44,4% (khớp mốc tham khảo, chưa log W&B) | **Bị chặn bởi 2 việc:** chốt chuẩn hóa và γ; xử lý bộ nhớ ở D = 25.000 |
| **E2** | Xong, đã ở `main` | Chưa chạy trên Kaggle. Đã thử trên laptop với dữ liệu và trọng số ViT thật | Kiên chạy `notebooks/02_e2_vit.ipynb` trên Kaggle |
| **E8** | Xong (dùng chung code E1) | Thử sơ bộ 6 cấu hình trên laptop, chưa log W&B (bảng dưới) | Chạy lại trên Kaggle có W&B |
| **E9** | Xong, nằm trên nhánh `feat/huy-sprint2`, **chưa merge** | So từng bước trên dữ liệu giả: dự đoán **trùng 100%** với code gốc (W&B run `icae4r9o`) | Chạy trên dữ liệu thật (MNIST, CIFAR-10); merge vào `main` |
| E3-E7 | Một phần | Không | Dời sang giai đoạn poster |

**Kết quả thử sơ bộ của E8** (laptop, CIFAR-10, D = 1.000, chưa log W&B):

| γ | Chuẩn hóa | Độ chính xác |
|---|---|---|
| 1,0 (bài ghi) | z-score | **9,8%**, ngang đoán ngẫu nhiên |
| 1,0 (bài ghi) | L2 theo từng ảnh | 45,5% |
| `scale` (code gốc) | z-score | 45,3% |

Ý nghĩa: **con số γ = 1,0 trong bài chỉ dùng được nếu ảnh được chuẩn hóa L2**, mà bài không hề nói điều này. Code gốc thì dùng `scale`. Đây là một phát hiện tốt cho phần phân tích sai lệch.

### 3.3. Ba việc đang chặn tiến độ

| Việc | Người | Hạn | Vì sao quan trọng |
|---|---|---|---|
| Chốt chuẩn hóa và γ cho E1 | Đăng, cả nhóm đồng ý | 14/11 | Mặc định hiện tại cho khoảng 10%; chạy E1 bây giờ là lãng phí GPU |
| Xử lý bộ nhớ ở D = 25.000 | Đăng | 15/11 | Có thể vượt 16 GB của GPU T4 trên Kaggle ([report2.md](report2.md), mục 3.3) |
| Chạy E2 trên Kaggle | Kiên | 15/11 | Code đã xong, chỉ còn chạy |

### 3.4. Các E dẫn tới sản phẩm nộp thế nào?

| Sản phẩm | Dùng thực nghiệm nào |
|---|---|
| **Deck** (hạn 21/11, 45% điểm) | E1, E2: bảng so sánh với bài. E8, E9: phân tích vì sao lệch |
| **Poster** (hạn 05/12, 50% điểm) | Thực nghiệm mới nhắm vào từng hạn chế của bài; có thể dùng lại E3, E5, E7, E8 |

---

## 4. Muốn đọc thêm

| Câu hỏi | Đọc ở đâu |
|---|---|
| Công thức và cấu hình chi tiết | [Instruction.md](Instruction.md) |
| Lịch từng ngày, ai làm gì | [implementation_plan.md](implementation_plan.md), mục 0 |
| Nguồn dữ liệu và giao thức của E2 | [e2_datasets.md](e2_datasets.md) |
| Môi trường Kaggle, tốc độ GPU, rủi ro bộ nhớ | [report2.md](report2.md) |
| Tình hình ngày 14/11 | [report3.md](report3.md) |
| Phát hiện của Huy về code gốc | `verification/README.md` |
