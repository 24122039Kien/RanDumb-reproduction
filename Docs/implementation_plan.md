# Kế hoạch thực hiện đồ án RanDumb

**Nhóm:** Kiên, Đăng, Huy
**Cập nhật:** 05/11/2026
**Phạm vi:** Core (E1, E2, E8, E9), làm thêm E3-E7 nếu đủ thời gian

| Sản phẩm | Hạn | Mục tiêu nộp | Điểm |
|---|---|---|---|
| Deck (bộ slide pptx) và link repo | Thứ 7, 21/11/2026 | Thứ 6, 20/11 | 45% |
| Poster (pptx một slide, A2 dọc) | Thứ 7, 05/12/2026 | Thứ 6, 04/12 | 50% |
| Repo, W&B, định dạng | Theo hai hạn trên | | 5% |

Tài liệu này chỉ nói ai làm gì và khi nào. Công thức, cấu hình và yêu cầu chi tiết nằm trong [Instruction.md](Instruction.md).

---

## 1. Tổng quan các sprint

| Sprint | Thời gian | Mục tiêu | Kết quả cuối sprint |
|---|---|---|---|
| **0. Khởi động** | 05/11 | Có repo, W&B, Kaggle | Mỗi người có commit đầu tiên |
| **1. Xây phần lõi** | 06-08/11 | Pipeline chạy từ đầu đến cuối | 6 test qua, có run đầu tiên trên W&B |
| **2. Đo và chốt cấu hình** | 09-11/11 | Biết chạy gì và tốn bao lâu | D, kiểu số, γ, chuẩn hóa, dataset E2 đã chốt |
| **3. Chạy thực nghiệm chính** | 12-15/11 | Có đủ số liệu cho deck | **Khóa kết quả** E1, E2, E8, E9 |
| **4. Hoàn thiện deck** | 16-21/11 | Phân tích và trình bày | **Nộp deck ngày 20/11** |
| **5. Critique** | 22-24/11 | Lập luận các hạn chế | 3 hạn chế có thiết kế thí nghiệm |
| **6. Empirics** | 25-29/11 | Đo bằng chứng | **Khóa kết quả**, có ít nhất 2 hạn chế được chứng minh |
| **7. Hoàn thiện poster** | 30/11-05/12 | Trình bày và luyện bảo vệ | **Nộp poster ngày 04/12** |

## 2. Phân công và quy tắc chung

| Người | Phụ trách | Sản phẩm chính |
|---|---|---|
| **Đăng** | Phần lõi và độ chính xác | `features.py`, `stats.py`, `classifier.py`, 6 test, E1, E8 |
| **Kiên** | Dữ liệu và hạ tầng chạy | `data.py`, `tracking.py`, `run_experiment.py`, đo giờ Kaggle, E2 |
| **Huy** | Kiểm chứng và deck | `verification/`, E9, khung deck, phần đọc bài |

- **Code gốc:** chỉ Huy đọc và chạy code của tác giả. Đăng viết phần lõi mà không mở code đó. Huy chỉ trao đổi về hành vi và số liệu, không chuyển code. Chép code có thể khiến cả project bị 0 điểm.
- **Commit:** mỗi người commit bằng tài khoản của mình, ít nhất mỗi ngày làm việc.
- **Họp nhanh:** 15 phút mỗi tối, gồm việc đã xong, việc đang kẹt và việc của ngày mai.
- **Số liệu:** mỗi run có seed, cấu hình và link W&B. Số không có link thì không đưa vào deck hay poster.
- **Đổi việc:** có thể đổi người theo thế mạnh, nhưng giữ quy tắc về code gốc.

## 3. Quyết định thiết kế

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Nơi chạy | Laptop để viết code và chạy thử ở D nhỏ, Kaggle T4 cho run thật | Laptop đã kiểm tra có torch bản CPU, RAM 19,7 GB, chưa có `wandb` và `datasets` |
| Nền tính toán | Viết phần lõi bằng torch, có tham số `device` | Cùng một code chạy được ở cả hai nơi |
| Sinh `ω` | Dùng `torch.Generator` trên CPU rồi chuyển sang device | Cùng seed cho cùng phép chiếu trên mọi máy |
| Kiểu số | Mặc định tạm `float64` cho `A`, chốt ở Sprint 2 | Cần so sánh độ chính xác và tốc độ thực tế trên T4 |
| OAS theo từng lớp | Xử lý lần lượt từng lớp rồi bỏ `S_c`, gộp theo số mẫu | Tránh giữ 100 ma trận D×D trên CIFAR-100 |
| Chọn γ và chuẩn hóa | Tách 10% train làm validation | Chọn trên test sẽ làm kết quả bị lạc quan |
| Quét γ và chuẩn hóa | Quét chéo, không đổi từng yếu tố riêng | γ = 1 có tác dụng rất khác giữa z-score và L2 |

**Những điểm khác với Instruction.md** (cần ghi lý do vào deck hoặc README):

| Điểm | Instruction.md | Kế hoạch này | Lý do |
|---|---|---|---|
| Thời điểm chạy E9 | Sau khi chạy thực nghiệm | Chạy E9 ở D nhỏ trước E1 | Phát hiện lỗi trước khi tốn GPU |
| Số seed của E2 | 3 seed | 1 seed, thêm một seed để xác nhận kết quả không đổi | Benchmark F không có RFF hay flip nên gần như không có yếu tố ngẫu nhiên |

---

# Giai đoạn deck

## Sprint 0: Khởi động (Thứ 5, 05/11)

**Mục tiêu:** cả nhóm sẵn sàng làm việc trên cùng một repo và W&B project.

**Cả nhóm**
- [ ] Xác nhận trên Moodle: form đã được chấp nhận, giờ nộp của hai hạn, ngày poster workshop.
- [ ] Xác minh tài khoản Kaggle và kiểm tra quota GPU thực tế.
- [ ] Đọc Mục 2 và 3 của bài báo.

**Đăng**
- [x] Repo đã có trên GitHub (`24122039Kien/RanDumb-reproduction`), có `.gitignore`.
- [x] Đã tạo khung cấu trúc thư mục: các hàm mới có chữ ký và docstring.
- [x] `randumb_benchmark.py` nằm trong `.gitignore` nên không bị commit. File này lặp lại pipeline của tác giả.
- [ ] Clone repo và commit lần đầu bằng tài khoản của mình.

**Kiên**
- [ ] Tạo W&B project public, kiểm tra mở được khi chưa đăng nhập.
- [ ] Tạo notebook Kaggle với GPU T4 x2, Internet On; lưu `WANDB_API_KEY` vào Secrets. Notebook mẫu: `notebooks/00_kaggle_smoke.ipynb`.
- [ ] Ghi lại phiên bản Python và torch trên Kaggle để ghim `requirements.txt`.
- [ ] Log một run thử với thẻ `smoke`.

**Huy**
- [ ] Tải bản PDF NeurIPS và đối chiếu các số "paper" trong Instruction.md.
- [ ] Lập khung deck theo bảng ở Sprint 4.

**Hoàn thành khi:** repo public có commit của cả ba người, W&B có run `smoke`, các mục trên Moodle đã được xác nhận.

---

## Sprint 1: Xây phần lõi (06-08/11)

**Mục tiêu:** pipeline chạy từ đầu đến cuối trên dữ liệu thật ở D nhỏ.

**Đăng**
- [ ] **06/11:** `classifier.py` gồm `oas_shrink`, `fit_lda` (Cholesky, lỗi thì dùng `lstsq`), `predict`. Viết test 4, 5.
- [ ] **07/11:** `stats.py` gồm `SufficientStats` với `update`, `finalize`, `save`, `load`. Viết test 2, 3.
- [ ] **07/11:** `features.py` gồm `RandomFourierFeatures` dạng D chiều và `RPReLU`. Viết test 1, 6.
- [ ] **08/11:** Ghép pipeline cùng Kiên, viết docstring cho các hàm chính.

**Kiên**
- [ ] **06/11:** `data.py` tải MNIST, CIFAR-10, CIFAR-100, TinyImageNet, miniImageNet, có `assert` về số mẫu và tên cột.
- [ ] **07/11:** Thêm 3 kiểu chuẩn hóa, flip và tách validation 10%.
- [ ] **07/11:** `tracking.py` đặt seed và log cấu hình, phiên bản, tên GPU.
- [ ] **07/11:** Lập bảng dữ liệu E2: nguồn, số lớp, cách chia, giấy phép, đã có sẵn hay phải tự upload.
- [ ] **08/11:** `run_experiment.py` nhận tham số dòng lệnh, log độ chính xác, thời gian và bộ nhớ đỉnh. Chạy MNIST, CIFAR-10 ở D = 1.000.

**Huy**
- [ ] **06-07/11:** Làm 4 slide phần bối cảnh của deck, kèm speaker notes. Xem hướng dẫn bên dưới.
- [ ] **07/11:** Làm 2 slide phần đóng góp, kèm speaker notes. Xem hướng dẫn bên dưới.
- [ ] **08/11:** Viết script trong `verification/` để clone code gốc vào thư mục tạm và ghim commit.

#### Hướng dẫn cho Huy: phần bối cảnh và đóng góp

**Đây là làm slide, không phải viết báo cáo Word.** Môn không có buổi thuyết trình, nên người chấm chỉ đọc file pptx. Mỗi slide gồm hai phần:

| Phần | Chứa gì | Yêu cầu của môn |
|---|---|---|
| **Mặt slide** | Bảng, sơ đồ, hình tự vẽ, vài gạch đầu dòng ngắn | Dùng hình và bảng, không viết đoạn văn dài |
| **Speaker notes** (ô ghi chú bên dưới slide trong PowerPoint) | Lời giải thích đầy đủ, như lời bạn sẽ nói nếu được thuyết trình | Slide nào cũng phải có, tối đa 500 từ |

**Sản phẩm:** khoảng 6 slide nháp trong một file pptx, kèm notes cho từng slide. Đến Sprint 4, các slide này được ghép vào deck chung 20-40 slide.

| # | Slide | Mặt slide | Speaker notes |
|---|---|---|---|
| 1 | Bài toán | Hình tự vẽ luồng dữ liệu theo thời gian, kèm ví dụ thực tế của nhóm | Online continual learning là gì; vì sao ví dụ của nhóm buộc phải học trực tuyến và không lưu mẫu |
| 2 | Giả định bị nghi ngờ | Câu hỏi: "học biểu diễn trong online continual learning có thật sự tốt hơn biểu diễn cố định không?" | Vì sao giả định này phổ biến, và vì sao theo tác giả nó chưa từng được kiểm chứng |
| 3 | Các phương pháp tiền nhiệm | Bảng 4 cột: phương pháp, ý chính, hạn chế, hệ quả | Giải thích từng phương pháp |
| 4 | Khoảng trống dẫn đến RanDumb | Sơ đồ so sánh: không cần backbone pretrained, không lưu mẫu, chỉ học một lượt | Vì sao khoảng trống này đáng quan tâm |
| 5 | Đóng góp theo lời bài báo | 4-5 gạch đầu dòng, diễn đạt lại bằng lời của nhóm | Giải thích từng đóng góp |
| 6 | Xếp loại đóng góp | Bảng 3 cột: loại, đóng góp, trước đây chưa làm được gì | Lý do xếp loại, và điều gì là mới so với các tiền nhiệm |

**Yêu cầu của môn cho hai phần này** (trang Reading):
- **Bài toán:** nêu đơn giản, kèm **một ví dụ thực tế do nhóm tự nghĩ**. Ví dụ gợi ý, nhóm nên tự chọn ví dụ khác: camera kiểm tra lỗi sản phẩm trong nhà máy, thỉnh thoảng có loại lỗi mới, thiết bị có bộ nhớ nhỏ và mỗi ảnh chỉ xử lý một lần.
- **Tiền nhiệm:** ít nhất 2 phương pháp. Với mỗi phương pháp, nêu đủ ý chính, hạn chế và **hệ quả thực tế của hạn chế**. Môn nói rõ: nêu hạn chế mà không có hệ quả thì không được tính là phân tích.
- **Đóng góp:** liệt kê theo cách bài báo nói, rồi xếp lại thành lý thuyết, kỹ thuật, thực nghiệm. Với mỗi đóng góp, nêu điều gì trước đây chưa làm được hoặc chưa ai thử.

**Gợi ý tiền nhiệm** (cần đọc lại bài gốc để kiểm tra trước khi đưa vào slide):

| Phương pháp | Ý chính | Hạn chế | Hệ quả |
|---|---|---|---|
| **SLDA** (Hayes và Kanan, 2020), bài báo ghi là "tương tự" | Giữ nguyên một CNN đã huấn luyện sẵn; chỉ cập nhật trung bình lớp và hiệp phương sai chung theo từng mẫu | Phụ thuộc hoàn toàn vào bộ trích đặc trưng đã huấn luyện sẵn | Không dùng được khi phải học từ pixel; LDA trên pixel thô thì kém (Bảng 6 của bài: CIFAR-10 đạt 41,6 so với 55,6 của RanDumb) |
| **GDumb** (Prabhu và cộng sự, 2020), cùng tác giả chính | Lưu bộ nhớ mẫu cân bằng giữa các lớp, rồi huấn luyện lại mạng từ đầu trên bộ nhớ đó; một baseline đơn giản nhưng vượt nhiều phương pháp phức tạp | Phải lưu mẫu và huấn luyện lại mạng sâu | Không dùng được khi bị cấm lưu dữ liệu hoặc bộ nhớ rất nhỏ; tốn tính toán |
| **RanPAC** (McDonnell và cộng sự, 2023), không bắt buộc | Chiếu ngẫu nhiên đặc trưng của mô hình pretrained lên nhiều chiều, rồi dùng bộ phân loại tuyến tính | Cần backbone pretrained mạnh | Không áp dụng được cho học từ pixel |

**Gợi ý xếp loại đóng góp** (theo trang 1-3 của bài):

| Loại | Đóng góp | Trước đây chưa làm được gì |
|---|---|---|
| **Lý thuyết** | Gần như không có: không có định lý mới, chỉ có trực giác về kernel và khử tương quan (Hình 2) | Nên nói thẳng điều này; đây cũng là ý cho phần Critique sau này |
| **Kỹ thuật** | Ghép các khối có sẵn (RFF, LDA dạng streaming, NCM) thành một phương pháp học từ pixel, không lưu mẫu, một lượt | SLDA cần backbone đã huấn luyện sẵn; GDumb cần lưu mẫu |
| **Thực nghiệm** | So sánh trên benchmark A-F; hai phát hiện: biểu diễn học được kém hơn biểu diễn ngẫu nhiên, và benchmark bị ràng buộc quá chặt | Chưa ai kiểm chứng giả định "biểu diễn học được tốt hơn biểu diễn cố định" trong online continual learning |

Một chi tiết nên nêu: Bảng 1 trái cho thấy RanDumb **thua** PEC 3,3 điểm trên CIFAR-10, nên câu "vượt mọi phương pháp" không đúng ở mọi dataset.

**Lưu ý khi làm slide:**
- **Thứ tự đọc bài:** tóm tắt, mở đầu, kết luận, các công trình liên quan, thực nghiệm, phương pháp đọc sau cùng.
- **Trích dẫn:** theo kiểu IEEE, đánh số, đặt ở cuối chính slide đó. Không có slide References riêng.
- **Hình:** tất cả phải do nhóm tự vẽ, không lấy Hình 1 và Hình 2 của bài.
- **Lời văn:** viết lại bằng lời của nhóm, không dịch nguyên câu của bài.

**Coi như xong khi:** có 6 slide nháp kèm notes, bảng tiền nhiệm đủ 3 cột ý chính, hạn chế, hệ quả, và mỗi đóng góp có dòng "trước đây chưa làm được gì".

**Lưu ý khi viết test**
- Test 2 và 3 cần có dung sai số thực, vì đổi thứ tự cộng sẽ không cho kết quả giống hệt từng bit.
- Test 4 chạy trên dữ liệu một lớp, vì `sklearn.covariance.OAS` trừ trung bình toàn cục còn `S` của nhóm là hiệp phương sai trong lớp.
- Công thức OAS phải xử lý trường hợp mẫu số bằng 0.

**Mốc tham khảo ở D = 1.000**, suy từ số trong Instruction.md, không phải tiêu chí đạt: MNIST khoảng 94,7, CIFAR-10 khoảng 45,2.

**Hoàn thành khi:** 6 test đều qua và có run MNIST, CIFAR-10 trên W&B.

**Nếu trễ:** Kiên chuyển sang hỗ trợ phần lõi, tạm dừng bảng dữ liệu E2.

---

## Sprint 2: Đo và chốt cấu hình (09-11/11)

**Mục tiêu:** chốt mọi cấu hình trước khi tốn GPU cho run thật.

**Kiên**
- [ ] **09/11:** Đo CIFAR-10 trên T4 với D = 1.000, 2.000, 5.000, so `float32` với `float64`, rồi ngoại suy lên D = 25.000.
- [ ] **09/11:** Chốt danh sách dataset E2, tự upload bộ còn thiếu nếu giấy phép cho phép.
- [ ] **10-11/11:** Viết `extract_vit_features.py`, chạy thử trên CIFAR với cả hai backbone, lưu đặc trưng thành artifact.
- [ ] **11/11:** Log tên mô hình và phiên bản `timm`; thử có và không chuẩn hóa mean/std.

**Đăng**
- [ ] **09/11:** Cùng Kiên chốt D, kiểu số và cách chia phiên chạy.
- [ ] **10/11:** Chạy E8 pha A: 3 kiểu chuẩn hóa × 4 giá trị γ (1,0; `scale` × 0,25; `scale`; `scale` × 4) trên CIFAR-10, D = 2.000, chấm trên validation.
- [ ] **11/11:** Kiểm tra lại 2 cấu hình tốt nhất trên CIFAR-100, rồi chốt cấu hình mặc định kèm lý do.
- [ ] **11/11:** Thêm các chế độ cho E8 pha B: OAS theo từng lớp, không OAS, λ tỉ lệ `trace(S)/D`.

**Huy**
- [ ] **09/11:** Viết phần phương pháp của deck: ký hiệu, bài toán hình thức, sơ đồ pipeline tự vẽ, từng thành phần.
- [ ] **10/11:** Chạy nguyên bản code gốc trên MNIST, CIFAR-10 ở D nhỏ, log thẻ `official-code`.
- [ ] **11/11:** So từng tầng với Đăng: dữ liệu sau chuẩn hóa, γ, trung bình lớp và `S`, độ chính xác từng lớp.
- [ ] **11/11:** Ghi các khác biệt và lỗi phát hiện trong code gốc, gồm `store_precision` và `get_feats_vit.py`.

**Hoàn thành khi:**
- Đã chốt D, kiểu số, γ, chuẩn hóa và danh sách dataset E2.
- Khác biệt với code gốc đã giải thích được.

**Nếu trễ:**
- Thiếu tài nguyên thì dùng D = 15.000 và ghi là sai lệch.
- Chưa giải thích hết khác biệt thì lùi tối đa 1 ngày, sau đó chạy E1 và ghi phần chưa giải thích vào phân tích sai lệch.

---

## Sprint 3: Chạy thực nghiệm chính (12-15/11)

**Mục tiêu:** có đủ số liệu cho deck trước tối 15/11.

**Đăng**
- [ ] **12/11:** E1 trên MNIST, CIFAR-10, 3 seed.
- [ ] **13/11:** E1 trên CIFAR-100, miniImageNet, 3 seed.
- [ ] **14/11:** E1 trên TinyImageNet, 3 seed. Điền bảng paper và team kèm sai lệch %.
- [ ] **14-15/11:** E8 pha B: shrink và λ trên CIFAR-10/100, ở D = 5.000 và một D lớn.

**Kiên**
- [ ] **12-13/11:** Trích đặc trưng ViT cho CUB, IN-R, IN-A, VTAB với cả hai backbone.
- [ ] **13-15/11:** Chạy RanDumb và NCM trên 5 dataset × 2 backbone (20 run).
- [ ] **15/11:** Chạy lại một cấu hình với seed khác để xác nhận kết quả không đổi. Điền bảng E2.

**Huy**
- [ ] **12-13/11:** E9 đầy đủ: độ chính xác từng lớp, kiểm tra tuyên bố lệch ±0,8% của tác giả.
- [ ] **14/11:** Nếu cần D lớn, tạo bản sao kiểm chứng với `store_precision=False` và ghi rõ đã sửa gì.
- [ ] **14-15/11:** Viết phần phương pháp đánh giá của deck: dataset, metric, giao thức.

**Làm thêm (chỉ khi E1 và E2 xong trước 14/11):** E3, rồi E5, vì hai thực nghiệm này rẻ.

**Hoàn thành khi (tối 15/11):** E1, E2, E8, E9 đã có trên W&B. Sau mốc này không chạy thêm thí nghiệm mới cho deck.

**Nếu trễ, cắt giảm theo thứ tự:**
1. Bỏ E3-E7.
2. Giảm D từ 25.000 xuống 15.000 và ghi là sai lệch.
3. Giảm E2: bỏ VTAB, rồi IN-A, nhưng vẫn giữ cả hai backbone.
4. Giữ nguyên E1, E8, E9.

---

## Sprint 4: Hoàn thiện deck (16-21/11)

**Mục tiêu:** biến số liệu thành deck hoàn chỉnh và nộp sớm một ngày.

**Đăng**
- [ ] **16-17/11:** Phân tích sai lệch E1 và E8, trả lời đủ 4 câu của Bước 6.
- [ ] **18/11:** Vẽ các hình kết quả, viết slide và speaker notes phần E1, E8.

**Kiên**
- [ ] **16-17/11:** Phân tích sai lệch E2, viết slide và speaker notes phần E2.
- [ ] **17/11:** Hoàn thiện README: cài đặt, cách chạy, thư viện ngoài, seed, trích dẫn, bảng mapping.
- [ ] **18/11:** Chạy lại từ đầu trên môi trường sạch, chỉ theo README. Sửa những chỗ không chạy được.

**Huy**
- [ ] **16-17/11:** Phân tích E9, ghép bảng paper và team, nêu các điểm bài báo không nhất quán.
- [ ] **18/11:** Ghép bản nháp đầy đủ, kiểm tra ký hiệu và trích dẫn thống nhất.

**Cả nhóm**
- [ ] **19/11:** Mỗi người trình bày lại phần của một người khác. Rà soát danh sách kiểm tra bên dưới.
- [ ] **20/11:** **Nộp deck và link repo trên Moodle.**
- [ ] **21/11:** Dự phòng, không xếp việc.

**Bố cục deck đề xuất (khoảng 33 slide):**

| Phần | Số slide | Người phụ trách |
|---|---|---|
| Tiêu đề | 1 | Huy |
| Bối cảnh, ví dụ thực tế, các tiền nhiệm | 4 | Huy |
| Đóng góp | 2 | Huy |
| Phương pháp, tính chất lý thuyết, độ phức tạp | 7 | Huy, Đăng rà soát |
| Phương pháp đánh giá | 2 | Huy |
| Lựa chọn scope và lý do | 1 | Cả nhóm |
| E1: kết quả và so sánh | 3 | Đăng |
| E2: kết quả và so sánh | 2 | Kiên |
| E8: phân tích độ nhạy | 3 | Đăng |
| E9: so sánh với code gốc | 2 | Huy |
| Phân tích sai lệch | 3 | Cả nhóm |
| Tổng kết | 1 | Huy |
| Phụ lục | 2 | Theo nội dung |

**Kiểm tra trước khi nộp deck:**
- [ ] File pptx, 20-40 slide, dưới 20 MB.
- [ ] Mỗi slide có speaker notes, tối đa 500 từ.
- [ ] Trích dẫn IEEE ở cuối chính slide đó, không có slide References riêng.
- [ ] Mọi hình do nhóm tự vẽ, không dán code thô, không dùng nguyên câu trong bài.
- [ ] Mỗi con số của nhóm có link tới đúng W&B run.
- [ ] Có lựa chọn scope kèm lý do, bảng paper và team với sai lệch %, phân tích sai lệch, so sánh với code gốc.
- [ ] README có bảng mapping và chạy lại được từ đầu.
- [ ] Không có token hay code của tác giả trong repo.

---

# Giai đoạn poster

Critique chỉ phê bình lý thuyết, còn Empirics đo bằng chứng cho từng hạn chế. Mọi run dùng cùng W&B project, gắn thẻ `lab:empirics`.

**Hạn chế đề xuất** (giả thuyết, chưa được kiểm chứng):

| Người | Hạn chế | Thí nghiệm | Bằng chứng cần có |
|---|---|---|---|
| **Kiên** | Thống kê tích lũy không có cơ chế quên nên có thể tệ khi có concept drift | Cho phân bố của lớp đổi dần trong luồng | Độ chính xác theo mức drift |
| **Huy** | RFF trên pixel không bất biến với dịch chuyển ảnh | Dịch, xoay, đổi sáng ảnh test | Đường suy giảm theo mức biến đổi |
| **Đăng** | λ cố định tương tác với D | Dùng lại E8, thêm các điểm D và λ tỉ lệ `trace(S)/D` | Độ chính xác theo D với hai cách đặt λ |

## Sprint 5: Critique (22-24/11)

**Mục tiêu:** mỗi hạn chế đủ rõ để thiết kế được thí nghiệm.

**Mỗi người, với hạn chế của mình**
- [ ] **22-23/11:** Viết lập luận lý thuyết: giả định nào của bài, vì sao có thể bị vi phạm.
- [ ] **23/11:** Viết thiết kế thí nghiệm: biến thay đổi, metric, mô hình so sánh, kết quả dự đoán.
- [ ] **24/11:** Đề xuất một hướng phát triển gắn với hạn chế đó.

**Cả nhóm**
- [ ] **24/11:** Thống nhất 3 điểm mạnh có bằng chứng và ít nhất 2 hướng phát triển.
- [ ] **24/11:** Rà soát chéo thiết kế thí nghiệm của nhau.

**Hoàn thành khi:** mỗi hạn chế có lập luận, metric và dự đoán kết quả.

**Nếu trễ:** bỏ hạn chế yếu nhất, tập trung vào 2 hạn chế còn lại.

---

## Sprint 6: Empirics (25-29/11)

**Mục tiêu:** đo được bằng chứng cho ít nhất 2 hạn chế.

**Kiên: concept drift**
- [ ] **25-26/11:** Viết bộ tạo luồng dữ liệu có drift, tăng dần theo mức.
- [ ] **27-28/11:** Chạy RanDumb và một mô hình so sánh có cơ chế quên, nhiều mức drift, nhiều seed.
- [ ] **29/11:** Vẽ đường độ chính xác theo mức drift.

**Huy: dịch chuyển ảnh**
- [ ] **25-26/11:** Viết các phép dịch, xoay, đổi sáng cho ảnh test, mỗi phép nhiều mức.
- [ ] **27-28/11:** Chạy trên mô hình đã huấn luyện, nhiều seed.
- [ ] **29/11:** Vẽ đường suy giảm theo mức biến đổi.

**Đăng: λ và D**
- [ ] **25-26/11:** Tổng hợp kết quả E8, xác định các điểm D và λ còn thiếu.
- [ ] **27-28/11:** Chạy các điểm còn thiếu, so λ cố định với λ tỉ lệ `trace(S)/D`.
- [ ] **29/11:** Vẽ độ chính xác theo D với hai cách đặt λ.

**Hoàn thành khi (tối 29/11):** có ít nhất 2 hạn chế được chứng minh, mọi run gắn thẻ `lab:empirics`.

**Nếu trễ:** thay bằng thí nghiệm scaling `D²` và `D³`, chỉ cần đo giờ thật.

---

## Sprint 7: Hoàn thiện poster (30/11-05/12)

**Mục tiêu:** có poster hoàn chỉnh, và cả ba người bảo vệ được mọi phần.

**Cả nhóm**
- [ ] **30/11:** Chọn 2 hạn chế rõ nhất cho poster. Đưa hạn chế còn lại vào README kèm bằng chứng.
- [ ] **03/12:** Rà soát chéo, kiểm tra link W&B, luyện trả lời câu hỏi.
- [ ] **04/12:** **Nộp poster trên Moodle.**
- [ ] **05/12:** Dự phòng, không xếp việc.

**Huy**
- [ ] **30/11-01/12:** Dựng bố cục poster A2 dọc, đặt lập luận ngay cạnh bằng chứng của từng hạn chế.
- [ ] **02/12:** Ghép bản hoàn chỉnh, kiểm tra độ dễ đọc khi in.

**Đăng và Kiên**
- [ ] **30/11-01/12:** Vẽ lại hình cho poster và viết phần của hạn chế mình phụ trách.
- [ ] **02/12:** Hoàn thiện README phần Empirics và cập nhật bảng mapping.

**Kiểm tra trước khi nộp poster:**
- [ ] File pptx một slide, khổ A2 dọc (420 × 594 mm).
- [ ] Mỗi hạn chế có lập luận lý thuyết đặt ngay cạnh bằng chứng đo được.
- [ ] Mỗi con số thực nghiệm dẫn tới W&B run.
- [ ] Hạn chế không lên poster nằm trong README kèm bằng chứng.
- [ ] Mỗi thành viên trả lời được câu hỏi về mọi phần.

---

# Rủi ro và giả định

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Không tái lập được độ chính xác | Lệch nhiều so với code gốc ở D nhỏ | So từng tầng, bắt đầu từ chuẩn hóa và γ |
| Thiếu dữ liệu đúng cách chia cho E2 | Bảng dữ liệu ở Sprint 1 | Tự upload nếu giấy phép cho phép; nếu không, giảm dataset và ghi rõ |
| Thiếu quota GPU | Số đo ở Sprint 2 | Dùng quota của tài khoản khác, rồi cắt giảm theo thứ tự ở Sprint 3 |
| `float64` quá chậm trên T4 | Số đo ở Sprint 2 | Chuyển sang `float32` nếu độ chính xác không đổi, và ghi là sai lệch |
| miniImageNet khác bản gốc | Lệch riêng ở dataset này | Ghi là "dataset tương đương" và phân tích sai lệch |
| Có thành viên không giải thích được phần việc | Lộ ra khi rà soát chéo | Mỗi người trình bày lại phần của người khác vào 19/11 và 03/12 |

**Chưa kiểm chứng:**
- Thời gian chạy và số run của E8 pha B, sẽ cập nhật sau Sprint 2.
- Thông số Kaggle (gồm 30 giờ GPU mỗi tuần) trên tài khoản của nhóm.
- Tốc độ `float64` trên T4.
- Dataset nào của E2 có sẵn đúng cách chia của LAMDA-PILOT.
- Giờ nộp, ngày poster workshop và trạng thái form đăng ký trên Moodle.
