# Hướng dẫn Thiết kế Khung Slide Deck – Dành cho AI Model

> **Người viết:** Phan Gia Huy (huyphan0806) – Nhóm 24TNT  
> **Mục đích:** Cung cấp bản hướng dẫn đầy đủ để một AI model khác tạo khung sườn slide deck `.pptx` cho bài thuyết trình tái lập bài báo *RanDumb: Random Representations Outperform Online Continually Learned Representations* (NeurIPS 2024), dựa trên yêu cầu từ file `ML - HKD.pdf`.  
> **Lưu ý:** Model nhận file này sẽ **chỉ có** `ML - HKD.pdf` và hướng dẫn này. Mọi số liệu thực nghiệm của nhóm sẽ được điền vào sau thông qua các marker `[INSERT: ...]`.

---

## PHẦN 0 – BỐI CẢNH DỰ ÁN (CONTEXT)

### 0.1. Bài báo gốc

**Tiêu đề:** *"RanDumb: Random Representations Outperform Online Continually Learned Representations"*  
**Tác giả:** Ameya Prabhu, Hasan Abed Al Kader Hammoud, Ser-Nam Lim, Bernard Ghanem, Philip H. S. Torr, Adel Bibi  
**Hội nghị:** NeurIPS 2024, vol. 37  
**Luận điểm cốt lõi:** Trong bài toán Online Continual Learning (học phân loại ảnh từ luồng dữ liệu liên tục, chỉ xem mỗi mẫu 1 lần, không lưu trữ dữ liệu cũ), các biểu diễn đặc trưng được học bởi mạng Deep Learning hiện đại thực chất **kém hiệu quả hơn** một phép chiếu ngẫu nhiên cố định được thiết kế bài bản.

**Phương pháp RanDumb gồm 2 phần:**
1. **Random Fourier Features (RFF):** Chiếu ảnh pixel lên không gian chiều cao (D = 25,000 chiều) xấp xỉ RBF Kernel, cố định vĩnh viễn, không học.
2. **Online LDA/Mahalanobis Classifier:** Cập nhật trực tuyến vector trung bình từng lớp và ma trận hiệp phương sai dùng chung (co hệ số OAS – Oracle Approximating Shrinkage). Phân loại bằng khoảng cách Mahalanobis cực tiểu.

### 0.2. Nhóm thực hiện tái lập

| Thành viên | MSSV / GitHub | Vai trò chính |
|---|---|---|
| Phan Gia Huy | huyphan0806 | Kiểm chứng code tác giả (E9), E8-B sensitivity, Slide deck |
| Trần Minh Kiên | 24122039Kien | Thực nghiệm E2 (ViT features), E8-A |
| Thành viên Đăng | — | Core code (`src/randumb/`), E1 pixel experiments |

**Repo:** `https://github.com/24122039Kien/RanDumb-reproduction` (branch: `main`)  
**W&B Project:** `randumb-reproduction-24TNT`  
**Entity:** `24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm`

### 0.3. Các thực nghiệm đã thực hiện

| Mã | Nội dung | Người thực hiện | Trạng thái |
|---|---|---|---|
| **E1** | Tái lập Benchmark A (Pixel-based, học từ scratch): MNIST, CIFAR-10, CIFAR-100, TinyImageNet, miniImageNet | Đăng | Hoàn thành |
| **E2** | Tái lập Benchmark F (ViT-B/16 features): CIFAR-100, CUB-200, ImageNet-R, ImageNet-A | Kiên | Hoàn thành (16 runs) |
| **E8-A/B** | Phân tích độ nhạy: OAS shrinkage (per-class vs pooled vs none) × λ mode (fixed vs relative) × D (5K, 25K) × CIFAR-10/100 × seed 0–2 | Huy & Kiên | Hoàn thành (24+ runs) |
| **E9** | Kiểm chứng từng tầng toán học: so sánh số học giữa code tự viết và code tác giả (commit `14a51ee`) | Huy | Hoàn thành |

> **Lưu ý cho model thiết kế:** E3–E7 (ablation study đầy đủ) được dời sang giai đoạn Poster (05/12/2026), **không xuất hiện trong slide deck lần này**.

---

## PHẦN 1 – YÊU CẦU KỸ THUẬT TUYỆT ĐỐI (HARD CONSTRAINTS)

Các yêu cầu này đến từ `ML - HKD.pdf` – tài liệu hướng dẫn chấm điểm của môn học. Model **bắt buộc** tuân thủ:

1. **Định dạng file:** `.pptx` (PowerPoint), kích thước < 20 MB.
2. **Số slide:** Tối thiểu 20 slide, tối đa 40 slide. Nhóm nhắm 28–32 slide.
3. **Cấu trúc mỗi slide – 2 phần tách biệt hoàn toàn:**
   - **Mặt slide (Face):** Trực quan, cô đọng. Dùng sơ đồ, bảng biểu, công thức, từ khóa. **TUYỆT ĐỐI KHÔNG viết đoạn văn dài trên mặt slide.**
   - **Speaker Notes (Ghi chú thuyết trình):** Lời giảng giải đầy đủ như khi đứng trước hội đồng. **Tối đa 500 từ/slide.**
4. **Trích dẫn IEEE:** Đánh số theo thứ tự xuất hiện `[1], [2], ...`. Đặt ở **cuối chính slide đó** dưới dạng văn bản nhỏ. **Không có slide References riêng biệt.**
5. **Hình ảnh/Sơ đồ:** 100% do nhóm tự vẽ. **Tuyệt đối không dùng hình chụp màn hình từ bài báo gốc.**
6. **Số liệu thực nghiệm:** Mọi con số phải có link W&B run công khai kèm theo. Nếu chưa có, dùng marker `[INSERT: ...]`.
7. **Ngôn ngữ trình bày:** Tiếng Việt là chính. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh.

---

## PHẦN 2 – CẤU TRÚC SLIDE THEO TỪNG PHẦN (SECTION BREAKDOWN)

### SECTION A – MỞ ĐẦU & BỐI CẢNH (Slide 1–2)

---

#### SLIDE 1: TRANG BÌA (TITLE SLIDE)

**Mặt slide:**
- **Tiêu đề chính:** Tái lập bài báo: *"RanDumb: Random Representations Outperform Online Continually Learned Representations"*
- **Hội nghị:** NeurIPS 2024
- **Nhóm:** 24TNT – Môn Nhập môn Học máy
- **Thành viên:** Phan Gia Huy, Trần Minh Kiên, [Thành viên Đăng]
- **Repo:** `https://github.com/24122039Kien/RanDumb-reproduction`
- **Logo trường** (nếu có template)

**Speaker Notes (< 100 từ):**
> Giới thiệu ngắn gọn: Nhóm 24TNT thực hiện tái lập bài báo RanDumb của Ameya Prabhu và cộng sự, công bố tại NeurIPS 2024. Bài báo thách thức một giả định nền tảng trong học máy: liệu mạng sâu có thực sự học được biểu diễn tốt khi chỉ nhìn dữ liệu một lần?

**IEEE Citations cuối slide:** *(không cần trích dẫn trang bìa)*

---

#### SLIDE 2: BÀI TOÁN HỌC LIÊN TỤC TRỰC TUYẾN

**Mặt slide:**
- **Tiêu đề:** Học liên tục trực tuyến trong điều kiện biên khắt khe
- **[CHÈN HÌNH/SƠ ĐỒ 1 – Horizontal Flowchart]:** Sơ đồ luồng dữ liệu Online Continual Learning (xem đặc tả tại Mục 5.4).
  ```
  Luồng dữ liệu: [x1,y1] --> [x2,y2] --> ... --> [xt,yt] --> ...
                       |            |                   |
           +--------------------------------------------------+
           |  Mô hình xử lý TRỰC TUYẾN (Online Processing)   |
           |  - Xem mỗi mẫu DUY NHẤT 1 lần                   |
           |  - Không lưu dữ liệu cũ (Zero Replay)            |
           |  - Lớp xuất hiện tuần tự (Class-Incremental)     |
           +--------------------------------------------------+
  ```
- **3 ràng buộc ngặt nghèo (highlight box):**
  1. Single-Pass: Mỗi mẫu chỉ 1 lần
  2. Zero Replay: Không lưu mẫu cũ
  3. Constant Memory: RAM cố định ∝ O(D^2 + C·D)
- **Ví dụ thực tế:** Camera kiểm tra bo mạch điện tử PCB trên băng chuyền 24/7

**Speaker Notes (< 500 từ):** Giải thích bài toán Online Continual Learning, 3 ràng buộc, hiện tượng Catastrophic Forgetting, và ví dụ thực tế Edge AI. Nhấn mạnh sự cấp thiết của việc giải quyết bài toán này mà không lưu trữ dữ liệu.

**IEEE Citations:** `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," NeurIPS, vol. 37, 2024.`

---

### SECTION B – PHÂN TÍCH BÀI BÁO (Slide 3–8)

---

#### SLIDE 3: GIẢ ĐỊNH BỊ NGHI NGỜ (QUESTIONED ASSUMPTION)

**Mặt slide:**
- **Tiêu đề:** Giả định trung tâm bị thách thức
- **Câu hỏi trung tâm (hộp lớn, nổi bật):**
  > *"Trong Online Continual Learning, biểu diễn học được từ mạng sâu có thực sự tốt hơn một biểu diễn ngẫu nhiên cố định không hề được học?"*
- **[CHÈN HÌNH/SƠ ĐỒ 2 – Comparison Diagram]:** Sơ đồ đối đầu 2 trường phái Representation (xem Mục 5.4):
  
  | Trường phái truyền thống | RanDumb (Phát hiện mới) |
  |---|---|
  | Ảnh → Mạng sâu (backprop) → Classifier | Ảnh → Phép chiếu ngẫu nhiên cố định → Classifier |
  | Tin: mạng ngày càng tốt | Thực tế: biểu diễn ngẫu nhiên ổn định hơn |
  | Bị quên thảm khốc | Miễn nhiễm với forgetting |
- **Nghịch lý:** Càng cố gắng học → Biểu diễn càng bị méo mó (Representation Drift)

**Speaker Notes (< 500 từ):** Giải thích tiên đề bị thách thức, tại sao backprop thất bại trong single-pass, và cơ chế representation drift.

**IEEE Citations:** `[1]`

---

#### SLIDE 4: CÁC PHƯƠNG PHÁP TIỀN NHIỆM

**Mặt slide:**
- **Tiêu đề:** Phân tích phương pháp tiền nhiệm – Khoảng trống nghiên cứu
- **Bảng 4 cột bắt buộc (yêu cầu môn học):**

| Phương pháp | Ý tưởng chính | Hạn chế cốt lõi | Hệ quả thực tế |
|---|---|---|---|
| **SLDA** (Hayes & Kanan, 2020) [2] | CNN pretrained cố định + cập nhật LDA trực tuyến | Phụ thuộc backbone pretrained | Vô dụng khi học từ pixel thô (41.6% vs 55.6%) |
| **GDumb** (Prabhu et al., 2020) [3] | Buffer lấy mẫu cân bằng + retrain từ đầu | Lưu trữ mẫu + chi phí retrain cực lớn | Vi phạm ràng buộc bảo mật & RAM nhỏ |
| **RanPAC** (McDonnell et al., 2023) [4] | Random Projection trên feature ViT/CLIP | Phụ thuộc Foundation Model khổng lồ | Không giải quyết từ scratch |

**Speaker Notes (< 500 từ):** Phân tích chi tiết từng phương pháp, hệ quả thực tế, và tóm lại tại sao cả 3 đều không giải quyết được ô "học từ pixel, không lưu mẫu".

**IEEE Citations:**  
`[2] T. L. Hayes and C. Kanan, "Lifelong machine learning with deep streaming linear discriminant analysis," CVPRW, 2020.`  
`[3] A. Prabhu et al., "GDumb: A Simple Approach that Questions Continual Learning," ECCV, 2020.`  
`[4] M. D. McDonnell et al., "RanPAC: Random Projections and Pre-trained Representations for Continual Learning," NeurIPS, 2023.`

---

#### SLIDE 5: KHOẢNG TRỐNG NGHIÊN CỨU & VỊ TRÍ CỦA RANDUMB

**Mặt slide:**
- **Tiêu đề:** Khoảng trống nghiên cứu (Research Gap)
- **[CHÈN HÌNH/SƠ ĐỒ 3 – 2x2 Grid Matrix]:** Ma trận định vị 2×2 Research Gap (xem đặc tả tại Mục 5.4).
  ```
                    CO LUU MAU        KHONG LUU MAU
  DUA TREN      BiC, DER, ER-AML  |  SLDA, RanPAC
  PRETRAINED                       |
  ---------------------------------+------------------
  TU HOC TU    GDumb, replay CL   |  * RANDUMB
  PIXEL THO                        |  Khong trong bi bo quen!
  ```
- **3 tiêu chí vàng RanDumb chinh phục đồng thời:**
  - Zero Replay Memory
  - Single-Pass Streaming  
  - No Pre-trained Dependency

**Speaker Notes (< 500 từ):** Dùng ma trận 2×2 để chỉ rõ góc "học từ scratch, không lưu mẫu" trước đây bị bỏ trống. RanDumb là công trình đầu tiên lấp đầy khoảng trống này.

**IEEE Citations:** `[1]`

---

#### SLIDE 6: ĐÓNG GÓP CỐT LÕI CỦA BÀI BÁO

**Mặt slide:**
- **Tiêu đề:** 4 Đóng góp khoa học của RanDumb
- **4 bullet điểm lớn:**
  1. Đề xuất RanDumb – baseline đơn giản nhưng vượt trội tất cả SotA trên 5 datasets
  2. Bác bỏ giả định về hiệu quả của Representation Learning trong OCL
  3. Phơi bày thiên lệch trong benchmark design hiện hành
  4. Mở rộng thành công lên Foundation Models (ViT) – đánh bại Prompt-tuning

**Speaker Notes (< 500 từ):** Trình bày từng đóng góp theo lời bài báo. Nhấn mạnh đây là công trình thực nghiệm, không phải lý thuyết – đóng góp lớn nhất nằm ở việc đặt lại câu hỏi nền tảng.

**IEEE Citations:** `[1]`

---

#### SLIDE 7: PHÂN LOẠI ĐÓNG GÓP & ĐIỂM CHƯA LÀM ĐƯỢC

**Mặt slide:**
- **Tiêu đề:** Phân loại đóng góp theo 3 nhánh học thuật
- **Bảng 3 cột bắt buộc (yêu cầu môn học):**

| Phân loại | Đóng góp cụ thể | Điều trước đây chưa làm được (Novelty) |
|---|---|---|
| **Lý thuyết** | Gần như không có định lý mới. Dựa hoàn toàn trên RFF (Rahimi & Recht 2007) và LDA/OAS | Lần đầu tiên liên hệ RFF với khả năng miễn nhiễm forgetting trong OCL |
| **Kỹ thuật** | Ghép nối 3 khối: RFF → Streaming covariance → OAS Cholesky | Pipeline đầu tiên chạy trực tiếp từ pixel thô mà không cần pretrained |
| **Thực nghiệm** | Thực nghiệm trên 6 benchmark (A–F), lật đổ giả định trung tâm | Lần đầu tiên kiểm chứng giả định này bằng số liệu thực nghiệm |

- **Phát hiện phản biện của nhóm (Critical Finding – highlight đỏ):**
  > CẢNH BÁO: CIFAR-10: RanDumb 55.6% < PEC 58.9% (thua 3.3%) → Tuyên bố "vượt trội tất cả" không tuyệt đối!

**Speaker Notes (< 500 từ):** Phân tích 3 phân loại một cách thẳng thắn. Đặc biệt nhấn mạnh phát hiện phản biện: RanDumb không phải silver bullet, còn thua PEC trên CIFAR-10.

**IEEE Citations:** `[1]` `[6] A. Rahimi and B. Recht, "Random features for large-scale kernel machines," NeurIPS, vol. 20, 2007.`

---

#### SLIDE 8: ĐỊNH NGHĨA TOÁN HỌC BÀI TOÁN (PROBLEM FORMULATION)

**Mặt slide:**
- **Tiêu đề:** Hình thức hóa toán học – Online Continual Learning
- **[CHÈN HÌNH/SƠ ĐỒ 4 – Timeline Diagram]:** Sơ đồ kịch bản Class-Incremental Streaming qua $K$ tasks (xem Mục 5.4).
- **Ký hiệu:**
  - Luồng dữ liệu: S = {(x_t, y_t)}_{t=1}^N, x_t trong R^d, y_t trong {1,...,C}
  - Xuất hiện tuần tự K giai đoạn: Y = union_{k=1}^K Y_k, Y_i giao Y_j = rỗng
- **3 ràng buộc toán học:**
  1. Single-pass: Mô hình xem (x_t, y_t) đúng 1 lần
  2. Zero buffer: M = rỗng (không lưu x_tau, tau < t)
  3. Constant memory: Space = O(D^2 + C*D) << O(N*d)
- **Mục tiêu (công thức):** maximize (1/t) * sum_{i=1}^t I[f_t(x_i) = y_i]

**Speaker Notes (< 500 từ):** Trình bày bài toán một cách chặt chẽ toán học. 3 ràng buộc là "luật chơi" bắt buộc. Mục tiêu là phân loại chính xác tích lũy.

**IEEE Citations:** `[1]` `[5] Z. Mai et al., "Online continual learning in image classification: An empirical survey," Neurocomputing, vol. 469, pp. 28-51, 2022.`

---

### SECTION C – PHƯƠNG PHÁP RANDUMB (Slide 9–13)

---

#### SLIDE 9: KIẾN TRÚC TỔNG THỂ PIPELINE RANDUMB

**Mặt slide:**
- **Tiêu đề:** Kiến trúc giải tích 4 khối của RanDumb
- **[CHÈN HÌNH/SƠ ĐỒ 5 – Vertical Flowchart]:** Sơ đồ Kiến trúc Pipeline 4 khối RanDumb (xem Mục 5.4).
  ```
  Anh dau vao x_t trong R^d
         |
  +----------------------------------+
  |  Khoi 1: Chuan hoa (Z-score)     |
  |  x_t_norm = (x_t - mu) / sigma  |
  +----------------------------------+
                     |
  +----------------------------------+
  |  Khoi 2: RFF Embedding (Co dinh)|
  |  Phi(x) = sqrt(2/D) cos(Omega*x)|
  |  Omega ~ N(0,2*gamma*I), b~U     |
  +----------------------------------+
                     |
  +----------------------------------+
  |  Khoi 3: Tich luy thong ke OAS  |
  |  Cap nhat n_c, mu_c, A = sum PhiPhi^T  |
  |  Sigma_OAS = (1-rho)S + rho(TrS/D)I   |
  +----------------------------------+
                     |
  +----------------------------------+
  |  Khoi 4: LDA Classifier         |
  |  Cholesky: Sigma_OAS * W = M    |
  |  y* = argmax_c [W_c^T * Phi(x)] |
  +----------------------------------+
  ```
- **Đặc điểm then chốt:**
  - 100% giai tich – KHONG Backpropagation
  - Bieu dien co dinh | Classifier truc tuyen

**Speaker Notes (< 500 từ):** Mô tả tổng quan 4 khối, nhấn mạnh tính phi tham số học (không gradient), và tại sao việc tách biệt biểu diễn cố định với classifier trực tuyến là ưu điểm.

**IEEE Citations:** `[1]`

---

#### SLIDE 10: TẦNG 1&2 – CHUẨN HÓA VÀ RANDOM FOURIER FEATURES

**Mặt slide:**
- **Tiêu đề:** Ánh xạ không gian Hilbert bằng Random Fourier Features
- **[CHÈN HÌNH/SƠ ĐỒ 6 – Hilbert Mapping Illustration]:** Trực quan hóa hình học phép ánh xạ Bochner & RFF từ 2D sang $D$ chiều (xem Mục 5.4).
- **Định lý Bochner (nền tảng lý thuyết):**
  - k(x, y) = E_omega[cos(omega^T * (x-y))], omega ~ p(omega)
- **Công thức RFF (Rahimi & Recht 2007):**
  - Phi(x) = sqrt(2/D) * [cos(omega_1^T x + b_1), ..., cos(omega_D^T x + b_D)]^T
  - omega_j ~ N(0, 2*gamma*I_d), b_j ~ U[0, 2*pi]
  - Khi D lon: <Phi(x), Phi(y)> ≈ exp(-gamma * ||x-y||^2) = RBF kernel
- **PHAT HIEN PHAN BIEN CUA NHOM (hop do, noi bat):**
  - Bai bao tuyen bo: gamma = 1.0
  - Phan tich nhom: gamma=1.0 tren pixel z-score → pha omega^T*x ≈ 78 rad >> 2*pi → nhieu trang → Acc ≈ 10%
  - Code thuc te tac gia: gamma = 'scale' = 1/(d * Var(X)) moi dat ket qua

**Speaker Notes (< 500 từ):** Giải thích Định lý Bochner, cách RFF xấp xỉ RBF Kernel, và phát hiện phản biện quan trọng về gamma.

**IEEE Citations:** `[1]` `[6]` `[7] S. Bochner, "Monotone Funktionen, Stieltjessche Integrale und harmonische Analyse," Math. Ann., 1932.`

---

#### SLIDE 11: TẦNG 3 – CẬP NHẬT THỐNG KÊ TRỰC TUYẾN VÀ OAS COVARIANCE

**Mặt slide:**
- **Tiêu đề:** Tích lũy trực tuyến và Co ma trận hiệp phương sai OAS
- **Cập nhật đệ quy:**
  - n_c ← n_c + 1; mu_c ← mu_c + (1/n_c)(Phi(x_t) − mu_c)
  - A ← A + Phi(x_t)*Phi(x_t)^T
- **Pooled Covariance:** S = (1/N)(A − sum_c n_c * mu_c * mu_c^T)
- **OAS Shrinkage:**
  - Sigma_OAS = (1−rho)*S + rho*(Tr(S)/D)*I_D
  - rho = min( [(1−2/D)*Tr(S^2) + Tr^2(S)] / [(N+1−2/D)*(Tr(S^2)−Tr^2(S)/D)], 1 )
- **Y nghia:** Khi D >> N (high-dim low-sample), OAS dam bao Sigma_OAS luon duong xac dinh va kha nghich.

**Speaker Notes (< 500 từ):** Giải thích cách streaming covariance hoạt động mà không cần lưu dữ liệu, vấn đề singular matrix khi D >> N, và cách OAS tự động tính rho.

**IEEE Citations:** `[1]` `[8] Y. Chen et al., "Shrinkage algorithms for MMSE covariance estimation," IEEE Trans. Signal Process., 2010.`

---

#### SLIDE 12: TẦNG 4&5 – PHÂN LOẠI LDA VÀ ĐỘ PHỨC TẠP THUẬT TOÁN

**Mặt slide:**
- **Tiêu đề:** Phân loại Mahalanobis và Phân tích độ phức tạp
- **Ham phan loai:** y* = argmax_c [mu_c^T * Sigma_OAS^(-1) * Phi(x) − (1/2) * mu_c^T * Sigma_OAS^(-1) * mu_c]
- **Cholesky solve:** Sigma_OAS = L*L^T; giai L*L^T*W = M → khong nghich dao tuong minh
- **Bang Do phuc tap:**

| Giai doan | Thoi gian | Bo nho |
|---|---|---|
| Cap nhat mau | O(d*D + D^2) | O(D^2 + C*D) |
| Tinh W (Cholesky, 1 lan) | O((1/3)*D^3 + C*D^2) | O(D^2 + C*D) |
| Du doan 1 mau | O(d*D + C*D) | O(D) |

**Speaker Notes (< 500 từ):** Giải thích hàm phân loại Mahalanobis, lý do dùng Cholesky thay vì nghịch đảo trực tiếp, và tại sao bộ nhớ hằng số là ưu điểm quyết định.

**IEEE Citations:** `[1]` `[9] G. H. Golub and C. F. Van Loan, Matrix Computations, 4th ed. Johns Hopkins Univ. Press, 2013.`

---

#### SLIDE 13: GIAO THỨC ĐÁNH GIÁ THỰC NGHIỆM

**Mặt slide:**
- **Tiêu đề:** Giao thức thực nghiệm và Thước đo đánh giá
- **2 nhánh thực nghiệm:**
  - E1 – Pixel-based (Benchmark A): Học từ scratch; MNIST, CIFAR-10, CIFAR-100, TinyImageNet, miniImageNet
  - E2 – Foundation Model (Benchmark F): ViT-B/16 features (768 chieu); CIFAR-100, CUB-200, ImageNet-R, ImageNet-A
- **Giao thức streaming:** Class-Incremental, single-pass, du lieu xao tron trong task
- **Thuoc do – Average Incremental Accuracy:**
  - A_inc = (1/T) * sum_{t=1}^T R_t
  - R_t = accuracy tren toan bo cac lop da hoc den cuoi buoc t

**Speaker Notes (< 500 từ):** Mô tả thiết kế 2 nhánh thực nghiệm, lý do chọn các tập dữ liệu, thước đo Average Incremental Accuracy và ý nghĩa của nó trong đánh giá Continual Learning.

**IEEE Citations:** `[1]` `[5]`

---

### SECTION D – KẾT QUẢ TÁI LẬP (Slide 14–18)

---

#### SLIDE 14: KẾT QUẢ E1 – PIXEL-BASED (BENCHMARK A)

**Mặt slide:**
- **Tiêu đề:** Thực nghiệm E1: Tái lập Benchmark A – Học từ pixel thô
- **Thiết lập:** [INSERT: mo ta ngan D, seed, cau hinh]
- **[CHÈN HÌNH/SƠ ĐỒ 7 – Grouped Bar Chart]:** Biểu đồ cột đối đầu kết quả E1 Paper vs Team trên 5 datasets (xem Mục 5.4).
- **Bảng kết quả so sánh E1:**

| Dataset | RanDumb (Bai bao) | RanDumb (Nhom tai lap) | Do lech | W&B Run |
|---|---|---|---|---|
| MNIST | 98.3% | [INSERT: E1 MNIST acc] | [INSERT] | [INSERT: W&B URL] |
| CIFAR-10 | 55.6% | [INSERT: E1 CIFAR-10 acc] | [INSERT] | [INSERT: W&B URL] |
| CIFAR-100 | 28.6% | [INSERT: E1 CIFAR-100 acc] | [INSERT] | [INSERT: W&B URL] |
| TinyImageNet | 11.1–11.6% | [INSERT: E1 TinyImageNet acc] | [INSERT] | [INSERT: W&B URL] |
| miniImageNet | 17.7% | [INSERT: E1 miniImageNet acc] | [INSERT] | [INSERT: W&B URL] |

- **Phan tich nhanh:** [INSERT: nhan xet tong the ve do trung khop]

**Speaker Notes (< 500 từ):** Trình bày kết quả tái lập E1. Giải thích cấu hình thực nghiệm (D, seed, λ). So sánh với bảng gốc bài báo. Phân tích nguyên nhân lệch nếu có.

**IEEE Citations:** `[1]`

---

#### SLIDE 15: KẾT QUẢ E2 – VIT FEATURES (BENCHMARK F)

**Mặt slide:**
- **Tiêu đề:** Thực nghiệm E2: Tái lập Benchmark F – ViT-B/16 Features
- **Thiết lập:** 16 runs (4 datasets × 2 backbones IN-21k/IN-1k); RanDumb vs NCM
- **[CHÈN HÌNH/SƠ ĐỒ 8 – Grouped Bar Chart]:** Biểu đồ cột kết quả 8 cặp thực nghiệm RanDumb vs NCM (xem Mục 5.4).
- **Bảng kết quả E2:**

| Dataset | Backbone | RanDumb (Bao) | RanDumb (Nhom) | Lech | W&B |
|---|---|---|---|---|---|
| CIFAR-100 | IN-21k | 88.4% | 88.08% | -0.32% | x06w28q2 |
| CIFAR-100 | IN-1k | 81.1% | 81.71% | +0.61% | 421dknz1 |
| CUB-200 | IN-21k | 88.6% | 89.57% | +0.97% | 52l9g0d4 |
| CUB-200 | IN-1k | 81.6% | 84.34% | +2.74% | n1e3i762 |
| ImageNet-R | IN-21k | 74.3% | 74.65% | +0.35% | 8a6t1l1i |
| ImageNet-R | IN-1k | 66.3% | 66.90% | +0.60% | b1v68m6o |
| ImageNet-A | IN-21k | 55.7% | 66.72% | +11.02% | 90n2w8g8 |
| ImageNet-A | IN-1k | 43.8% | 61.35% | +17.55% | 8s583j3z |

- **Ket luan:** RanDumb >> NCM 100%; lech < ±3% tren CIFAR-100/CUB/ImageNet-R; lech lon tren ImageNet-A do split khac.

**Speaker Notes (< 500 từ):** Trình bày kết quả E2 chi tiết. Giải thích 3 kết luận: RanDumb >> NCM; tái lập gần chính xác trên 3/4 datasets; ImageNet-A lệch do split.

**IEEE Citations:** `[1]`

---

#### SLIDE 16: PHÂN TÍCH ĐỘ NHẠY E8 – OAS VÀ LAMBDA

**Mặt slide:**
- **Tiêu đề:** Thực nghiệm E8: Phân tích độ nhạy OAS Shrinkage và λ
- **Thiết lập E8:** [INSERT: mo ta ngan grid search]
- **Bảng kết quả E8-B:**

| Dataset | D | Shrinkage | Lambda-mode | Acc (3 seeds) | W&B |
|---|---|---|---|---|---|
| CIFAR-10 | 25,000 | OAS | Relative | [INSERT: E8 CIFAR-10 relative acc] | [INSERT: W&B URL] |
| CIFAR-10 | 25,000 | OAS | Fixed | [INSERT: E8 CIFAR-10 fixed acc] | [INSERT: W&B URL] |
| CIFAR-100 | 25,000 | OAS | Relative | [INSERT: E8 CIFAR-100 relative acc] | [INSERT: W&B URL] |
| CIFAR-100 | 25,000 | OAS | Fixed | [INSERT: E8 CIFAR-100 fixed acc] | [INSERT: W&B URL] |

- **Ket luan chinh:** [INSERT: nhan xet ve lambda-mode va OAS impact]

**Speaker Notes (< 500 từ):** Giải thích thiết kế E8 (grid search OAS × λ × D × dataset × seed). Kết quả: OAS vs no-OAS, relative λ vs fixed λ, và ảnh hưởng của D.

**IEEE Citations:** `[1]`

---

#### SLIDE 17: SO SÁNH VỚI SOTA (ABLATION STUDY BÀI BÁO)

**Mặt slide:**
- **Tiêu đề:** Ablation Study từ bài báo – Tầm quan trọng từng thành phần
- **[CHÈN HÌNH/SƠ ĐỒ 9 – Grouped Bar Chart]:** Biểu đồ cột 4 nhóm thể hiện mức sụt giảm khi bỏ OAS / RFF (xem Mục 5.4).
- **Bảng Ablation:**

| Cau hinh | MNIST | CIFAR-10 | CIFAR-100 | TinyImageNet |
|---|---|---|---|---|
| RanDumb day du | 98.3% | 55.6% | 28.6% | 11.1% |
| Bo Decorrelation (OAS) | 83.8% | 30.0% | 12.0% | 4.7% |
| Bo RFF Embedding | 88.0% | 41.6% | 19.0% | 8.0% |
| Bo ca hai (NCM tho) | 82.1% | 28.5% | 10.4% | 4.1% |

- **Kết luận:** Decorrelation quan trọng hơn Embedding trên CIFAR-10 (giảm 25.6% vs 14%)

**Speaker Notes (< 500 từ):** Phân tích ablation study từ bài báo. Giải thích ý nghĩa từng thành phần: OAS đóng vai trò "cân bằng phương sai", RFF đóng vai trò "nâng chiều". Cả hai đều thiết yếu.

**IEEE Citations:** `[1]`

---

### SECTION E – KIỂM CHỨNG CODE VÀ PHÂN TÍCH (Slide 18–22)

---

#### SLIDE 18: THỰC NGHIỆM E9 – KIỂM CHỨNG TỪNG TẦNG

**Mặt slide:**
- **Tiêu đề:** Thực nghiệm E9: So sánh số học từng tầng với mã nguồn tác giả
- **Dieu kien:** Code tac gia ghim tai commit `14a51ee`, moi truong cach ly (clean-room)
- **W&B Run:** icae4r9o (link: forge.coreweave.com/wandb/.../runs/icae4r9o)
- **Bang doi chieu 5 tang:**

| Tang | Chi so | Code tac gia | Code nhom | Do lech |
|---|---|---|---|---|
| 1. Chuan hoa | Max abs diff | — | — | 6.64 × 10^-4 |
| 2. RFF | gamma relative diff | 0.0003253 | 0.0003255 | 0.073% |
| 3. OAS | rho trung khop | 0.94603... | 0.94603... | 2.6 × 10^-18 ≈ machine epsilon |
| 4. LDA weights W | Cosine Similarity | W_author | W_team | 1.000000 (100%) |
| 5. Du doan | Agreement Rate | Acc = 7.50% | Acc = 7.50% | 100.00% |

- **Ket luan:** Pipeline nhom dat do chinh xac so hoc tuyet doi voi code tac gia.

**Speaker Notes (< 500 từ):** Mô tả cách thiết kế E9 (dữ liệu synthetic, seed cố định, kiểm tra từng layer output). Giải thích ý nghĩa từng chỉ số. Nhấn mạnh mức machine epsilon ở OAS và 100% agreement.

**IEEE Citations:** `[1]`

---

#### SLIDE 19: PHÁT HIỆN 6 ĐIỂM MÂU THUẪN TRONG CODE TÁC GIẢ

**Mặt slide:**
- **Tiêu đề:** Phân tích phản biện – 6 điểm bất nhất giữa bài báo và mã nguồn tác giả
- **Bảng 6 phat hien:**

| # | Van de | Bai bao tuyen bo | Thuc te trong code | Tac dong |
|---|---|---|---|---|
| 1 | Tham so gamma | gamma = 1.0 | gamma='scale' | gamma=1.0 → Acc≈10% (nhieu trang) |
| 2 | Luong online | Single-pass streaming | Batch NumPy array | Vi pham tinh than edge device |
| 3 | He so lambda | lambda trong [1e-6, 1e-4] | lambda = 0 (bo qua) | Lech ±0.8% voi bang cong bo |
| 4 | Loi syntax ViT | Code chay Benchmark F | IndentationError + NameError | Chua qua CI/CD |
| 5 | Co Augment | Benchmark F khong augment | --augment trong runall.sh | Thieu cong bang |
| 6 | Nghen co chai OAS | Chi phi tinh toan thap | store_precision=True → dao D×D | RAM overflow khi D>5000 |

**Speaker Notes (< 500 từ):** Phân tích từng phát hiện. Nhấn mạnh phát hiện #1 (gamma) là chấn động nhất – nếu chỉ đọc paper mà không xem code, không thể tái lập. Phát hiện #4 cho thấy code chưa được kiểm thử nghiêm túc.

**IEEE Citations:** `[1]` `[11] Ma nguon chinh thuc RanDumb, commit 14a51ee, apd20/randumb, 2024.`

---

#### SLIDE 20: PHÂN TÍCH NGUYÊN NHÂN SAI LỆCH (DEVIATION ANALYSIS)

**Mặt slide:**
- **Tiêu đề:** Phân tích nguyên nhân sai lệch trong tái lập
- **[CHÈN HÌNH/SƠ ĐỒ 10 – Fishbone Diagram]:** Sơ đồ xương cá phân tích 4 nhóm nguyên nhân sai lệch (xem Mục 5.4).
- **3 nhóm nguyên nhân chính:**
  1. Sai lệch tài liệu hóa: γ paper ≠ γ code; λ paper ≠ λ code
  2. Sai lệch thiết lập: Train/test split (ImageNet-A); seed không công bố
  3. Sai lệch kỹ thuật: Precision float64 vs float32; OAS batch vs streaming
- **Tóm tắt độ lệch:**

| Thuc nghiem | Dataset | Do lech tuyet doi | Nguyen nhan chinh |
|---|---|---|---|
| E2 | CIFAR-100 | ≤ 0.5% | Khop hoan hao |
| E2 | CUB-200, ImageNet-R | ≤ 2.74% | Split nho |
| E2 | ImageNet-A | 11–17% | Split khac biet lon |
| E1 | [INSERT dataset] | [INSERT lech] | [INSERT nguyen nhan] |

**Speaker Notes (< 500 từ):** Phân tích có hệ thống các nguồn gây lệch. Phân biệt lệch do "thiếu thông tin từ paper" vs "khác biệt thực sự về thuật toán". Rút ra bài học về reproducibility.

**IEEE Citations:** `[1]`

---

### SECTION F – PHÂN TÍCH NHẠY CẢM & THẢO LUẬN (Slide 21–25)

---

#### SLIDE 21: PHÂN TÍCH ẢNH HƯỞNG CHIỀU D (EMBEDDING DIMENSIONS)

**Mặt slide:**
- **Tiêu đề:** Ảnh hưởng của số chiều RFF (D) đến hiệu năng
- **[CHÈN HÌNH/SƠ ĐỒ 11 – Multi-line Chart]:** Biểu đồ đường Accuracy vs Chiều $D \in [1\text{K}, 25\text{K}]$ có đường gióng bão hòa tại $15\text{K}$ (xem Mục 5.4).
- **Dữ liệu từ bài báo:**

| Dataset | D=1K | D=5K | D=15K | D=25K | Muc tang 1K→25K |
|---|---|---|---|---|---|
| MNIST | ~94.7% | ~97.2% | ~98.2% | 98.3% | +3.6% |
| CIFAR-10 | ~45.2% | ~51.5% | ~54.2% | 55.6% | +10.4% |
| CIFAR-100 | ~21.6% | ~25.3% | ~27.5% | 28.6% | +7.0% |
| TinyImageNet | ~8.6% | ~9.9% | ~10.9% | 11.1% | +2.5% |

- **Kết luận:** Mức tăng bão hòa sau $D=15\text{K}$. $D=15\text{K}$ là điểm cân bằng tốt (hiệu năng/chi phí).

**Speaker Notes (< 500 từ):** Giải thích vì sao D lớn hơn thì tốt hơn (xấp xỉ kernel tốt hơn), nhưng có điểm bão hòa. Liên hệ với kết quả thực nghiệm E8 của nhóm.

**IEEE Citations:** `[1]`

---

#### SLIDE 22: PHÂN TÍCH ẢNH HƯỞNG DATA AUGMENTATION VÀ LAMBDA

**Mặt slide:**
- **Tiêu đề:** Ảnh hưởng của Flip Augmentation và tham số λ
- **Phan A – Flip Augmentation:**

| Dataset | Co Flip | Khong Flip | Muc giam |
|---|---|---|---|
| CIFAR-10 | 55.6% | 52.5% | -3.1% |
| CIFAR-100 | 28.6% | 26.9% | -1.7% |
| TinyImageNet | 11.1% | 10.7% | -0.4% |

- **Phan B – Ridge Parameter λ:** λ qua nho → Under-regularized → kem on dinh; λ toi uu khac nhau theo dataset; dataset phuc tap → can λ lon hon.
- **Nhom su dung:** λ relative (tuong doi) → on dinh hon tren moi cau hinh

**Speaker Notes (< 500 từ):** Giải thích ý nghĩa flip augmentation trong RanDumb (không dùng RandomCrop phức tạp). Phân tích λ: sự đánh đổi Under/Over-regularization. Kết quả E8 của nhóm về λ relative vs fixed.

**IEEE Citations:** `[1]`

---

#### SLIDE 23: SO SÁNH RANDUMB VỚI JOINT TRAINING (UPPER BOUND)

**Mặt slide:**
- **Tiêu đề:** RanDumb so với Upper Bound – Joint Training (CIFAR-100)
- **Du lieu tu bai bao:**

| Model | Accuracy CIFAR-100 |
|---|---|
| CNN × 1 | 62.2% |
| CNN × 8 | 69.9% |
| CNN × 16 | 76.8% |
| ResNet-50 | 56.2% |
| ResNet-101 | 56.8% |
| **RanDumb** | **74.8%** |
| **Joint Training** | **79.58%** |

- **Ket luan:** RanDumb dat ~94% hieu nang cua Joint Training ma khong can luu du lieu.
- **Y nghia:** RanDumb thu hep 70–90% khoang cach giua OCL va Joint Learning.

**Speaker Notes (< 500 từ):** Giải thích Joint Training là upper bound không thực tế (truy cập toàn bộ dữ liệu). RanDumb không học biểu diễn nhưng lại tiệm cận kết quả của các kiến trúc lớn được huấn luyện đầy đủ. Hàm ý: benchmark design hiện tại bất lợi cho Deep CL methods.

**IEEE Citations:** `[1]`

---

#### SLIDE 24: HẠN CHẾ VÀ HƯỚNG NGHIÊN CỨU TƯƠNG LAI

**Mặt slide:**
- **Tiêu đề:** Hạn chế của RanDumb và Hướng nghiên cứu tương lai
- **Han che (theo bai bao + phan tich nhom):**
  1. Thieu giai thich ly thuyet tai sao Deep CL hoc bieu dien kem
  2. Chi hieu qua trong Low-exemplar / Online CL – khong phai moi tinh huong
  3. Tinh toan A = sum(PhiPhi^T) ton O(D^2) RAM = 25K^2 × 8 bytes ≈ 50 GB (float64)
  4. Khong phu hop khi mien du lieu hoan toan xa la voi bat ky prior nao
- **Huong tuong lai:**
  - Giai thich ly thuyet ve representation drift trong OCL
  - Ap dung RFF dang sparse / hierarchical de giam RAM
  - Ket hop RanDumb voi it exemplar (hybrid approach)

**Speaker Notes (< 500 từ):** Trình bày hạn chế một cách thẳng thắn. Nhấn mạnh vấn đề RAM (D² = 625M entries float64 = 50GB) là rào cản thực tế. Hướng tương lai là giảm D bằng feature selection thay vì random.

**IEEE Citations:** `[1]`

---

### SECTION G – KẾT LUẬN VÀ TIẾN ĐỘ (Slide 25–28)

---

#### SLIDE 25: ĐIỂM MẤU CHỐT – CÁC PHÁT HIỆN QUAN TRỌNG NHẤT

**Mặt slide:**
- **Tiêu đề:** Tổng hợp – Những phát hiện quan trọng nhất từ tái lập
- **5 phat hien chinh (5 hop mau):**
  1. E1: Tai lap thanh cong Benchmark A voi do lech [INSERT] tren [INSERT dataset chinh]
  2. E2: 15/16 runs lech < 3% so voi bai bao; ImageNet-A lech do split
  3. E9: Code nhom trung khop 100% voi code tac gia o muc machine epsilon
  4. 6 loi phat hien: gamma trong paper sai, lambda=0 khong cong bo, IndentationError trong ViT code
  5. E8: lambda-relative on dinh hon lambda-fixed; OAS cai thien [INSERT]% so voi no-OAS

**Speaker Notes (< 500 từ):** Tổng kết toàn bộ hành trình tái lập. Nhấn mạnh giá trị học thuật của các phát hiện phản biện. Kết luận về độ tin cậy của bài báo gốc.

**IEEE Citations:** `[1]`

---

#### SLIDE 26: NHẬN XÉT TỔNG QUAN VỀ BÀI BÁO

**Mặt slide:**
- **Tiêu đề:** Đánh giá tổng thể bài báo RanDumb
- **Bang Diem manh / Diem yeu:**

| Diem manh | Diem yeu |
|---|---|
| Y tuong don gian, ket qua an tuong | Thieu giai thich ly thuyet |
| Thuc nghiem quy mo lon, 6 benchmark | gamma trong paper sai so voi code |
| Lat do gia dinh da ton tai nhieu nam | CIFAR-10 van thua PEC (3.3%) |
| Code cong khai, co the tai lap | Code chua loi syntax, thieu CI/CD |
| Phuong phap hoi tu nhanh, khong can GPU | RAM O(D^2) = 50GB khi D=25K float64 |

- **Nhan xet cuoi:** Dong gop thuc nghiem xuat sac; dong gop ly thuyet han che.

**Speaker Notes (< 500 từ):** Trình bày đánh giá cân bằng và khách quan về bài báo. Phân biệt rõ: giá trị của bài báo nằm ở "đặt lại câu hỏi" chứ không phải "giải quyết hoàn toàn" bài toán.

**IEEE Citations:** `[1]`

---

#### SLIDE 27: TIẾN ĐỘ ĐỒ ÁN VÀ KẾ HOẠCH

**Mặt slide:**
- **Tiêu đề:** Tiến độ thực hiện và Kế hoạch hoàn thiện
- **Bang trang thai:**

| Hang muc | Nguoi phu trach | Trang thai | Minh chung |
|---|---|---|---|
| Ma nguon loi (src/randumb/) | Dang (lead) | 100% | GitHub main, unit tests |
| Thuc nghiem E9 | Huy | 100% | W&B run icae4r9o |
| Thuc nghiem E2 (ViT) | Kien | 100% | 16 W&B runs |
| Thuc nghiem E1 (pixel) | Dang | Hoan thanh | [INSERT: W&B group link] |
| Thuc nghiem E8 (sensitivity) | Huy & Kien | 24/24 runs | W&B group E8 |
| Slide Deck | Huy (lead) | Dang hoan thien | File nay |
| Poster (05/12) | Ca doi | Chua bat dau | — |

- **Moc thoi gian:** Slide nop Moodle 21/11/2026 | Poster Workshop 05/12/2026

**Speaker Notes (< 500 từ):** Tổng kết tiến độ, nêu bật những gì đã hoàn thành. Kế hoạch: sau slide deck sẽ dồn toàn lực cho poster (E3–E7 ablation study chi tiết).

**IEEE Citations:** `[1]`

---

#### SLIDE 28: KẾT LUẬN (CONCLUSION)

**Mặt slide:**
- **Tiêu đề:** Kết luận
- **3 thong diep chinh (noi bat, toi gian):**
  1. RanDumb xac nhan: Trong OCL voi rang buoc khat khe, bieu dien ngau nhien vuot troi bieu dien hoc duoc
  2. Nhom tai lap thanh cong: E1, E2, E9 voi do trung khop cao va 6 phat hien phan bien co gia tri
  3. Bai hoc reproducibility: Paper quality ≠ Code quality; kiem chung doc lap la bat buoc trong nghien cuu khoa hoc
- **Cam on & Repo link:**
  - `https://github.com/24122039Kien/RanDumb-reproduction`
  - W&B: `randumb-reproduction-24TNT`

**Speaker Notes (< 500 từ):** Tóm lại toàn bộ nội dung thuyết trình. Nhấn mạnh bài học về reproducibility và academic integrity. Cảm ơn hội đồng.

**IEEE Citations:** `[1]`

---

## PHẦN 3 – DANH SÁCH TRÍCH DẪN IEEE (MASTER REFERENCE LIST)

```
[1] A. Prabhu, H. A. Al Kader Hammoud, S.-N. Lim, B. Ghanem, P. H. S. Torr, and A. Bibi,
    "RanDumb: Random Representations Outperform Online Continually Learned Representations,"
    in Advances in Neural Information Processing Systems (NeurIPS), vol. 37, 2024.

[2] T. L. Hayes and C. Kanan, "Lifelong machine learning with deep streaming linear
    discriminant analysis," in Proc. IEEE/CVF Conf. Comput. Vision Pattern Recognit. Workshops
    (CVPRW), 2020.

[3] A. Prabhu, P. H. S. Torr, and P. K. Dokania, "GDumb: A Simple Approach that Questions
    Our Progress in Continual Learning," in Proc. Eur. Conf. Comput. Vision (ECCV), 2020.

[4] M. D. McDonnell, D. Gong, A. Parvaneh, E. Abbasnejad, and A. van den Hengel, "RanPAC:
    Random Projections and Pre-trained Models for Continual Learning," in Advances in Neural
    Information Processing Systems (NeurIPS), 2023.

[5] Z. Mai, R. Li, J. Jeong, D. Quispe, H. Kim, and S. Sanner, "Online continual learning in
    image classification: An empirical survey," Neurocomputing, vol. 469, pp. 28-51, 2022.

[6] A. Rahimi and B. Recht, "Random features for large-scale kernel machines," in Advances in
    Neural Information Processing Systems (NeurIPS), vol. 20, 2007.

[7] S. Bochner, "Monotone Funktionen, Stieltjessche Integrale und harmonische Analyse,"
    Mathematische Annalen, vol. 108, pp. 378-410, 1933.

[8] Y. Chen, A. Wiesel, Y. C. Eldar, and A. O. Hero, "Shrinkage algorithms for MMSE
    covariance estimation," IEEE Transactions on Signal Processing, vol. 58, no. 10,
    pp. 5016-5029, 2010.

[9] G. H. Golub and C. F. Van Loan, Matrix Computations, 4th ed. Baltimore, MD: Johns
    Hopkins Univ. Press, 2013.

[10] Nhom nghien cuu 24TNT, "Experiment E9 Layer-by-Layer Verification Report," W&B Run
     icae4r9o, Nov. 2026. [Online]. Available:
     https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/icae4r9o

[11] A. Prabhu, "RanDumb Official Code Repository," Commit 14a51ee, apd20/randumb,
     GitHub, 2024. [Online]. Available: https://github.com/apd20/randumb
```

---

## PHẦN 4 – DANH SÁCH PLACEHOLDER (CẦN NHÓM ĐIỀN SAU)

| Placeholder | Vi tri (Slide) | Nguon du lieu |
|---|---|---|
| `[INSERT: E1 MNIST acc]` | Slide 14 | W&B group E1, metric test_acc |
| `[INSERT: E1 CIFAR-10 acc]` | Slide 14 | W&B group E1 |
| `[INSERT: E1 CIFAR-100 acc]` | Slide 14 | W&B group E1 |
| `[INSERT: E1 TinyImageNet acc]` | Slide 14 | W&B group E1 |
| `[INSERT: E1 miniImageNet acc]` | Slide 14 | W&B group E1 |
| `[INSERT: W&B URL]` (E1 ×5) | Slide 14 | Copy tu W&B runs |
| `[INSERT: E8 CIFAR-10 relative acc]` | Slide 16 | W&B group E8, D=25K, OAS, lambda-relative |
| `[INSERT: E8 CIFAR-10 fixed acc]` | Slide 16 | W&B group E8, D=25K, OAS, lambda-fixed |
| `[INSERT: E8 CIFAR-100 relative acc]` | Slide 16 | W&B group E8 |
| `[INSERT: E8 CIFAR-100 fixed acc]` | Slide 16 | W&B group E8 |
| `[INSERT: W&B URL]` (E8 ×4) | Slide 16 | Copy tu W&B runs |
| `[INSERT: nhan xet tong the E1]` | Slide 14 | Phan tich sau khi co so |
| `[INSERT: nhan xet E8 lambda]` | Slide 16 | Phan tich sau khi co so |
| `[INSERT: nhan xet E8 OAS impact]` | Slide 16 | Phan tich sau khi co so |
| `[INSERT: W&B URL E1]` (cac runs) | Slides 14, 20 | W&B project page |

---

## PHẦN 5 – HƯỚNG DẪN THIẾT KẾ TRỰC QUAN CHO MODEL

### 5.1. Màu sắc và Theme đề xuất

```
Mau chu dao:
- Primary:   #1E3A5F (Xanh dam – hoc thuat)
- Accent:    #E63946 (Do – phat hien phan bien, canh bao)
- Highlight: #457B9D (Xanh trung – phuong phap)
- Success:   #2D6A4F (Xanh la – ket qua tot)
- Neutral:   #F1FAEE (Nen trang nga)
```

### 5.2. Font chữ đề xuất

```
Tieu de (Heading): Calibri Bold, 28-36pt
Noi dung (Body):   Calibri Regular, 18-22pt
Code/Formula:      Consolas Regular, 14-16pt
Speaker Notes:     Calibri Regular, 11-12pt
```

### 5.3. Layout đề xuất cho từng loại slide

- **Slide Title (1):** Toan trang, hinh nen gradient, text centered
- **Slide Content thong thuong:** 70% noi dung chinh, 30% sidebar ghi diem chinh
- **Slide Bang bieu:** Header dam mau, alternating row colors
- **Slide So do:** Chua 60% dien tich cho diagram, 40% cho caption + citations
- **Slide Section Header:** Mau nen dam, so section noi bat, tieu de lon

### 5.4. Bảng danh mục hình ảnh và sơ đồ tự vẽ chi tiết

> **Quy định môn học:** Tuyệt đối KHÔNG chụp ảnh màn hình từ paper. Toàn bộ hình ảnh, sơ đồ phải do nhóm/model tự vẽ bằng code (matplotlib, seaborn), SmartArt, hoặc công cụ vẽ vector.

Dưới đây là danh mục chi tiết 11 hình ảnh/sơ đồ xuất hiện trong toàn bộ Slide Deck:

| Slide | Tên hình ảnh / Sơ đồ | Loại hình | Nội dung chi tiết cần thể hiện | Vị trí trên slide & Lưu ý |
|---|---|---|---|---|
| **Slide 2** | Sơ đồ luồng dữ liệu Online Continual Learning | Horizontal Flowchart | Luồng ảnh $[x_1, y_1] \to [x_2, y_2] \to \dots \to [x_t, y_t]$ trôi qua mô hình xử lý trực tuyến. Kèm 3 icon/hộp: Single-pass (1 mũi tên), Zero-buffer (thùng rác gạch chéo), Constant memory (RAM cố định). | Căn giữa nửa trên slide (chiếm 45% diện tích). |
| **Slide 3** | Sơ đồ đối đầu: Học biểu diễn vs Biểu diễn ngẫu nhiên | 2-Column Comparison Diagram | Cột trái (Mạng sâu): Mũi tên đỏ lượn sóng biểu thị Representation Drift / Quên thảm khốc. Cột phải (RanDumb): Phép chiếu RFF cố định phẳng lặng + Linear Classifier vững chắc. | Chia đôi nửa dưới slide, dùng màu đỏ (thất bại) vs xanh (ổn định). |
| **Slide 5** | Ma trận định vị $2 \times 2$ Research Gap | $2 \times 2$ Grid Matrix | Trục tung: Nguồn biểu diễn (Pretrained vs Pixel thô). Trục hoành: Bộ nhớ (Có lưu mẫu vs Zero-replay). Nêu tên các thuật toán ở 3 ô, ô góc dưới-phải (Pixel thô + Zero-replay) highlight sao vàng cho RanDumb. | Căn giữa slide, chiếm 60% chiều ngang slide. |
| **Slide 8** | Sơ đồ kịch bản Class-Incremental Streaming | Timeline / Task Sequence Diagram | Dòng thời gian phân chia $K$ task: Task 1 (Lớp 1, 2) $\to$ Task 2 (Lớp 3, 4) $\to \dots$ Mỗi bước dữ liệu cũ biến mất vĩnh viễn, mô hình chỉ cập nhật trạng thái $\theta_t \to \theta_{t+1}$. | Đặt bên phải slide, chiếm 40% diện tích. |
| **Slide 9** | Sơ đồ Kiến trúc Pipeline 4 khối RanDumb | Vertical Block Diagram (Flowchart) | 4 khối nối nhau bằng mũi tên xuôi: Khối 1 (Chuẩn hóa Z-Score) $\to$ Khối 2 (RFF Cosine Projection) $\to$ Khối 3 (Tích lũy thống kê OAS) $\to$ Khối 4 (Giải hệ Cholesky LDA). | Chiếm toàn bộ nửa trái slide (55% diện tích), màu xanh học thuật #1E3A5F. |
| **Slide 10** | Minh họa hình học Định lý Bochner & RFF | Hilbert Space Mapping Illustration | Không gian pixel 2D phi tuyến uốn lượn $\to$ Ánh xạ qua $\cos(\Omega x + b)$ $\to$ Không gian Hilbert $D$ chiều nơi các điểm dữ liệu phân tách tuyến tính bằng siêu phẳng. | Nửa phải slide, trực quan hóa cơ chế ánh xạ. |
| **Slide 14** | Biểu đồ cột kết quả tái lập E1 | Grouped Bar Chart | Trục x: 5 datasets (MNIST, CIFAR-10, CIFAR-100, TinyImageNet, miniImageNet). Trục y: Accuracy (0–100%). Mỗi dataset gồm 2 cột: Cột xám (Paper) vs Cột xanh đậm (Nhóm tái lập). | Nửa phải slide bên cạnh bảng số liệu. |
| **Slide 15** | Biểu đồ cột kết quả ViT Benchmark F (E2) | Grouped Bar Chart | So sánh RanDumb vs NCM trên 8 cặp thực nghiệm (4 datasets $\times$ 2 backbones). Thể hiện rõ RanDumb luôn cao hơn NCM từ 3% đến 17%. | Đặt dưới bảng kết quả hoặc chiếm 40% diện tích bên phải. |
| **Slide 17** | Biểu đồ Ablation Study thành phần | Grouped Bar Chart (4 nhóm) | 4 cụm cột ứng với 4 dataset. Mỗi cụm 4 thanh: Full RanDumb, Không OAS, Không RFF, Không cả hai (NCM thô). Thể hiện rõ sự tụt dốc thảm hại khi bỏ OAS. | Căn giữa phía dưới slide. |
| **Slide 20** | Sơ đồ xương cá phân tích sai lệch (Fishbone) | Cause-and-Effect / Fishbone Diagram | Đầu cá: "Sai lệch thực nghiệm". Các nhánh xương: (1) Tài liệu hóa (γ, λ), (2) Phân chia dữ liệu (split ImageNet-A), (3) Độ chính xác số (float32 vs float64), (4) Phần cứng/thư viện. | Nửa phải slide, bổ trợ cho bảng phân tích nguyên nhân. |
| **Slide 21** | Biểu đồ đường ảnh hưởng số chiều RFF ($D$) | Multi-line Chart | Trục x: $D \in [1\text{K}, 5\text{K}, 10\text{K}, 15\text{K}, 25\text{K}]$. Trục y: Accuracy. 4 đường biểu diễn 4 dataset. Điểm $D=15\text{K}$ có đường gióng nét đứt ghi chú "Điểm bão hòa tối ưu". | Chiếm 50% diện tích slide (bên phải bảng số liệu). |

---

## PHẦN 6 – CHECKLIST TRƯỚC KHI NỘP

- [ ] **20–40 slides** (nham 28–32)
- [ ] **Moi slide co 2 phan:** Mat slide + Speaker Notes (≤500 tu)
- [ ] **Mat slide:** Khong co doan van dai (≥3 cau lien tiep); dung bullet, bang, so do
- [ ] **IEEE citations** o cuoi moi slide do (khong co slide References rieng)
- [ ] **Tat ca hinh anh:** Tu ve (khong screenshot tu paper)
- [ ] **Moi so lieu E2:** Co link W&B run day du
- [ ] **Moi placeholder `[INSERT: ...]`:** Da duoc nhom dien truoc khi nop
- [ ] **File size:** < 20 MB
- [ ] **Khong co loi chinh ta** tieng Viet hoac sai cong thuc
- [ ] **Slide 1:** Title slide day du ten nhom, repo, hoi nghi
- [ ] **Slide cuoi:** Ket luan + repo + W&B link

---

*Het huong dan. Model co the bat dau thiet ke khung slide theo cau truc tren.*  
*Moi thac mac ve noi dung ky thuat, lien he: Phan Gia Huy (huyphan0806 on GitHub).*
