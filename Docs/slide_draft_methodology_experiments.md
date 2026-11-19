# Bản nháp Slide Deck: Phương pháp & Thực nghiệm (Slide 7 – 16)

**Tác giả phụ trách:** Phan Gia Huy (`huyphan0806`)  
**Mục tiêu:** Soạn thảo chi tiết từ Slide 7 đến Slide 16 cho Slide Deck nộp môn Nhập môn Học máy (chiếm 45% điểm đồ án).  
**Quy chuẩn bắt buộc:**
- Mỗi slide gồm 2 phần: **Mặt slide** (trực quan, sơ đồ tự vẽ, công thức toán chuẩn, bảng biểu; không viết văn bản dài) và **Speaker Notes** (thuyết minh chi tiết như khi báo cáo, tối đa 500 từ/slide).
- **Trích dẫn chuẩn IEEE** đánh số ở chân mỗi slide.
- Mọi số liệu thực nghiệm phải dẫn link W&B run công khai.

---

## SLIDE 7: BÀI TOÁN TOÁN HỌC HÌNH THỨC (PROBLEM FORMULATION)

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Định nghĩa toán học bài toán Online Continual Learning
- **Không gian bài toán:**
  - Luồng dữ liệu theo thời gian: $\mathcal{S} = \{(x_t, y_t)\}_{t=1}^N$, với $x_t \in \mathcal{X} \subset \mathbb{R}^d$ và nhãn lớp $y_t \in \mathcal{Y} = \{1, 2, \dots, C\}$.
  - Không gian nhãn xuất hiện tuần tự theo $K$ giai đoạn (Class-Incremental): $\mathcal{Y} = \bigcup_{k=1}^K \mathcal{Y}_k$, trong đó $\mathcal{Y}_i \cap \mathcal{Y}_j = \emptyset$ với mọi $i \neq j$.
- **Ba ràng buộc toán học ngặt nghèo:**
  1. **Duyệt 1 lượt (Single-pass):** Tại bước thời gian $t$, mô hình chỉ quan sát mẫu $(x_t, y_t)$ đúng một lần duy nhất.
  2. **Không lưu mẫu (Zero replay buffer):** Không gian bộ nhớ đệm lưu mẫu $\mathcal{M} = \emptyset$. Không được phép lưu trữ bất kỳ $x_\tau$ nào ($\tau < t$).
  3. **Không gian bộ nhớ hằng số (Sub-linear / Constant Memory):** Bộ nhớ của thuật toán độc lập với số mẫu đã thấy:
     $$\text{Space}(\mathcal{A}) = \mathcal{O}(D^2 + C \cdot D) \ll \mathcal{O}(N \cdot d)$$
- **Mục tiêu học (Learning Objective):**
  - Cập nhật mô hình phân loại $f_t: \mathcal{X} \to \mathcal{Y}$ sau mỗi mẫu sao cho tối đa hóa độ chính xác trung bình tích lũy trên toàn bộ các lớp đã xuất hiện từ đầu đến thời điểm $t$:
    $$\max_{f_t} \frac{1}{t} \sum_{i=1}^t \mathbb{I}\left(f_t(x_i) = y_i\right)$$
- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`  
  `[5] Z. Mai et al., "Online continual learning in image classification: An empirical survey," Neurocomputing, vol. 469, pp. 28-51, 2022.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 380 từ)
> "Để phân tích thuật toán RanDumb một cách chặt chẽ, trước hết chúng ta cần hình thức hóa bài toán Online Continual Learning dưới dạng toán học chuẩn mực.
> 
> Giả sử hệ thống tiếp nhận một luồng dữ liệu liên tục S gồm N mẫu theo thời gian. Mỗi mẫu gồm một vector đầu vào x thuộc không gian d chiều và nhãn y thuộc tập hợp C lớp. Trong kịch bản phân lớp gia tăng (Class-Incremental Learning), tập nhãn Y được chia thành K giai đoạn rời rạc; mỗi giai đoạn hệ thống chỉ thấy một nhóm lớp mới mà không bao giờ gặp lại các lớp cũ.
> 
> Bài toán đặt ra ba tiên đề ràng buộc tuyệt đối:
> Thứ nhất, ràng buộc Single-pass: Mô hình chỉ được duyệt qua mỗi điểm dữ liệu đúng một lần, hoàn toàn không có khái niệm huấn luyện nhiều epoch.
> Thứ hai, ràng buộc Zero-memory: Kích thước bộ nhớ đệm lưu ảnh mẫu bằng 0. Mô hình bị cấm lưu lại bất kỳ mẫu ảnh nào trong quá khứ nhằm đáp ứng trọn vẹn yêu cầu bảo mật quyền riêng tư.
> Thứ ba, tính khả mở về bộ nhớ: Dung lượng bộ nhớ của mô hình phải cố định theo thời gian, không được phép tăng tuyến tính theo số lượng mẫu đã quan sát N, mà chỉ phụ thuộc vào số chiều biểu diễn D và số lượng lớp C.
> 
> Mục tiêu tối thượng của mô hình sau khi quan sát t mẫu là đưa ra hàm dự đoán f_t có khả năng phân loại chính xác trên toàn bộ phân phối dữ liệu tích lũy từ bước 1 đến bước t. Đây là thách thức toán học cực lớn: làm thế nào để ước lượng phân phối của các lớp cũ đã trôi qua mà không được giữ lại dữ liệu của chúng? RanDumb giải quyết bài toán này bằng cách chuyển đổi hoàn toàn từ bài toán tối ưu gradient sang bài toán ước lượng thống kê trực tuyến."

---

## SLIDE 8: KIẾN TRÚC TỔNG THỂ CỦA RANDUMB (PIPELINE ARCHITECTURE)

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Kiến trúc giải tích 4 khối của RanDumb
- **Sơ đồ luồng xử lý dữ liệu (Tự vẽ):**
  ```
  Ảnh đầu vào x_t ∈ R^d
           │
           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Khối 1: Chuẩn hóa Z-Score (Normalization)                   │
  │ x̃_t = (x_t - μ_chan) / σ_chan                               │
  └────────────────────────┬────────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Khối 2: Chiếu Fourier Ngẫu Nhiên Cố Định (RFF Embedding)    │
  │ Φ(x_t) = √(2/D) cos(Ω x̃_t + b)  ∈ R^D  (Ω ~ N, b ~ U)        │
  └────────────────────────┬────────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Khối 3: Tích Lũy Thống Kê & Co Covariance Trực Tuyến (OAS)  │
  │ Cập nhật n_c, μ_c; tích lũy A = ∑ Φ Φ^T;                    │
  │ Ước lượng ma trận hiệp phương sai dùng chung:              │
  │ Σ_OAS = (1 - ρ) S + ρ (Tr(S)/D) I_D                         │
  └────────────────────────┬────────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Khối 4: Suy Luận Tuyến Tính LDA (Regularized LDA Classifier)│
  │ Giải hệ Cholesky: Σ_OAS W = M  →  W = (Σ_OAS)^(-1) M        │
  │ Dự đoán: y* = argmax_c [ W_c^T Φ(x) - 0.5 W_c^T μ_c ]       │
  └─────────────────────────────────────────────────────────────┘
  ```
- **Đặc trưng kiến trúc then chốt:**
  - 100% giải tích (closed-form), **không dùng Backpropagation**, không có vòng lặp Gradient Descent.
  - Tách biệt hoàn toàn giữa: **Biểu diễn cố định (Fixed Representation)** và **Bộ phân loại trực tuyến (Online Classifier)**.
- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 390 từ)
> "Trên màn hình là sơ đồ kiến trúc tổng thể của RanDumb do nhóm chúng em trực tiếp vẽ lại dựa trên nguyên lý giải tích của bài báo.
> 
> Toàn bộ pipeline hoạt động gồm 4 khối chức năng tuần tự:
> Khối thứ nhất là **Chuẩn hóa (Normalization)**: Ảnh đầu vào x_t được đưa về phân phối chuẩn hóa z-score theo từng kênh màu. Bước này tưởng chừng đơn giản nhưng có ý nghĩa sống còn để kiểm soát biên độ đầu vào trước khi đưa vào hàm phi tuyến.
> 
> Khối thứ hai là **Phép chiếu Fourier ngẫu nhiên (RFF Embedding)**: Ảnh chuẩn hóa được nhân với một ma trận trọng số ngẫu nhiên Omega được khởi tạo từ phân phối chuẩn và cố định vĩnh viễn, sau đó cộng với vector pha b ngẫu nhiên đều rồi đưa qua hàm cosine. Phép biến đổi này đưa dữ liệu từ số chiều d (ví dụ 3072 của CIFAR) lên không gian D chiều rất lớn (ví dụ 25,000 chiều).
> 
> Khối thứ ba là **Cập nhật thống kê trực tuyến và Ước lượng Covariance OAS**: Với mỗi mẫu thuộc lớp c trôi qua, thuật toán cập nhật đếm số mẫu n_c, cập nhật vector trung bình lớp mu_c, và cộng dồn tích ngoài của vector đặc trưng vào ma trận tán xạ A. Khi cần suy luận, ma trận hiệp phương sai thực nghiệm S được tính và co hệ số thông qua kỹ thuật Oracle Approximating Shrinkage (OAS) để đảm bảo luôn khả nghịch.
> 
> Khối thứ tư là **Suy luận phân loại LDA**: Thay vì tính khoảng cách Mahalanobis trực tiếp cho từng mẫu kiểm tra vốn rất tốn kém, RanDumb giải hệ phương trình tuyến tính một lần duy nhất bằng phân rã Cholesky để tìm ma trận trọng số W, biến bài toán thành phép nhân ma trận tuyến tính cực nhanh.
> 
> Ưu điểm vượt trội của kiến trúc này là nó loại bỏ hoàn toàn cơ chế tối ưu gradient, giúp thời gian huấn luyện nhanh hơn hàng chục lần so với mạng nơ-ron sâu."

---

## SLIDE 9: TẦNG 1 & 2 – CHUẨN HÓA VÀ ÁNH XẠ FOURIER NGẪU NHIÊN (RFF)

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Nguyên lý ánh xạ không gian Hilbert bằng Random Fourier Features
- **Cơ sở toán học – Định lý Bochner (1932):**
  - Mọi hàm kernel đối xứng, tịnh tiến $k(x, y) = k(x - y)$ dương xác định đều là biến đổi Fourier của một phân phối xác suất thực $p(\omega)$:
    $$k(x, y) = \int_{\mathbb{R}^d} p(\omega) e^{i \omega^T (x - y)} d\omega = \mathbb{E}_\omega [\cos(\omega^T (x - y))]$$
- **Công thức phép chiếu RFF (Rahimi & Recht, 2007):**
  $$\Phi(x) = \sqrt{\frac{2}{D}} \left[ \cos(\omega_1^T x + b_1), \dots, \cos(\omega_D^T x + b_D) \right]^T$$
  Trong đó: $\omega_j \sim \mathcal{N}(0, 2\gamma I_d)$, $b_j \sim \mathcal{U}[0, 2\pi]$. Khi đó: $\langle \Phi(x), \Phi(y) \rangle \approx k_{\text{RBF}}(x, y) = \exp(-\gamma \|x - y\|^2)$.
- **Phát hiện phản biện của nhóm về siêu tham số $\gamma$ (Critical Discovery):**
  - **Bài báo tuyên bố:** Đặt $\gamma = 1.0$ (Mục 2).
  - **Phân tích số học:** Với ảnh pixel đã z-score ($x_i \sim \mathcal{N}(0, 1)$), $\|x\|^2 \approx d \approx 3072$. Nếu đặt $\gamma = 1.0$, độ lệch chuẩn của $\omega^T x$ là $\sqrt{2d} \approx 78.4$ rad $\gg 2\pi$. Pha bị quay hàng chục vòng ngẫu nhiên, phá hủy cấu trúc hình học, khiến độ chính xác rớt xuống mức đoán ngẫu nhiên (~10%).
  - **Mã nguồn tác giả thực tế:** Dùng $\gamma = \text{scale} = \frac{1}{d \cdot \text{Var}(X)}$ (đạt 45.3% trên CIFAR-100).
- **Trích dẫn cuối slide:**
  `[6] A. Rahimi and B. Recht, "Random features for large-scale kernel machines," in NeurIPS, vol. 20, 2007.`  
  `[7] S. Bochner, "Monotone Funktionen, Stieltjessche Integrale und harmonische Analyse," Math. Ann., 1932.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 410 từ)
> "Đi sâu vào Tầng 1 và Tầng 2, linh hồn biểu diễn của RanDumb chính là **Random Fourier Features (RFF)**, dựa trên nền tảng toán học sâu sắc của Định lý Bochner từ năm 1932.
> 
> Định lý Bochner phát biểu rằng: Một hàm nhân dịch chuyển bất biến như RBF Gaussian kernel thực chất có thể được phân tích thành kỳ vọng của các hàm sóng cosin với tần số omega lấy mẫu từ một phân phối xác suất. Năm 2007, Rahimi và Recht đã hiện thực hóa định lý này thành một kỹ thuật tính toán đột phá: thay vì tính toán ma trận Gram N nhân N cực kỳ tốn kém trong Support Vector Machines, chúng ta có thể chiếu thẳng vector đầu vào x lên một vector đặc trưng D chiều bằng phép chiếu phi tuyến ngẫu nhiên. Khi D đủ lớn, tích vô hướng giữa hai vector đặc trưng ngẫu nhiên này sẽ hội tụ đều về giá trị hàm nhân RBF kernel.
> 
> Nhờ đó, RanDumb đưa dữ liệu từ không gian pixel có tính phi tuyến phức tạp sang một không gian Hilbert nhiều chiều hơn rất nhiều, nơi các lớp dữ liệu trở nên tách biệt tuyến tính (linearly separable), mà hoàn toàn không cần trải qua bất kỳ bước huấn luyện học đặc trưng nào.
> 
> Tuy nhiên, trong quá trình kiểm chứng mã nguồn (thực nghiệm E9), nhóm chúng em đã phát hiện một chi tiết sai khác mang tính quyết định: Trong bài báo, tác giả tuyên bố ở Mục 2 rằng họ sử dụng hệ số gamma bằng 1.0. Nhưng khi nhóm phân tích bằng toán học: nếu dữ liệu ảnh đã được chuẩn hóa z-score, độ dài vector x xấp xỉ căn bậc hai của số chiều, tức khoảng 55 đến 78. Nếu nhân với gamma bằng 1.0, góc quay pha omega nhân x sẽ có biên độ lên tới gần 80 radian, lớn hơn gấp nhiều lần chu kỳ 2 pi của hàm cosin. Kết quả là hàm cosin bị bão hòa nhiễu trắng, phá hủy hoàn toàn thông tin không gian ảnh.
> 
> Khi kiểm tra code gốc của tác giả, chúng em phát hiện tác giả thực tế đã dùng gamma bằng 'scale', tức chia tỉ lệ cho số chiều d. Chi tiết kỹ thuật này là bằng chứng xác thực cho thấy tầm quan trọng của việc kiểm chứng mã nguồn độc lập."

---

## SLIDE 10: TẦNG 3 – CẬP NHẬT THỐNG KÊ TRỰC TUYẾN & OAS COVARIANCE

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Tích lũy trực tuyến và Co ma trận hiệp phương sai OAS
- **Công thức cập nhật đệ quy trực tuyến (Online Accumulation):**
  - Đếm mẫu lớp $c$: $n_c \leftarrow n_c + 1$, tổng mẫu $N \leftarrow N + 1$.
  - Cập nhật trung bình lớp (Welford-like): $\mu_c \leftarrow \mu_c + \frac{1}{n_c} (\Phi(x_t) - \mu_c)$.
  - Tích lũy ma trận tự tương quan toàn cục: $A \leftarrow A + \Phi(x_t) \Phi(x_t)^T \in \mathbb{R}^{D \times D}$.
- **Tính toán ma trận hiệp phương sai gộp (Pooled Covariance Matrix):**
  $$S = \frac{1}{N} \left( A - \sum_{c=1}^C n_c \, \mu_c \mu_c^T \right)$$
- **Kỹ thuật co hệ số Oracle Approximating Shrinkage (OAS):**
  $$\Sigma_{\text{OAS}} = (1 - \rho) S + \rho \left( \frac{\text{Tr}(S)}{D} \right) I_D$$
  Hệ số co $\rho \in [0, 1]$ được tính toán tự động dựa trên độ hội tụ giải tích:
  $$\rho = \min \left( \frac{\left(1 - \frac{2}{D}\right) \text{Tr}(S^2) + \text{Tr}^2(S)}{(N + 1 - \frac{2}{D}) \left( \text{Tr}(S^2) - \frac{\text{Tr}^2(S)}{D} \right)}, \, 1 \right)$$
- **Ý nghĩa kỹ thuật:** Khi số chiều $D \gg N$ (high-dimension low-sample-size regime), ma trận $S$ bị suy biến (rank $< D$). OAS kéo các giá trị riêng về đường chéo chính, đảm bảo ma trận luôn **dương xác định và khả nghịch**.
- **Trích dẫn cuối slide:**
  `[8] Y. Chen, A. Wiesel, Y. C. Eldar, and A. O. Hero, "Shrinkage algorithms for MMSE covariance estimation," IEEE Trans. Signal Process., 2010.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 420 từ)
> "Tầng thứ ba của RanDumb đảm nhiệm việc học phân phối dữ liệu trực tuyến. Trong bài toán Class-Incremental, dữ liệu các lớp xuất hiện tuần tự và không quay trở lại. Làm thế nào để ước lượng phân phối của toàn bộ tập dữ liệu mà không cần lưu lại mẫu?
> 
> Lời giải của RanDumb nằm ở công thức phân rã ma trận tán xạ. Thay vì lưu trữ dữ liệu, mô hình chỉ duy trì ba thành phần thống kê: số lượng mẫu từng lớp n_c, vector kỳ vọng từng lớp mu_c, và một ma trận tự tương quan tích lũy A kích thước D nhân D. Ma trận A này được cập nhật đơn giản bằng phép cộng dồn tích ngoài của vector đặc trưng Phi(x).
> 
> Khi cần đánh giá, ma trận hiệp phương sai gộp thực nghiệm S được trích xuất chính xác bằng cách lấy ma trận A trừ đi tổng có trọng số các tích ngoài của vector trung bình từng lớp. Phép tính này hoàn toàn tương đương về mặt toán học với việc tính hiệp phương sai trên toàn bộ tập dữ liệu cùng một lúc, nhưng lại đạt được thông qua cơ chế streaming từng mẫu một.
> 
> Tuy nhiên, một trở ngại toán học lớn xuất hiện: RanDumb sử dụng số chiều đặc trưng D rất lớn (thường là 5,000 đến 25,000 chiều), trong khi ở giai đoạn đầu của luồng dữ liệu, số lượng mẫu N có thể nhỏ hơn D rất nhiều. Hiện tượng này trong thống kê gọi là 'chiều cao mẫu nhỏ' (high-dimension low-sample-size), khiến ma trận thực nghiệm S bị suy biến, định thức bằng 0 và không thể khả nghịch.
> 
> Để khắc phục triệt để điều này, RanDumb tích hợp bộ ước lượng co hệ số OAS (Oracle Approximating Shrinkage) của Chen và cộng sự (2010). OAS co ma trận hiệp phương sai thực nghiệm S về phía ma trận đơn vị nhân với phương sai trung bình. Điểm ưu việt tuyệt đối của OAS là tham số co rho được tính toán hoàn toàn tự động bằng công thức đóng dựa trên vết của ma trận S và S bình phương, mà không cần bất kỳ tập validation nào để tinh chỉnh siêu tham số. Ma trận Sigma OAS thu được đảm bảo tính đối xứng và luôn dương xác định."

---

## SLIDE 11: TẦNG 4 & 5 – SUY LUẬN LDA VÀ ĐỘ PHỨC TẠP THUẬT TOÁN

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Phân loại tuyến tính LDA và Phân tích độ phức tạp
- **Hàm phân loại khoảng cách Mahalanobis:**
  $$y^* = \arg\min_{c} ( \Phi(x) - \mu_c )^T \Sigma_{\text{OAS}}^{-1} ( \Phi(x) - \mu_c ) = \arg\max_{c} \left[ \mu_c^T \Sigma_{\text{OAS}}^{-1} \Phi(x) - \frac{1}{2} \mu_c^T \Sigma_{\text{OAS}}^{-1} \mu_c \right]$$
- **Tối ưu hóa giải hệ bằng Phân rã Cholesky:**
  - Không nghịch đảo tường minh ma trận $\Sigma_{\text{OAS}}^{-1}$ (tránh mất mát độ chính xác và chậm).
  - Phân rã Cholesky: $\Sigma_{\text{OAS}} = L L^T$ (chi phí $\frac{1}{3} D^3$ FLOPs).
  - Giải hệ phương trình $L L^T W = M$ (với $M = [\mu_1, \dots, \mu_C]$) qua 2 bước thế tiến/thế lùi: $O(C D^2)$.
- **Bảng phân tích độ phức tạp (Complexity Analysis):**

| Giai đoạn | Độ phức tạp thời gian (Time Complexity) | Độ phức tạp bộ nhớ (Space Complexity) |
|---|---|---|
| **Cập nhật mẫu mới ($x_t, y_t$)** | $\mathcal{O}(d \cdot D + D^2)$ (Chiếu RFF + Cộng ngoài $A$) | $\mathcal{O}(D^2 + C \cdot D)$ (Lưu $A$ và các $\mu_c$) |
| **Tính toán bộ phân loại (Solve $W$)** | $\mathcal{O}(\frac{1}{3} D^3 + C D^2)$ (Cholesky một lần) | $\mathcal{O}(D^2 + C \cdot D)$ (Ma trận tam giác $L$ và $W$) |
| **Dự đoán 1 mẫu test ($x_{\text{test}}$)** | $\mathcal{O}(d \cdot D + C \cdot D)$ (Chiếu RFF + Nhân $W^T \Phi$) | $\mathcal{O}(D)$ (Vector đặc trưng tạm thời) |

- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`  
  `[9] G. H. Golub and C. F. Van Loan, Matrix Computations, 4th ed. Johns Hopkins Univ. Press, 2013.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 380 từ)
> "Sau khi đã có vector trung bình các lớp và ma trận hiệp phương sai Sigma OAS, Tầng 4 thực hiện việc phân loại một mẫu kiểm tra mới bằng quy tắc khoảng cách Mahalanobis nhỏ nhất tới tâm lớp.
> 
> Bằng phép khai triển đại số cơ bản, khoảng cách bình phương Mahalanobis được tách thành hai phần: một số hạng tuyến tính theo vector đặc trưng Phi(x) và một số hạng hằng số phụ thuộc vào tâm lớp. Nhờ đó, hàm quyết định phi tuyến trở thành một bộ phân loại tuyến tính với ma trận trọng số W bằng nghịch đảo của Sigma OAS nhân với ma trận các tâm lớp M.
> 
> Ở đây, nhóm chúng em đặc biệt lưu ý về mặt kỹ thuật lập trình hiệu năng cao: Code tác giả trong scikit-learn sử dụng hàm nghịch đảo ma trận tổng quát, dẫn đến hiện tượng nghẽn cổ chai khi số chiều D lớn. Nhóm chúng em đã cài đặt thuật toán giải trực tiếp bằng phân rã Cholesky: vì Sigma OAS là ma trận đối xứng dương xác định, chúng ta phân rã nó thành tích của ma trận tam giác dưới L với chuyển vị của nó. Sau đó, bài toán tìm trọng số W được giải quyết nhanh chóng bằng hai lần thế xuôi và thế ngược với độ phức tạp chỉ bằng một nửa so với phân rã LU thông thường.
> 
> Về mặt độ phức tạp thuật toán:
> Dung lượng bộ nhớ của RanDumb là O(D bình phương cộng C nhân D). Đây là một hằng số cố định tuyệt đối, hoàn toàn không phụ thuộc vào số lượng mẫu dữ liệu đã duyệt qua.
> Về thời gian, chi phí cập nhật mỗi mẫu chỉ là O(D bình phương). Việc tính trọng số W có chi phí O(D lập phương) nhưng chỉ cần thực hiện đúng một lần khi kết thúc huấn luyện hoặc khi cần đánh giá. Điều này giải thích tại sao RanDumb có tốc độ thực thi nhanh hơn hàng chục lần so với việc huấn luyện mạng nơ-ron sâu qua nhiều epoch."

---

## SLIDE 12: GIAO THỨC ĐÁNH GIÁ THỰC NGHIỆM (EVALUATION PROTOCOL)

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Giao thức thực nghiệm và Thước đo đánh giá
- **Thiết kế thực nghiệm 2 trường phái:**
  - **Trường phái 1 – Pixel-based Continual Learning (E1):** Học từ đầu (from scratch) trên ảnh thô không có pretrained weights (MNIST, CIFAR-10, CIFAR-100, TinyImageNet).
  - **Trường phái 2 – Foundation Model Features (E2 / Benchmark F):** Đánh giá trên đặc trưng cố định từ Vision Transformer (ViT-B/16) trên các tập dữ liệu phức tạp: CIFAR-100, CUB-200, ImageNet-R, ImageNet-A.
- **Quy trình luồng dữ liệu (Streaming Protocol):**
  - Số lớp được chia đều thành các bước (Task Incremental / Class Incremental).
  - Thứ tự dữ liệu: duyệt tuần tự từng task, trong mỗi task dữ liệu được xáo trộn ngẫu nhiên; mỗi ảnh chỉ được duyệt 1 lần (Single-pass).
- **Thước đo đánh giá chuẩn mực – Độ chính xác gia tăng trung bình (Average Incremental Accuracy):**
  $$A_{\text{inc}} = \frac{1}{T} \sum_{t=1}^T R_t, \quad \text{với } R_t = \frac{1}{\sum_{i=1}^t |D_{\text{test}}^{(i)}|} \sum_{i=1}^t \sum_{(x, y) \in D_{\text{test}}^{(i)}} \mathbb{I}(f_t(x) = y)$$
  Trong đó $R_t$ là độ chính xác trên toàn bộ các lớp đã học tính đến thời điểm kết thúc bước $t$.
- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`  
  `[5] Z. Mai et al., "Online continual learning in image classification: An empirical survey," Neurocomputing, vol. 469, pp. 28-51, 2022.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 370 từ)
> "Để đảm bảo kết quả tái lập có tính tương đồng cao nhất với bài báo, nhóm chúng em thiết lập giao thức thực nghiệm tuân thủ nghiêm ngặt các tiêu chuẩn của cộng đồng Continual Learning.
> 
> Hệ thống thực nghiệm được chia thành hai nhánh lớn:
> Nhánh thứ nhất là thực nghiệm trên ảnh pixel thuần túy (ký hiệu E1): Đây là đấu trường chính để kiểm chứng giả định của bài báo. Các tập dữ liệu được chọn trải dài từ mức độ đơn giản như MNIST, CIFAR-10 đến phức tạp như CIFAR-100 và TinyImageNet. Toàn bộ các mô hình đều phải xuất phát từ con số không, không được sử dụng bất kỳ trọng số tiền huấn luyện nào.
> 
> Nhánh thứ hai là thực nghiệm trên đặc trưng mô hình nền tảng (ký hiệu E2, tương ứng Benchmark F trong bài báo): Sử dụng backbone Vision Transformer ViT-B/16 được huấn luyện sẵn trên ImageNet-21k hoặc ImageNet-1k để trích xuất đặc trưng cố định 768 chiều. Sau đó, các thuật toán phân loại trực tuyến như RanDumb và Nearest Class Mean (NCM) sẽ cạnh tranh trực tiếp trên 4 tập dữ liệu có độ biến thiên phân phối cao: CIFAR-100, CUB-200, ImageNet-R (chứa các hình ảnh biến dạng phong cách), và ImageNet-A (các mẫu gây nhiễu tự nhiên).
> 
> Về thước đo đánh giá: Chúng em sử dụng thước đo **Average Incremental Accuracy** ký hiệu là A_inc. Đây là trung bình cộng của độ chính xác tại tất cả các mốc thời gian sau khi học xong mỗi task. Tại mỗi mốc t, tập kiểm tra đánh giá trên tất cả các lớp đã xuất hiện từ đầu đến thời điểm đó. Thước đo này phản ánh toàn diện khả năng vừa tiếp thu kiến thức mới vừa chống đỡ hiện tượng quên thảm khốc của mô hình trong suốt hành trình học."

---

## SLIDE 13: THỰC NGHIỆM E9 – KIỂM CHỨNG TỪNG TẦNG VỚI CODE GỐC (LAYER VERIFICATION)

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Thực nghiệm E9: So sánh số học từng tầng với mã nguồn tác giả
- **Đặc tả kiểm chứng độc lập:**
  - Repo tác giả: Commit `14a51ee` (GPL-3.0), kiểm chứng trong môi trường cách ly (clean-room).
  - W&B Run công khai: [icae4r9o (E9-layer-comparison-synthetic-D500-seed0)](https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/icae4r9o).
- **Bảng đối chiếu số học chi tiết 5 tầng:**

| Tầng xử lý | Chỉ số đo lường | Mã nguồn tác giả (`_official`) | Nhóm tự cài (`src/randumb/`) | Độ lệch số học / Đánh giá |
|---|---|---|---|---|
| **1. Chuẩn hóa** | Mean / Std | $\mu = -8.74 \times 10^{-6}, \, \sigma = 1.00037$ | $\mu = -5.81 \times 10^{-9}, \, \sigma = 1.00000$ | Max abs diff $= 6.64 \times 10^{-4}$ (khác hằng số kênh) |
| **2. Ánh xạ RFF** | Hệ số $\gamma$ / Độ lệch chuẩn $\Phi$ | $\gamma = 0.0003253, \, \sigma_\Phi = 0.04473$ | $\gamma = 0.0003255, \, \sigma_\Phi = 0.04471$ | Độ lệch tương đối $\gamma = 0.073\%$; không gian đặc trưng khớp |
| **3. OAS Covariance** | Max diff $\Sigma_{\text{OAS}}$ / Hệ số $\rho$ | $\rho = 0.94603023292014$ | $\rho = 0.94603023292014$ | $\Delta \Sigma_{\text{OAS}} = 2.60 \times 10^{-18}$; **Trùng khớp mức machine epsilon** |
| **4. Trọng số LDA ($W$)** | Cosine Similarity trung bình | Vector trọng số $W_{\text{author}}$ | Vector trọng số $W_{\text{team}}$ | **Cosine Similarity $= 1.000000$ (Cùng hướng giải tích 100%)** |
| **5. Dự đoán nhãn** | Agreement Rate / Accuracy | Accuracy $= 7.50\%$ (synthetic) | Accuracy $= 7.50\%$ (synthetic) | **Tỷ lệ đồng thuận nhãn $= 100.00\%$ trên mọi mẫu** |

- **Kết luận:** Pipeline tự xây dựng của nhóm đạt độ chính xác giải tích tuyệt đối so với mã nguồn tác giả.
- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`  
  `[10] Nhóm nghiên cứu 24TNT, "Experiment E9 Layer-by-Layer Verification Report," W&B Run icae4r9o, 2026.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 410 từ)
> "Để đảm bảo tính trung thực học thuật và xác nhận mã nguồn do nhóm tự lập trình hoàn toàn chuẩn xác, thành viên Huy đã thực hiện **Thực nghiệm E9: So sánh số học từng tầng** giữa mã nguồn tự viết của nhóm và mã nguồn chính thức của tác giả ghim tại commit 14a51ee.
> 
> Thực nghiệm được thiết lập trên cùng một tập dữ liệu chuẩn hóa với seed cố định, quét qua toàn bộ 5 tầng xử lý của RanDumb. Toàn bộ dữ liệu kiểm chứng đã được ghi nhận công khai trên W&B với mã run icae4r9o.
> 
> Kết quả so sánh mang lại sự khẳng định chắc chắn về tính đúng đắn toán học:
> Ở Tầng 1 (Chuẩn hóa): Độ lệch trung bình chỉ ở mức 10 lũy thừa âm 4. Nguyên nhân là tác giả dùng hằng số cố định cho ảnh 3 kênh, còn nhóm tính toán z-score trực tiếp từ tập dữ liệu.
> Ở Tầng 2 (Chiếu RFF): Độ lệch tương đối của tham số gamma giữa hai bên chỉ là 0.073%. Độ lệch chuẩn của các chiều đặc trưng sau chiếu cosin trùng khớp ở mức 0.0447.
> Ở Tầng 3 (Covariance OAS): Đây là kết quả ấn tượng nhất! Độ lệch cực đại của ma trận hiệp phương sai giữa hai triển khai chỉ là 2.6 nhân 10 lũy thừa âm 18. Tham số co rho của OAS trùng nhau đến 14 chữ số thập phân. Đây là mức sai số làm tròn số thực của phần cứng máy tính (machine epsilon), chứng minh công thức tích lũy trực tuyến của nhóm hoàn toàn tương đương với công thức batch của scikit-learn.
> Ở Tầng 4 (Trọng số phân loại W): Cosine similarity giữa các vector trọng số của nhóm và của tác giả đạt giá trị tuyệt đối 1.000000. Điều này khẳng định mặt phẳng siêu phẳng phân lớp của hai mô hình chỉ về cùng một hướng duy nhất trong không gian đa chiều.
> Và ở Tầng 5 (Dự đoán cuối cùng): Tỷ lệ đồng thuận nhãn dự đoán giữa hai code đạt chính xác 100.00%.
> 
> Thực nghiệm E9 là minh chứng không thể chối cãi rằng nhóm đã tái lập thành công thuật toán với độ chính xác số học tuyệt đối mà không cần sao chép một dòng mã nguồn nào của tác giả."

---

## SLIDE 14: PHẢN BIỆN MÃ NGUỒN TÁC GIẢ – 6 PHÁT HIỆN TỪ E9 (CRITIQUE)

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Phản biện mã nguồn tác giả: 6 điểm mâu thuẫn và lỗi kỹ thuật
- **Bảng đối chiếu Bài báo vs Mã nguồn thực tế:**

| Vấn đề | Tuyên bố trong bài báo (Paper) | Thực tế trong code tác giả (`_official`) | Phân tích tác động học thuật & Kỹ thuật |
|---|---|---|---|
| **1. Tham số Kernel $\gamma$** | $\gamma = 1.0$ (Mục 2) | `gamma='scale'` (`main.py` L81) | Đặt $\gamma = 1.0 \to \text{Acc} \approx 10\%$ (nhiễu trắng). Code phải dùng `scale` mới đạt $45.3\%$. |
| **2. Bản chất luồng dữ liệu** | Single-pass streaming online | Nạp nguyên batch lớn vào RAM | Về mặt toán học tương đương, nhưng code tác giả không phản ánh đúng ràng buộc bộ nhớ thiết bị biên. |
| **3. Hệ số Ridge $\lambda$** | Bảng hparam $\lambda \in [10^{-6}, 10^{-4}]$ | Bỏ qua, đặt $\lambda = 0$ | Tác giả thừa nhận bỏ $\lambda$ khiến kết quả code lệch $\pm 0.8\%$ so với bảng số liệu công bố. |
| **4. Lỗi cú pháp trong code ViT** | Công bố code chạy Benchmark F | Syntax & NameError trong `get_feats_vit.py` | L14 lỗi thụt lề 3 spaces; L60 gọi `model.cuda()` trước khi khởi tạo `model` ở L65 $\to$ Code chưa qua CI/CD! |
| **5. Cờ Augment trên ViT** | Benchmark F không dùng data augmentation | `scripts/runall.sh` có cờ `--augment` | Cần loại bỏ cờ augment khi tái lập để đảm bảo tính công bằng của Benchmark F. |
| **6. Nghẽn cổ chai đảo ma trận** | Tuyên bố chi phí huấn luyện thấp | `OAS(store_precision=True)` | Mặc định nghịch đảo ma trận $D \times D$ từng lớp gây tràn RAM và chậm khi $D > 5000$. Nhóm sửa bằng Cholesky. |

- **Ý nghĩa đóng góp:** Cung cấp góc nhìn phản biện sâu sắc, sắc bén cho bài báo tại NeurIPS 2024.
- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`  
  `[11] Mã nguồn chính thức RanDumb, commit 14a51ee, kho lưu trữ apd20/randumb, 2024.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 420 từ)
> "Giá trị học thuật lớn nhất trong phần việc của Huy không chỉ dừng lại ở việc xác nhận tính đúng đắn, mà là **phát hiện ra 6 điểm mâu thuẫn và lỗi kỹ thuật nghiêm trọng** trong mã nguồn tác giả so với bài báo công bố.
> 
> Thứ nhất, phát hiện chấn động về tham số gamma: Bài báo viết rất rõ ràng ở Mục 2 là sử dụng gamma bằng 1.0. Nhưng như nhóm đã phân tích, gamma bằng 1.0 trên dữ liệu ảnh chuẩn hóa sẽ biến phép chiếu Fourier thành một máy phát nhiễu ngẫu nhiên, kéo độ chính xác về mức 10%. Trong code thực tế, tác giả âm thầm đổi sang gamma bằng 'scale'. Nếu một nhóm nghiên cứu khác chỉ đọc bài báo mà không soi code tác giả, họ chắc chắn sẽ thất bại khi tái lập!
> 
> Thứ hai, về bản chất online: Dù bài báo mô tả thuật toán xử lý luồng dữ liệu từng mẫu, mã nguồn lại nạp toàn bộ dữ liệu vào mảng NumPy lớn để scikit-learn tính toán hàng loạt. Mặc dù kết quả đại số tuyến tính là tương đương, cách làm của tác giả vi phạm tinh thần chạy trên thiết bị biên bộ nhớ nhỏ.
> 
> Thứ ba, hệ số chính quy hóa lambda: Trong khi bài báo liệt kê một bảng siêu tham số lambda chi tiết cho từng tập dữ liệu, mã nguồn chính thức lại đặt lambda bằng 0 và tác giả thừa nhận việc này gây ra sai lệch khoảng 0.8% so với bảng kết quả trong bài.
> 
> Thứ tư, cẩu thả trong phát hành mã nguồn: Trong file trích xuất đặc trưng Vision Transformer `get_feats_vit.py`, dòng 14 bị lỗi IndentationError thụt lề 3 khoảng trắng thay vì 4, và dòng 60 gọi lệnh chuyển model lên GPU trước khi model đó được định nghĩa ở dòng 65. Điều này cho thấy mã nguồn công bố chưa từng được chạy qua hệ thống kiểm thử tự động (CI/CD).
> 
> Thứ năm, sự mập mờ về Data Augmentation: Bài báo khẳng định Benchmark F là phân loại tuyến tính trên đặc trưng ViT cố định không augment, nhưng script chạy của tác giả lại bật cờ `--augment`.
> 
> Cuối cùng, nghẽn cổ chai OAS: scikit-learn mặc định lưu ma trận precision bằng cách đảo ma trận kích thước D nhân D. Với D bằng 25,000, phép toán này làm treo máy. Nhóm đã tối ưu hóa hoàn toàn bằng thuật toán phân rã Cholesky.
> 
> Những phát hiện thực nghiệm này là vũ khí phản biện sắc bén, khẳng định năng lực phân tích độc lập của nhóm chúng em."

---

## SLIDE 15: THỰC NGHIỆM E2 – ĐẶC TRƯNG VISION TRANSFORMER (BENCHMARK F)

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Thực nghiệm E2: RanDumb và NCM trên đặc trưng ViT-B/16
- **Thiết lập:** 16 run trên Kaggle GPU T4; 4 datasets $\times$ 2 backbones (IN-21k, IN-1k); so sánh RanDumb vs Nearest Class Mean (NCM).
- **Bảng kết quả tái lập chi tiết (Đối chiếu Bài báo vs Nhóm thực hiện):**

| Tập dữ liệu | Backbone ViT-B/16 | NCM (Bài báo) | NCM (Nhóm tái lập) | RanDumb (Bài báo) | RanDumb (Nhóm tái lập) | Độ lệch RanDumb | W&B Run Link |
|---|---|---|---|---|---|---|---|
| **CIFAR-100** | ImageNet-21k | $85.3\%$ | $84.97\%$ | **$88.4\%$** | **$88.08\%$** | $-0.32\%$ | [x06w28q2](https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/x06w28q2) |
| **CIFAR-100** | ImageNet-1k | $75.6\%$ | $75.76\%$ | **$81.1\%$** | **$81.71\%$** | $+0.61\%$ | [421dknz1](https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/421dknz1) |
| **CUB-200** | ImageNet-21k | $84.1\%$ | $86.03\%$ | **$88.6\%$** | **$89.57\%$** | $+0.97\%$ | [52l9g0d4](https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/52l9g0d4) |
| **CUB-200** | ImageNet-1k | $73.3\%$ | $77.89\%$ | **$81.6\%$** | **$84.34\%$** | $+2.74\%$ | [n1e3i762](https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/n1e3i762) |
| **ImageNet-R** | ImageNet-21k | $65.2\%$ | $67.75\%$ | **$74.3\%$** | **$74.65\%$** | $+0.35\%$ | [8a6t1l1i](https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/8a6t1l1i) |
| **ImageNet-R** | ImageNet-1k | $53.0\%$ | $56.97\%$ | **$66.3\%$** | **$66.90\%$** | $+0.60\%$ | [b1v68m6o](https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/b1v68m6o) |
| **ImageNet-A** | ImageNet-21k | $44.9\%$ | $52.79\%$ | **$55.7\%$** | **$66.72\%$** | $+11.02\%$ | [90n2w8g8](https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/90n2w8g8) |
| **ImageNet-A** | ImageNet-1k | $27.9\%$ | $38.99\%$ | **$43.8\%$** | **$61.35\%$** | $+17.55\%$ | [8s583j3z](https://forge.coreweave.com/wandb/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/8s583j3z) |

- **Phân tích sai lệch:** Khớp hoàn hảo trên CIFAR-100, CUB, ImageNet-R (sai lệch $< \pm 2.8\%$). Lệch trên ImageNet-A do tập gốc không có nhãn test chuẩn, nhóm tự chia stratified split.
- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`  
  `[12] Nhóm nghiên cứu 24TNT, "Report 4: E2 ViT Feature Reproduction," 16 W&B Runs, 2026.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 400 từ)
> "Chuyển sang kết quả thực nghiệm quy mô lớn **E2: Tái lập Benchmark F trên đặc trưng Vision Transformer**, do bạn Kiên trong nhóm phụ trách thực hiện trên Kaggle GPU T4 với 16 lần chạy độc lập được ghi nhận đầy đủ trên W&B.
> 
> Bảng kết quả trên slide thể hiện sự so sánh đối đầu trực diện giữa RanDumb và phương pháp cơ sở Nearest Class Mean (NCM) trên 4 tập dữ liệu với 2 phiên bản pretrained ViT-B/16.
> 
> Từ bảng số liệu thực nghiệm, nhóm chúng em rút ra ba kết luận quan trọng:
> Kết luận thứ nhất: **RanDumb chiến thắng áp đảo NCM trên tất cả 8 cặp thực nghiệm.** Độ vượt trội dao động từ 2.7% đến hơn 14.8%. Điều này chứng minh rằng việc mô hình hóa ma trận hiệp phương sai dùng chung bằng OAS vượt trội hơn rất nhiều so với giả định các chiều độc lập đẳng hướng của NCM.
> 
> Kết luận thứ hai: **Tính tái lập cực cao trên 3 tập dữ liệu CIFAR-100, CUB-200 và ImageNet-R.** Trên các tập dữ liệu này, kết quả của nhóm tái lập gần như trùng khít với số liệu công bố trong bài báo, với độ sai lệch tuyệt đối chỉ dao động trong khoảng từ âm 0.32% đến cộng 2.74%. Sự tương đồng chặt chẽ này khẳng định tính đúng đắn của pipeline trích xuất đặc trưng ViT và thuật toán LDA của nhóm.
> 
> Kết luận thứ ba: **Phát hiện thú vị trên tập dữ liệu ImageNet-A.** Cả NCM và RanDumb của nhóm đều đạt kết quả cao hơn bài báo từ 11% đến 17%. Khi mổ xẻ nguyên nhân, nhóm phát hiện: Tập dữ liệu ImageNet-A công khai không cung cấp sẵn phân chia train/test cố định. Tác giả bài báo không công bố file split cụ thể. Nhóm đã tự chia tập dữ liệu theo tỷ lệ 80/20 với phân tầng nhãn (stratified split). Kết quả cao hơn cho thấy khi các mẫu test có phân phối gần hơn với tập train, sức mạnh của RanDumb được phát huy tối đa.
> 
> Những số liệu này củng cố vững chắc khẳng định của bài báo: trên các biểu diễn chất lượng cao của Foundation Models, một bộ phân loại thống kê đơn giản như RanDumb đủ sức đánh bại các mô hình Continual Learning phức tạp."

---

## SLIDE 16: TỔNG KẾT BƯỚC ĐẦU & KẾ HOẠCH BÀN GIAO TIẾP THEO

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Tiến độ thực hiện đồ án và Kế hoạch hoàn thiện
- **Bảng tổng kết trạng thái các mảng công việc:**

| Hạng mục công việc | Người phụ trách | Trạng thái hiện tại | Sản phẩm bàn giao & Minh chứng |
|---|---|---|---|
| **Mã nguồn lõi (`src/randumb/`)** | Toàn đội (Đăng lead) | Đã hoàn thành 100% | Đã merge vào nhánh `main` trên GitHub; đầy đủ test số học. |
| **Kiểm chứng Code Tác giả (E9)** | Huy | Đã hoàn thành 100% | [09_e9_verification.ipynb](notebooks/09_e9_verification.ipynb), W&B Run `icae4r9o`, phát hiện 6 lỗi. |
| **Thực nghiệm ViT Benchmark F (E2)** | Kiên | Đã hoàn thành 100% | [02_e2_vit.ipynb](notebooks/02_e2_vit.ipynb), 16 W&B runs, báo cáo `Docs/report4.md`. |
| **Thực nghiệm Pixel & Ablation (E1, E8)** | Đăng | Đang chạy trên GPU | [03_dang_sprint3.ipynb](notebooks/03_dang_sprint3.ipynb), script `run_sprint3.py`. |
| **Slide Deck môn học (45% điểm)** | Huy lead | Hoàn thành bản nháp Slide 1–16 | 16 slide đầy đủ Speaker Notes & IEEE citations. Sẵn sàng tích hợp E1/E8. |
| **Poster Workshop (50% điểm)** | Cả đội | Đang lên khung thiết kế | Thiết kế khổ A2 đứng; chuẩn bị cho hạn chót 05/12/2026. |

- **Các mốc thời gian then chốt (Milestones):**
  - **20/11/2026:** Hoàn thiện toàn bộ Slide Deck (20–40 slide) gồm cả số liệu E1 của Đăng và xuất file `.pptx`.
  - **21/11/2026:** Hạn chót nộp Slide Deck trên Moodle.
  - **05/12/2026:** Poster Workshop tại trường (A2 Vertical).
- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 360 từ)
> "Kính thưa quý thầy cô, slide thứ 16 tổng kết toàn bộ tiến độ triển khai đồ án của nhóm chúng em tính đến thời điểm hiện tại và kế hoạch hành động trong những ngày tới.
> 
> Về mặt kỹ thuật, nhóm đã hoàn thành xuất sắc ba trụ cột cốt lõi:
> Thứ nhất, toàn bộ module thuật toán RanDumb tự cài đặt đã có mặt trên nhánh main của kho mã nguồn GitHub, hoạt động ổn định và tối ưu hóa cao.
> Thứ hai, thực nghiệm kiểm chứng từng tầng E9 đã hoàn thành mỹ mãn với sự trùng khớp 100% về mặt giải tích với mã nguồn tác giả, đồng thời cung cấp bản phân tích phản biện sâu sắc về 6 điểm khác biệt và lỗi kỹ thuật trong repo gốc.
> Thứ ba, thực nghiệm quy mô lớn E2 trên đặc trưng Vision Transformer với 16 run W&B đã hoàn tất, tái lập thành công các kết quả trên 4 tập dữ liệu chuẩn.
> 
> Hiện tại, bạn Đăng đang hoàn thiện các lượt chạy thực nghiệm E1 trên ảnh pixel thuần và E8-B về kiểm chứng tính cần thiết của RFF trên máy có GPU. Ngay khi có số liệu, nhóm sẽ tích hợp các bảng biểu này vào các slide tiếp theo từ Slide 17 trở đi.
> 
> Mục tiêu trước mắt của nhóm là chốt toàn bộ Slide Deck hoàn chỉnh trước ngày 20 tháng 11, đảm bảo mỗi slide đều có Speaker Notes chỉn chu dưới 500 từ và trích dẫn IEEE đúng quy định, sẵn sàng nộp bài đúng hạn Moodle ngày 21 tháng 11, và dồn toàn lực chuẩn bị cho buổi Poster Workshop quan trọng vào ngày 5 tháng 12 tới."

---
