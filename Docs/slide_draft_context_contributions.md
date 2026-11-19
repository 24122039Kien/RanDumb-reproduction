# Bản nháp Slide Deck: Phần Bối cảnh & Đóng góp (Slide 1 – 6)

**Tác giả phụ trách:** Huy  
**Mục tiêu:** Hoàn thiện 6 slide đầu tiên cho Slide Deck nộp môn Nhập môn Học máy (chiếm 45% điểm đồ án).  
**Quy cách bắt buộc của môn học:**
- Mỗi slide gồm 2 phần tách biệt: **Mặt slide** (trực quan, sơ đồ/hình tự vẽ, bảng so sánh, từ khóa cô đọng; KHÔNG viết đoạn văn dài) và **Speaker Notes** (lời giảng giải chi tiết như khi thuyết trình, tối đa 500 từ/slide).
- **Trích dẫn chuẩn IEEE** đánh số đặt ở cuối chính slide đó. Không có slide References riêng.
- Toàn bộ hình ảnh, sơ đồ phải do nhóm tự vẽ. Không dán hình chụp từ bài báo.

---

## SLIDE 1: BÀI TOÁN HỌC LIÊN TỤC TRỰC TUYẾN (ONLINE CONTINUAL LEARNING)

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Học máy liên tục trực tuyến trong điều kiện biên khắt khe
- **Sơ đồ luồng dữ liệu (Tự vẽ):**
  ```
  Luồng dữ liệu (Data Stream) theo thời gian:
  [Mẫu x_1, y_1] → [Mẫu x_2, y_2] → ... → [Mẫu x_t, y_t] → ...
         ↓                ↓                       ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Mô hình xử lý trực tuyến (Single-Pass, Online Processing)  │
  │  - Xem mỗi mẫu duy nhất 1 lần                               │
  │  - Không lưu trữ mẫu cũ vào bộ nhớ đệm (Zero-Memory Replay) │
  │  - Lớp dữ liệu xuất hiện tuần tự (Class-Incremental)        │
  └─────────────────────────────────────────────────────────────┘
  ```
- **Ví dụ thực tế của nhóm (Edge AI / IoT):**
  - **Hệ thống camera kiểm tra lỗi bo mạch điện tử (PCB inspection) trên băng chuyền sản xuất tự động.**
  - *Đặc điểm:* Thiết bị gắn tại xưởng có bộ nhớ RAM siêu nhỏ (vi xử lý nhúng); dây chuyền chạy liên tục 24/7; các linh kiện và loại lỗi mới phát sinh tuần tự theo từng lô sản xuất.
  - *Ràng buộc:* Cấm lưu trữ ảnh chụp vi mạch cũ vì lý do bảo mật sở hữu trí tuệ công nghệ; không thể tạm dừng dây chuyền để train lại từ đầu.
- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 380 từ)
> "Kính thưa quý thầy cô và các bạn, bài toán mà nhóm chúng em nghiên cứu là **Online Continual Learning (Học liên tục trực tuyến)** trong kịch bản tăng dần lớp (class-incremental). 
> 
> Trong kịch bản thực tế này, hệ thống nhận dữ liệu dưới dạng một luồng liên tục theo thời gian. Mô hình chỉ được phép quan sát mỗi điểm dữ liệu đúng một lần duy nhất (single-pass) và phải cập nhật tri thức ngay lập tức mà không được lưu trữ dữ liệu cũ vào bộ nhớ đệm (không có replay buffer). Thách thức cốt lõi của bài toán là hiện tượng **quên thảm khốc (catastrophic forgetting)**: khi học các lớp dữ liệu mới, mạng nơ-ron sâu có xu hướng ghi đè lên các trọng số cũ, làm suy giảm nghiêm trọng độ chính xác trên các lớp đã học trước đó.
> 
> Để minh họa rõ ràng tính cấp thiết và các ràng buộc ngặt nghèo của bài toán, nhóm chúng em đưa ra một ví dụ thực tế: **Hệ thống camera quang học tự động kiểm tra lỗi bề mặt linh kiện bo mạch điện tử (PCB)** đặt ngay tại dây chuyền nhà máy. Trong bối cảnh này:
> Thứ nhất, dây chuyền sản xuất diễn ra liên tục, mỗi giây có hàng chục sản phẩm đi qua, camera phải chụp và xử lý ngay tại thiết bị biên (Edge computing) vốn có dung lượng RAM và bộ nhớ cực kỳ hạn chế.
> Thứ hai, theo chu kỳ công nghệ, các vi mạch mới hoặc dạng lỗi linh kiện mới liên tục xuất hiện theo thời gian.
> Thứ ba, vì lý do bảo mật thiết kế vi mạch độc quyền của đối tác sản xuất, hệ thống bị cấm hoàn toàn việc ghi lại hay lưu trữ hình ảnh sản phẩm vào ổ cứng.
> 
> Một thiết bị như vậy buộc phải học theo cơ chế online continual learning thuần túy: thấy một mẫu, cập nhật tham số ngay, giải phóng bộ nhớ lập tức, và duy trì khả năng phân loại tốt trên mọi loại lỗi từ trước tới nay mà không bao giờ được học lại."

---

## SLIDE 2: GIẢ ĐỊNH TRUNG TÂM BỊ NGHI NGỜ (QUESTIONED ASSUMPTION)

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Giả định trung tâm trong học biểu diễn: Liệu có luôn đúng?
- **Câu hỏi then chốt:**
  > *"Trong Online Continual Learning, biểu diễn học được bằng mạng sâu có thực sự tốt hơn một biểu diễn ngẫu nhiên không hề được học?"*
- **Sơ đồ so sánh hai trường phái:**
  - **Trường phái truyền thống (Continual Representation Learning):**
    `Ảnh thô → [Mạng sâu cập nhật online liên tục (Backprop)] → Classifier`
    *(Niềm tin: Mạng sâu sẽ liên tục thích nghi để tạo ra biểu diễn trừu tượng ngày càng tối ưu).*
  - **Phát hiện của RanDumb (Frozen Random Representation):**
    `Ảnh thô → [Phép chiếu ngẫu nhiên cố định (RFF / Random Projection)] → Classifier`
    *(Thực tế: Biểu diễn ngẫu nhiên cố định không bao giờ bị quên, hoạt động ổn định và vượt trội).*
- **Nghịch lý thực nghiệm:** Càng cố gắng cập nhật trọng số mạng sâu trên luồng dữ liệu 1 lượt, biểu diễn càng bị méo mó và suy giảm hiệu năng do thiếu dữ liệu lặp lại.
- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 390 từ)
> "Trong suốt hơn một thập kỷ phát triển của học sâu (Deep Learning), cộng đồng nghiên cứu luôn vận hành dựa trên một tiên đề gần như mặc định: **Học biểu diễn (Representation Learning) luôn vượt trội hơn các biểu diễn thủ công hoặc ngẫu nhiên.** Khi đối mặt với bài toán Online Continual Learning, phản xạ tự nhiên của hầu hết các nhà nghiên cứu là thiết kế các kiến trúc mạng nơ-ron phức tạp, kết hợp các hàm mất mát chống quên để liên tục tinh chỉnh các tầng trích xuất đặc trưng theo từng batch dữ liệu.
> 
> Tuy nhiên, bài báo **RanDumb** của nhóm tác giả Ameya Prabhu và cộng sự tại NeurIPS 2024 đã dũng cảm đặt ra một câu hỏi phản biện mang tính nền tảng: *Liệu trong điều kiện trực tuyến khắt khe – chỉ xem mẫu một lần, không lưu trữ dữ liệu cũ – biểu diễn mà mạng sâu cố gắng học được có thực sự tốt hơn một biểu diễn ngẫu nhiên cố định không hề qua huấn luyện?*
> 
> Theo các tác giả, giả định 'mạng sâu học biểu diễn tốt hơn' thực chất chưa từng được kiểm chứng thực nghiệm nghiêm ngặt trong bối cảnh online stream. Khi dữ liệu chỉ đi qua một lần, thuật toán lan truyền ngược (backpropagation) không có đủ số bước lặp (iterations) để hội tụ về các biểu diễn tổng quát hóa tốt. Nguy hiểm hơn, việc cập nhật trọng số của mạng sâu liên tục trên các phân phối dữ liệu thay đổi sẽ gây ra hiện tượng méo mó không gian đặc trưng (representation drift).
> 
> Ngược lại, một phép chiếu ngẫu nhiên cố định (như Random Fourier Features) tuy không được tối ưu cho riêng dữ liệu nào, nhưng lại có một ưu điểm tuyệt đối: **nó hoàn toàn miễn nhiễm với hiện tượng quên thảm khốc.** Không gian đặc trưng được giữ bất biến từ đầu đến cuối luồng dữ liệu, cho phép các bộ phân loại tuyến tính cổ điển tích lũy thống kê một cách tối ưu. RanDumb chứng minh rằng: niềm tin bấy lâu nay vào việc học biểu diễn trực tuyến thực chất là một sự ngộ nhận trong các kịch bản benchmark hiện hành."

---

## SLIDE 3: CÁC PHƯƠNG PHÁP TIỀN NHIỆM VÀ HỆ QUẢ THỰC TẾ

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Phân tích các phương pháp tiền nhiệm
- **Bảng so sánh 4 cột (Yêu cầu bắt buộc của môn học):**

| Phương pháp | Ý tưởng chính | Hạn chế cốt lõi | Hệ quả thực tế của hạn chế |
|---|---|---|---|
| **SLDA** *(Hayes & Kanan, 2020) [2]* | Cố định một mạng CNN đã pretrained; chỉ cập nhật trực tuyến vector trung bình lớp $\mu_c$ và ma trận hiệp phương sai dùng chung $\Sigma$. | Phụ thuộc hoàn toàn vào backbone CNN đã được huấn luyện sẵn từ trước trên tập dữ liệu lớn. | **Vô dụng khi phải học từ pixel thuần.** Nếu áp dụng LDA trực tiếp trên pixel thô, hiệu năng rất thấp (trên CIFAR-10 chỉ đạt 41.6% so với 55.6% của RanDumb). |
| **GDumb** *(Prabhu et al., 2020) [3]* | Dùng một bộ nhớ đệm (buffer) lấy mẫu cân bằng giữa các lớp; khi kiểm tra, huấn luyện lại một mạng sâu từ đầu trên buffer này. | Bắt buộc phải lưu trữ lại một lượng lớn mẫu ảnh thô và tốn chi phí tính toán cực lớn để huấn luyện lại từ đầu. | **Vi phạm hoàn toàn ràng buộc bảo mật và tài nguyên.** Không thể triển khai trên phần cứng nhúng (RAM nhỏ) hoặc các ứng dụng nhạy cảm cấm lưu trữ dữ liệu người dùng. |
| **RanPAC** *(McDonnell et al., 2023) [4]* | Chiếu ngẫu nhiên các đặc trưng rút trích từ mô hình Foundation/Pretrained lên không gian nhiều chiều, sau đó áp dụng Ridge Classifier. | Vẫn phụ thuộc chặt chẽ vào biểu diễn chất lượng cao trích xuất từ các pre-trained backbone khổng lồ (ViT, CLIP). | **Không giải quyết được bài toán học liên tục từ đầu (learning from scratch)** khi miền dữ liệu mới hoàn toàn xa lạ với dữ liệu tiền huấn luyện. |

- **Trích dẫn cuối slide:**
  `[2] T. L. Hayes and C. Kanan, "Lifelong machine learning with deep streaming linear discriminant analysis," in CVPRW, 2020.`  
  `[3] A. Prabhu et al., "GDumb: A Simple Approach that Questions Continual Learning," in ECCV, 2020.`  
  `[4] M. D. McDonnell et al., "RanPAC: Random Projections and Pre-trained Representations for Continual Learning," in NeurIPS, 2023.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 410 từ)
> "Để thấy rõ đóng góp của RanDumb, chúng ta cần nhìn lại các nỗ lực nghiên cứu tiền nhiệm trong lĩnh vực Continual Learning:
> 
> Thứ nhất là phương pháp **SLDA (Streaming Linear Discriminant Analysis)** của Hayes và Kanan năm 2020. Ý tưởng của SLDA rất thanh lịch: giữ cố định một mạng CNN đã tiền huấn luyện, và mỗi khi có mẫu mới đến, mô hình chỉ cập nhật trực tuyến vector trung bình của lớp và một ma trận hiệp phương sai chung. Hạn chế chí mạng của SLDA là nó phụ thuộc hoàn toàn vào một backbone trích xuất đặc trưng đã được học sẵn. Hệ quả thực tế là SLDA không thể áp dụng khi chúng ta bắt buộc phải học từ ảnh pixel thô; bài báo RanDumb chứng minh nếu chạy SLDA trực tiếp trên pixel, độ chính xác trên CIFAR-10 chỉ đạt vỏn vẹn 41.6%, kém xa mức 55.6% của RanDumb.
> 
> Thứ hai là **GDumb** do chính tác giả Ameya Prabhu công bố tại ECCV 2020. GDumb đưa ra một giải pháp châm biếm: lưu lại các mẫu vào một bộ nhớ đệm cân bằng lớp, rồi mỗi khi cần đánh giá thì train lại toàn bộ mạng nơ-ron từ đầu. Hạn chế của GDumb là nó đòi hỏi một vùng nhớ đệm lớn và tiêu tốn tài nguyên tính toán khổng lồ để retrain. Hệ quả thực tế là GDumb hoàn toàn bất khả thi trong bài toán kiểm tra bo mạch vi mạch hay bảo mật y tế của nhóm chúng em, nơi dữ liệu cũ tuyệt đối không được phép lưu trữ lại.
> 
> Thứ ba là **RanPAC** (NeurIPS 2023). RanPAC sử dụng phép chiếu ngẫu nhiên (Random Projection), nhưng là chiếu trên các đặc trưng trích từ các mô hình nền tảng khổng lồ (Foundation Models). Hạn chế của RanPAC là nó né tránh bài toán học liên tục thực sự bằng cách dựa dẫm vào tri thức đã học sẵn từ hàng triệu ảnh bên ngoài.
> 
> Tóm lại, các phương pháp tiền nhiệm hoặc phải dựa vào mô hình pretrained cực mạnh, hoặc phải lưu lại dữ liệu mẫu. Trước RanDumb, chưa có một phương pháp nào giải quyết được bài toán học trực tuyến hiệu quả từ pixel thuần mà hoàn toàn không lưu trữ mẫu."

---

## SLIDE 4: KHOẢNG TRỐNG NGHIÊN CỨU DẪN ĐẾN RANDUMB

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Khoảng trống nghiên cứu (Research Gap)
- **Ma trận định vị các phương pháp học trực tuyến:**

```
                  CẦN LƯU MẪU (Replay Buffer)     KHÔNG LƯU MẪU (Zero-Replay)
                ┌───────────────────────────────┬───────────────────────────────┐
DỰA TRÊN        │                               │  • SLDA (Hayes et al., 2020)  │
PRETRAINED      │  • BiC, DER, ER-AML...        │  • RanPAC (McDonnell, 2023)   │
BACKBONE        │                               │                               │
                ├───────────────────────────────┼───────────────────────────────┤
TỰ HỌC TỪ PIXEL │                               │     KHOẢNG TRỐNG LỚN:         │
THUẦN (SCRATCH) │  • GDumb (Prabhu et al., 2020)│    ★ RANDUMB (NeurIPS 2024)   │
                │                               │  (Học 1 lượt, 0 mẫu lưu trữ,  │
                │                               │   RFF cố định + Online LDA)   │
                └───────────────────────────────┴───────────────────────────────┘
```

- **Ba tiêu chí vàng mà RanDumb chinh phục đồng thời:**
  1. *Zero Replay Memory:* Bộ nhớ đệm dữ liệu quá khứ bằng 0 (đáp ứng trọn vẹn quyền riêng tư & bảo mật).
  2. *Single-Pass Streaming:* Dữ liệu trôi qua 1 lần duy nhất, không yêu cầu huấn luyện lặp nhiều epoch.
  3. *No Pre-trained Dependency:* Hoạt động mạnh mẽ ngay từ pixel thô thông qua phép chiếu Fourier ngẫu nhiên.
- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 360 từ)
> "Từ việc phân tích các phương pháp tiền nhiệm, chúng ta nhìn thấy một **khoảng trống nghiên cứu rất rõ ràng** trên bức tranh toàn cảnh của Continual Learning.
> 
> Nếu phân loại theo hai trục: Trục tung là nguồn gốc biểu diễn (Dựa vào mô hình Pretrained vs Tự học từ pixel thuần), và Trục hoành là yêu cầu lưu trữ (Có lưu mẫu vào buffer vs Không lưu mẫu nào):
> - Ở góc trên bên phải, chúng ta có SLDA và RanPAC: chúng không lưu mẫu, nhưng buộc phải có một mạng sâu cực mạnh được huấn luyện sẵn từ trước.
> - Ở góc dưới bên trái, chúng ta có GDumb và hàng loạt thuật toán replay truyền thống: chúng có thể học từ pixel thô, nhưng lại phụ thuộc nặng nề vào việc lưu trữ hàng ngàn bức ảnh trong bộ nhớ đệm và train lại nhiều lần.
> 
> Ô dưới cùng bên phải – **học từ pixel thô, không có backbone tiền huấn luyện, và không được lưu lại bất kỳ mẫu ảnh nào** – chính là khoảng trống bị bỏ quên nhiều năm qua. Các thuật toán deep learning cố gắng giải quyết ô này đều thất bại nặng nề bởi hiện tượng quên thảm khốc: khi mạng cố cập nhật trên luồng dữ liệu 1 lượt không có replay, độ chính xác tụt dốc thảm hại.
> 
> Sự xuất hiện của RanDumb đã lấp đầy chính xác khoảng trống này. Bằng cách kết hợp phép chiếu ngẫu nhiên Random Fourier Features (đóng vai trò xấp xỉ kernel vô hạn chiều mà không cần học) với ước lượng thống kê hiệp phương sai trực tuyến, RanDumb trở thành một baseline vững chắc, chứng minh rằng không cần mạng sâu phức tạp, một giải pháp giải tích thuần túy vẫn có thể hoạt động xuất sắc tại góc khó khăn nhất của bài toán học trực tuyến."

---

## SLIDE 5: ĐÓNG GÓP THEO LỜI BÀI BÁO (AUTHORS' CLAIMS)

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Các đóng góp cốt lõi theo công bố của bài báo
- **Tóm tắt 4 luận điểm đóng góp chính (Diễn đạt bằng lời của nhóm):**
  - **1. Đề xuất RanDumb – Một đường cơ sở (Baseline) cực kỳ đơn giản nhưng vượt trội:**
    Kết hợp Random Fourier Features (RFF) cố định và phân loại khoảng cách Mahalanobis trực tuyến; thiết lập kỷ lục hiệu năng mới trên nhiều benchmark mà không cần tối ưu trọng số mạng.
  - **2. Đặt nghi vấn về tính thực chất của các phương pháp học biểu diễn liên tục:**
    Chỉ ra rằng trên thực tế, các phương pháp deep continual learning phức tạp hiện nay không học được biểu diễn tốt hơn một phép chiếu ngẫu nhiên cố định khi chỉ được duyệt dữ liệu 1 lần.
  - **3. Phơi bày các điểm yếu và sự thiên lệch trong các Benchmark hiện hành:**
    Chứng minh rằng các quy chuẩn benchmark online continual learning hiện tại đặt ra các ràng buộc quá ngặt nghèo (1 epoch, ít dữ liệu) khiến học sâu bị tê liệt, trong khi lại tạo lợi thế tuyệt đối cho các phương pháp phi tham số.
  - **4. Mở rộng thành công lên các mô hình Foundation (Benchmark F):**
    Chứng minh cơ chế phân loại Mahalanobis với co hệ số OAS hoạt động vượt trội trên đặc trưng trích từ Vision Transformer (ViT), vượt qua các phương pháp prompt-tuning phức tạp.
- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 380 từ)
> "Trong bài báo công bố tại NeurIPS 2024, nhóm tác giả đã tuyên bố bốn đóng góp khoa học chính, được chúng em tóm lược lại như sau:
> 
> Đóng góp thứ nhất: Tác giả xây dựng thành công **RanDumb**, một phương pháp phân loại trực tuyến đơn giản đến kinh ngạc. Phương pháp này chỉ gồm hai bước: chiếu dữ liệu qua Random Fourier Features để xấp xỉ RBF kernel trên không gian chiều cao (ví dụ 25,000 chiều), sau đó tính khoảng cách Mahalanobis trực tuyến dựa trên trung bình lớp và ma trận hiệp phương sai chung được co hệ số bằng kỹ thuật OAS. Đơn giản là vậy, nhưng RanDumb vượt qua tất cả các mô hình deep learning phức tạp trên 5 tập dữ liệu kinh điển từ MNIST, CIFAR đến TinyImageNet.
> 
> Đóng góp thứ hai: Bài báo đóng vai trò như một **hồi chuông cảnh tỉnh** đối với cộng đồng nghiên cứu Continual Learning. Bằng chứng thực nghiệm đanh thép của bài báo cho thấy nỗ lực học biểu diễn trực tuyến bằng mạng sâu trong điều kiện luồng dữ liệu 1 pass thực chất chỉ mang lại ảo tưởng về sự thích nghi; một biểu diễn ngẫu nhiên tĩnh không học gì cả lại cho kết quả phân loại vượt trội và hoàn toàn ổn định.
> 
> Đóng góp thứ ba: Tác giả chỉ ra **lỗ hổng trong thiết kế benchmark**. Các kịch bản đánh giá continual learning hiện tại vô tình đặt các mạng sâu vào tình thế 'chắc chắn thất bại': bắt mạng nơ-ron học hàng ngàn tham số phi tuyến tính chỉ qua 1 lượt xem ảnh và cấm lưu mẫu, khiến gradient bị nhiễu và quên sạch quá khứ.
> 
> Cuối cùng, đóng góp thứ tư: Khi chuyển sang kịch bản có mô hình nền tảng tiền huấn luyện (Benchmark F), tác giả chứng minh chỉ cần lấy đặc trưng ViT đưa thẳng vào bộ phân loại Mahalanobis của RanDumb là đã đủ đánh bại các kỹ thuật Parameter-Efficient Fine-Tuning và Prompt-Tuning phức tạp nhất hiện nay."

---

## SLIDE 6: XẾP LOẠI ĐÓNG GÓP & ĐIỀU CHƯA LÀM ĐƯỢC

### 1. Bố cục mặt slide (Visual & Content)
- **Tiêu đề:** Phân loại đóng góp và Giá trị mới so với tiền nhiệm
- **Bảng phân loại đóng góp 3 nhánh (Yêu cầu bắt buộc của môn học):**

| Phân loại | Đóng góp cụ thể của bài báo | Điều gì trước đây chưa làm được? (Novelty) |
|---|---|---|
| **Lý thuyết** | Hầu như không có định lý hay phát minh toán học mới. Dựa hoàn toàn trên các công cụ sẵn có: RFF (Rahimi & Recht, 2007) và LDA / OAS Shrinkage. | Trước đây chưa ai chỉ ra mối liên hệ thực nghiệm rõ ràng giữa xấp xỉ Kernel Hilbert và khả năng miễn nhiễm quên thảm khốc trong Online Continual Learning. |
| **Kỹ thuật** | Ghép nối khéo léo 3 khối: Chiếu ngẫu nhiên RFF $\to$ Cập nhật thống kê trực tuyến một lượt $\to$ Co hệ số hiệp phương sai OAS và giải hệ Cholesky. | Trước đây, SLDA chỉ chạy được trên feature pretrained, còn RanPAC chỉ chạy trên backbone lớn; RanDumb là kỹ thuật đầu tiên hiện thực hóa pipeline này **trực tiếp từ pixel thô**. |
| **Thực nghiệm** | Thực hiện hệ thống thực nghiệm đồ sộ trên 6 benchmark (A đến F); chứng minh biểu diễn ngẫu nhiên vượt trội hơn biểu diễn học được. | Trước RanDumb, **chưa từng có công trình nào kiểm chứng giả định trung tâm** của Online Continual Learning; đây là lần đầu tiên giả định này bị bác bỏ bằng số liệu thực nghiệm. |

- **Phát hiện phản biện của nhóm (Critical Finding):**
  - Mặc dù tác giả tuyên bố RanDumb vượt trội, nhưng ở Bảng 1 (trái), trên **CIFAR-10**, RanDumb ($55.6\%$) vẫn **thua phương pháp PEC ($58.9\%$) tới 3.3%**. Tuyên bố *"vượt trội mọi phương pháp"* không hoàn toàn tuyệt đối trên mọi tập dữ liệu!
- **Trích dẫn cuối slide:**
  `[1] A. Prabhu et al., "RanDumb: Random Representations Outperform Online Continually Learned Representations," in NeurIPS, vol. 37, 2024.`

### 2. Speaker Notes (Lời thuyết trình chi tiết - 420 từ)
> "Khi tiến hành mổ xẻ và phân loại các đóng góp của bài báo theo các tiêu chuẩn học thuật của môn học, nhóm chúng em đưa ra ba nhận định thẳng thắn sau:
> 
> Thứ nhất, **về mặt Lý thuyết:** Nhóm chúng em khẳng định bài báo này hầu như không có đóng góp lý thuyết mới. Tác giả không đề xuất bất kỳ một định lý toán học nào, không chứng minh được cận trên hay cận dưới về sai số tổng quát hóa. Toàn bộ nền tảng toán học đều được mượn từ các công trình kinh điển: Random Fourier Features của Rahimi và Recht từ năm 2007, bộ ước lượng hiệp phương sai OAS của Chen năm 2010, và thuật toán phân loại tuyến tính Fisher LDA từ thế kỷ trước. Sự mới mẻ ở đây chỉ dừng lại ở trực giác hình học: chứng minh rằng chiếu lên không gian Hilbert ngẫu nhiên đủ lớn sẽ giúp dữ liệu phân tách tuyến tính tốt mà không cần học.
> 
> Thứ hai, **về mặt Kỹ thuật:** Đóng góp kỹ thuật của bài nằm ở sự lắp ghép tinh gọn và hiệu quả. Trước đây, các nghiên cứu như SLDA hay RanPAC đều bó tay nếu không có mô hình pretrained có sẵn. RanDumb là công trình đầu tiên ghép RFF với streaming covariance để tạo ra một pipeline hoàn chỉnh chạy trực tiếp từ ảnh pixel thô, hoàn thành trong thời gian tính toán cực ngắn và bộ nhớ hằng số theo số lượng mẫu.
> 
> Thứ ba, **về mặt Thực nghiệm:** Đây chính là đóng góp giá trị nhất của bài báo. Tác giả đã tiến hành một khối lượng thực nghiệm khổng lồ quét qua 6 benchmark từ A đến F để trực tiếp lật đổ một giả định tồn tại suốt nhiều năm. Trước công trình này, chưa ai từng đặt biểu diễn ngẫu nhiên lên bàn cân sòng phẳng với biểu diễn của mạng sâu trong điều kiện online stream.
> 
> Tuy nhiên, dưới góc nhìn phản biện độc lập của nhóm, chúng em phát hiện một chi tiết rất đáng chú ý mà bài báo cố tình nói giảm nói tránh: Dù tác giả tuyên bố RanDumb vượt qua tất cả các phương pháp Continual Learning, nhưng ngay trong Bảng 1 của bài báo, trên tập dữ liệu **CIFAR-10**, RanDumb chỉ đạt $55.6\%$, hoàn toàn **thua kém phương pháp PEC ($58.9\%$) tới $3.3\%$**. Điều này chứng minh rằng biểu diễn ngẫu nhiên không phải là viên đạn bạc cho mọi bài toán, mở ra cơ sở lý luận rất quan trọng cho phần Critique và Empirics của nhóm chúng em ở nửa sau của học kỳ."

---

