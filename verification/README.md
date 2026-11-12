# Kiểm chứng với code gốc (E9 Verification)

Thư mục này dành riêng cho **Huy** để phục vụ thực nghiệm **E9: So sánh và kiểm chứng với code gốc của tác giả**.

---

## 1. Nguyên tắc đạo đức học thuật (Tuyệt đối tuân thủ)
- **Không đưa code của tác giả vào git repo của nhóm.** Repo gốc dùng giấy phép GPL-3.0.
- Mã nguồn tác giả được clone tự động vào thư mục `verification/_official/` (thư mục này đã được khai báo trong `.gitignore`).
- Chỉ Huy đọc và chạy code của tác giả. Bạn Đăng (người code thuật toán) không mở code này.
- Ghim chính xác commit hash: **`14a51ee0c045bff642f6ffbfe481efa4d49a3033`** (lưu tại `official_commit_sha.txt`).
- Mọi run thực nghiệm kiểm chứng trên Weights & Biases bắt buộc phải gắn thẻ: `official-code`.

---

## 2. Hướng dẫn thiết lập môi trường kiểm chứng
Chạy lệnh sau để tải và ghim commit repo của tác giả:
```bash
python verification/setup_official.py
```

---

## 3. Các phát hiện quan trọng từ Code gốc (Findings của Huy)
Sau khi phân tích trực tiếp mã nguồn trong `verification/_official/`, Huy đã ghi nhận 6 điểm sai khác và lỗi cú pháp thực tế sau đây (dùng để đưa vào Slide Deck và phân tích sai lệch E9):

| Yếu tố | Bài báo công bố (Paper) | Thực tế code của tác giả (`_official`) | Phân tích tác động |
|---|---|---|---|
| **Tham số Kernel $\gamma$** | Bài ghi $\gamma = 1.0$ (Mục 2) | Trong `src/main.py` dòng 81: `RBFSampler(gamma='scale', n_components=...)` | $\gamma = 1.0$ trên ảnh pixel đã z-score sẽ làm $\omega^T x$ dao động quá lớn ($\sim 78$ rad), sinh ra nhiễu thuần túy. Việc code gốc dùng `gamma='scale'` là mấu chốt giúp phương pháp hoạt động! |
| **Chuẩn hóa ảnh** | Bài ghi chung chung "normalize all images" | Trong `feats/get_feats.py`: chia 255, sau đó trừ mean và chia std theo từng kênh (z-score) của từng dataset. | Cần áp dụng đúng chuẩn hóa channel-wise này để kết quả tái lập khớp số. |
| **Bản chất tính toán (Online vs Batch)** | Bài viết thuật toán online từng mẫu một | Trong `README.md` và `src/main.py`: tác giả thừa nhận code nạp batch lớn cùng lúc để chạy nhanh. `legacy_model.py` tính mean trước rồi đưa vào `OAS(assume_centered=True)`. | Toán học của cập nhật online theo batch tích lũy $A = \sum \Phi^T \Phi$ tương đương chính xác về mặt giải tích với batch một lần. |
| **Hệ số $\lambda$ (Ridge penalty)** | Bài ghi có $\lambda$ tùy dataset ($10^{-6}$ đến $10^{-4}$) | `src/main.py` dùng LDA của scikit-learn với $\lambda = 0$. `legacy_model.py` dùng `shrinkage_param=1e-4`. | Tác giả ghi trong README là bản đơn giản hóa bỏ $\lambda$ để không phụ thuộc hparam, dẫn đến lệch khoảng $\pm 0.8\%$ so với bài báo. |
| **Flip Augmentation trên ViT (Benchmark F)** | Bài mô tả Benchmark F chỉ học classifier tuyến tính trên ViT, không augment | Trong `scripts/runall.sh` dòng 198-211: có cả các lệnh chạy `--augment` trên feature ViT. | Cần lưu ý khi so sánh bảng Benchmark F (chỉ lấy số không augment theo đúng lý thuyết). |
| **Lỗi cú pháp trong `get_feats_vit.py`** | Không đề cập | `feats/get_feats_vit.py` dòng 14 bị lỗi thụt lề (IndentationError: 3 spaces), dòng 60 gọi `model.cuda()` trước khi khởi tạo `model` ở dòng 65 (`NameError`). | Minh chứng rõ ràng về việc code tác giả chưa được trau chuốt, là điểm cộng lớn khi báo cáo phân tích cho môn học. |
