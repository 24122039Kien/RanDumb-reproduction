# Bàn giao: sửa tài liệu dịch bài báo để soạn slide

**Người giao:** Kiên (chủ repo)
**Ngày:** 20/11/2026. Hạn nộp deck: **21/11/2026**.
**Người nhận:** phiên Claude có trình duyệt (Claude Desktop) đã đăng nhập Google của Kiên.

Đọc hết file này trước khi mở tài liệu. Mọi số liệu dưới đây đã được đối chiếu với bài báo gốc (`Randumbofficial.pdf`, 19 trang) và với kết quả chạy của nhóm.

---

## 1. Bối cảnh

- **Đồ án:** tái lập và phản biện bài *Random Representations Outperform Online Continually Learned Representations* (RanDumb), Prabhu, Sinha, Kumaraguru, Torr, Sener, Dokania, NeurIPS 2024. Môn Nhập môn Học máy, HCMUS.
- **Nhóm:** Kiên, Đăng, Huy.
- **Tài liệu cần sửa:** Google Doc dịch và giải thích bài báo, https://docs.google.com/document/d/1_aKaNZ0ie60fQgpwF3l-LB6Gk-dgdJvV-OaiOkflGJc/edit
- **Vai trò của tài liệu:** tài liệu nội bộ để nhóm hiểu bài và soạn slide. **Không phải sản phẩm nộp.** Vì vậy cần đúng số liệu và đủ ý cho slide, không cần văn phong trau chuốt.
- **Quy định của môn ảnh hưởng tới tài liệu này:**
  - Không chép chữ hay hình của bài vào deck và poster. Bản dịch chỉ dùng để đọc hiểu, slide phải viết lại.
  - Mọi số liệu của nhóm trên slide phải có link tới đúng W&B run.

## 2. Nguồn sự thật (đọc khi cần đối chiếu)

| Nguồn | Dùng cho |
|---|---|
| `Randumbofficial.pdf` ở gốc repo | Mọi câu chữ, số liệu, bảng của bài |
| [report4.md](report4.md) mục 2 và mục 7 | Kết quả E2 và E2 bổ sung, kèm link W&B |
| [e2_followup/](e2_followup/) | Số liệu gốc E2 bổ sung (CSV) |
| [e1-e8-A/summary.csv](e1-e8-A/summary.csv) | Kết quả E8 pha A (32 run) |
| [e1_e8_fidelity.md](e1_e8_fidelity.md) | Cấu hình E1/E8 và các điểm bài không nói rõ |
| [../verification/README.md](../verification/README.md) | Phát hiện của Huy khi đọc code gốc (E9) |
| [Instruction.md](Instruction.md) | Yêu cầu của môn |

Kết quả **E1** (của Đăng) và **E8 pha B** (của Huy) đã chạy xong trên W&B, nhóm `E1` và `E8`, nhưng file CSV **chưa có trong repo**. Nếu cần số đó thì hỏi Kiên, không tự điền.

---

## 3. Việc 1: sửa Google Doc

### Cách làm
1. Mở tài liệu, bật chế độ **Suggesting (Đề xuất)** (góc phải trên, biểu tượng bút chì). Kiên sẽ duyệt từng đề xuất.
2. Tìm chỗ cần sửa bằng `Ctrl+F`, theo tên mục ghi trong bảng dưới.
3. Giữ cấu trúc và văn phong của tài liệu. Chỉ sửa chỗ sai, thêm chỗ thiếu.
4. Mọi nhận xét **của nhóm** (không phải của bài) phải mở đầu bằng **"Ghi chú của nhóm:"**, để không lẫn với phần dịch.
5. Không động vào các công thức đang đúng (xem mục 3.6).
6. Làm xong, báo lại Kiên: đã sửa những mục nào, mục nào bỏ qua và vì sao.

### 3.1. Lỗi sai số liệu (bắt buộc sửa)

| Vị trí trong doc | Doc đang ghi | Sửa thành (theo bài báo) |
|---|---|---|
| Phần 4.1, bảng benchmark, dòng A | "1 class/task, 1 sample/step" | "1 lớp mỗi task, **10 mẫu mỗi bước**, 0 exemplar". "1 mẫu mỗi bước" là dòng của RanDumb. Thêm cột exemplar cho các dòng khác: B1 100-2000, B2 100-1000 (có contrastive và augment), C 0 (nhiều lượt), D 1000, E 1000-5000, F 0 (nhiều lượt, 1 lớp mỗi task). |
| Phần 5.2, Benchmark B.1, dòng CIFAR-10 | "OCM 35.2" | "**DVC** 35,2" (Bảng 3, M = 0,1k). OCM không có trong Bảng 3. |
| Phần 5.2, Benchmark B.1, dòng MNIST | "IL2A 90.2, +8.1" | Mạnh nhất là "**BiC 90,4**". Bài in chênh lệch **+7,8**. |
| Phần 5.3, Benchmark B.2, dòng MNIST | "OnPro / SCR không có kết quả" | Chỉ OnPro không có MNIST. **SCR 86,2, OCM 90,7**. RanDumb 98,3, bài in **+7,5**. Thêm hàng bộ nhớ: MNIST và CIFAR-10 M = 0,1k, CIFAR-100 M = 0,5k, Tiny M = 1k. |
| Phần 5.5, Benchmark D | "vượt qua **phần lớn** phương pháp trên CIFAR-100 với khoảng cách 9.1%" | "vượt qua **tất cả** phương pháp trước trên CIFAR-100, hơn 9,1 điểm so với VR-MCL (19,5)". |

### 3.2. Diễn giải gây hiểu nhầm (bắt buộc sửa)

**Phần 6.6, so sánh kiến trúc:**
- Thêm thiết lập: đây là **CIFAR-100 task-incremental**, so với baseline **ER** theo thiết lập của Mirzadeh et al. Thiết lập này khác hẳn các bảng trước, nên RanDumb ở đây đạt 74,8% chứ không phải 28,6%.
- Thêm: RanDumb (74,8) **thấp hơn CNN×16 (76,8)**, bài ghi "−2,0". Kết luận của bài là "vượt **gần như** mọi kiến trúc", không phải tất cả.
- Bổ sung các dòng còn thiếu của bảng:

| Kiến trúc | Độ chính xác (%) |
|---|---|
| ResNet-18 | 45,0 |
| ResNet-34 | 44,8 |
| WRN-10-2 | 50,5 |
| WRN-10-10 | 56,8 |
| WRN-16-2 | 44,6 |
| WRN-16-10 | 51,3 |
| WRN-28-2 | 46,6 |
| WRN-28-10 | 49,3 |
| ViT-512/1024 | 51,7 |
| ViT-1024/1546 | 60,4 |

**Phần 6.1, ablation:**
- Sửa "Tác giả thực hiện ba cấu hình" thành "bốn cấu hình", vì danh sách liệt kê 4 dòng.
- Thêm cột miniImageNet: RanDumb 17,7; bỏ Decorrelation 8,9 (−8,8); bỏ Embedding 12,9 (−4,8); bỏ cả hai 7,28 (−10,4).
- Thêm tên gọi của bài: bỏ Decorrelation tương đương "Kernel-NCM"; bỏ Embedding tương đương baseline SLDA. Bài nêu mức giảm 6-25% và 3-14%; bỏ cả hai giảm trung bình 17%.
- Thay câu "Bảng 6 sử dụng cấu hình thực nghiệm riêng" bằng: "Ghi chú của nhóm: Tiny 11,1 trong Bảng 6 trùng với dòng λ = 10⁻⁵, còn các Bảng 3-5 dùng 11,6 ứng với λ = 10⁻⁴. Bài không giải thích sự khác nhau này."

**Phần 6.4, tham số λ:** thêm bảng số từ Bảng 6.

| λ | MNIST | CIFAR-10 | CIFAR-100 | TinyImageNet | miniImageNet |
|---|---|---|---|---|---|
| 10⁻⁶ | 98,3 | 53,9 | 27,8 | 10,3 | 15,8 |
| 10⁻⁵ | (không có) | 55,6 | 28,6 | 11,1 | 15,9 |
| 10⁻⁴ | 96,6 | 52,6 | 26,1 | 11,6 | 17,7 |

Sửa ý "λ quá nhỏ làm giảm CIFAR" thành: λ = 10⁻⁶ làm giảm nhẹ **cả bốn dataset** CIFAR-10, CIFAR-100, Tiny, mini. λ = 10⁻⁴ làm CIFAR-10 và CIFAR-100 giảm 3 và 2,5 điểm, nhưng làm Tiny tăng 0,5 và mini tăng 1,8.

**Phần 7.2, Benchmark F:**
- Thêm dòng **NCM** vào bảng IN-1K: 78,3 / 44,3 / 62,5 / 84,8 / 88,2. Đây là số E2 của nhóm dùng để so.
- Thêm ghi chú: với IN-21K trên IN-A, Adam-Adapt (48,8), Adam-FT (48,5) và Adam-SSF (46,0) **cao hơn** RanDumb (42,2). Với IN-1K trên IN-A, RanDumb 49,5 chỉ nhỉnh hơn Adam-Adapt và Adam-FT (cùng 49,3).
- Thêm câu của bài: RanPAC thêm RP+ReLU và fine-tune ở task đầu, nhưng vẫn không cao hơn RanDumb.

### 3.3. Thiếu ý quan trọng cho slide (nên thêm)

- **Phần 4.2:** thêm "Bài dùng **cài đặt Random Fourier Features của Scikit-Learn**, 25K chiều, γ = 1,0". Kèm ghi chú: "Ghi chú của nhóm: Mục 2 của bài mô tả φ(x) gồm cos và sin, tức 2D chiều. Cài đặt của scikit-learn (`RBFSampler`) lại là D chiều dạng `√(2/D)·cos(ωᵀx + b)`. Đây là mâu thuẫn trong chính bài báo."
- **Phần 4.2:** thêm giao thức đánh giá. Bài đo độ chính xác trên tập test của **mọi lớp đã thấy, sau khi đi hết một lượt dữ liệu**, tức là độ chính xác trung bình sau task cuối. Với Benchmark A và F (1 lớp mỗi task), con số này không phụ thuộc thứ tự task. **Không phải** Average Incremental Accuracy.
- **Phần 4.2 hoặc 7.2, Benchmark F:**
  - Bài chạy 7 dataset của LAMDA-PILOT, thay ObjectNet bằng Stanford Cars (vì giấy phép ObjectNet cấm huấn luyện).
  - Các phương pháp khác chạy bằng codebase LAMDA-PILOT, giữ siêu tham số gốc. Chỉ đổi thành 2 lớp ban đầu và 1 lớp mỗi task.
  - Không lật ảnh ở Benchmark F.
- **Phần 2.3 hoặc Phần 13:** thêm số của Bảng 1. Đây là căn cứ cho câu "thu hẹp 70-90% khoảng cách":
  - Online, so với Joint một lượt: MNIST 100%, CIFAR-10 75%, CIFAR-100 87%, miniImageNet 70%. Joint: 98,3 / 74,2 / 33,0 / 25,3.
  - Offline ViT IN-21K: CIFAR 93%, IN-A 60%, IN-R 75%, CUB 97%, OB 92%, VTAB 97%, Cars 77%.
- **Phần 3.3:** thêm ý của Hình 2. Decorrelate đưa mỗi chiều đặc trưng về **phương sai đơn vị**, nhờ đó độ tương đồng cosine tách được các lớp.
- **Phần 6.3, flip:** thêm MNIST 98,3 (không lật vì MNIST không dùng augment).

### 3.4. Thêm một mục mới: "Mâu thuẫn trong chính bài báo"

Đặt ngay sau Phần 9 (hoặc trước Phần 13), tiêu đề **"Ghi chú của nhóm: các điểm không nhất quán trong bài"**. Nhóm dùng mục này cho slide phản biện.

1. **Dạng RFF:** Mục 2 viết dạng cos + sin, 2D chiều. Phần cài đặt dùng scikit-learn, D chiều.
2. **Phương trình 3:** được gọi là "hàm tương đồng" và đi với argmax ở phương trình 1, nhưng thực chất là khoảng cách Mahalanobis nên phải dùng argmin. Doc đã có ý này ở Phần 3.3, chỉ cần dẫn lại.
3. **TinyImageNet:** Bảng 3, 4, 5 dùng 11,6 (λ = 10⁻⁴); dòng RanDumb của Bảng 6 dùng 11,1 (λ = 10⁻⁵).
4. **Bảng 3, CIFAR-10 M = 0,2k:** bài in RanDumb **+5,9**, nhưng IL2A đạt 54,7 nên chênh thật chỉ **+0,9**.
5. **Bảng 3, MNIST:** in +7,8, nhưng chênh với BiC 90,4 là 7,9.
6. **Chênh với PEC trên MNIST:** Bảng 1 in +6,0, Bảng 2 in +5,9.
7. **So với RanPAC (RP+ReLU):** bài viết "trung bình hơn 3,4%", nhưng tính từ bảng (+3,1, +6,8, +5,5, +1,4, +2,0) ra **3,76**.
8. **Tiêu đề cột Bảng 6:** ghi CIFAR-100 "(10/1)", đúng ra phải là (100/1).
9. **Chú thích Bảng 5:** nói RanDumb "vượt mọi phương pháp trừ VR-MCL trên TinyImageNet", nhưng cùng bảng đó GDumb (61,2) cao hơn RanDumb (55,6) trên CIFAR-10.
10. **γ = 1,0 (bài) và `gamma='scale'` (code gốc):** xem mục 4.2. Thực nghiệm của nhóm xác nhận chỉ cấu hình của code gốc mới ra đúng số của bài.

### 3.5. Dọn dẹp
- Xóa 4 dòng cuối tài liệu: ghi chú "6/11 : xong sprint 0…" và link ChatGPT share.
- Văn phong "Tôi / bạn" kiểu trợ lý có thể giữ, vì đây là tài liệu nội bộ. Không cần viết lại.

### 3.6. Đã kiểm tra là đúng, không sửa
- **Công thức:** NCM, RBF, RFF với `ω ~ N(0, 2γI)`, φ(x) gồm 2D chiều với hệ số `1/√D`, Mahalanobis, `S + λI`, Sherman–Morrison, định nghĩa OCL ở Phụ lục B.
- **Siêu tham số:** 25K, γ = 1,0, λ 10⁻⁶ / 10⁻⁵ / 10⁻⁴, flip, ảnh 32×32, vector 784 / 3.072 / 768 chiều.
- **Bảng số liệu:** Benchmark A, C, E; các số ablation đang có; số chiều; flip; RanPAC; hai bảng Benchmark F.
- **Các phần khác:** Related Works, Kết luận, Hạn chế, Phụ lục A/B, checklist, 82 tài liệu tham khảo và tên tác giả các công trình chính.

---

## 4. Việc 2: số liệu đã kiểm chứng của nhóm (dùng cho slide)

### 4.1. E2: RanDumb và NCM trên đặc trưng ViT-B/16 (Benchmark F)

Link run: `https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/<ID>`

| Dataset | Backbone | RanDumb nhóm | RanDumb bài | Run | NCM nhóm | NCM bài | Run |
|---|---|---|---|---|---|---|---|
| CIFAR-100 | IN-21K | 88,26 | 86,8 | qgyoenv0 | 79,28 | 83,5 | 658h5g12 |
| CIFAR-100 | IN-1K | 87,79 | 84,5 | iap1teh9 | 82,68 | 78,3 | u6z1f3hr |
| CUB | IN-21K | 86,71 | 88,5 | gwcehtms | 84,02 | 86,5 | gqarb26i |
| CUB | IN-1K | 86,88 | 88,0 | 4xbdp2lr | 84,21 | 84,8 | o31zb7vu |
| ImageNet-R | IN-21K | 65,04 | 64,9 | k9b69i12 | 50,23 | 54,8 | 2nwh32an |
| ImageNet-R | IN-1K | 66,69 | 66,9 | pih4rrhp | 59,30 | 62,5 | 5a1val65 |
| ImageNet-A | IN-21K | 49,33 | 42,2 | th91yx1d | 37,27 | 41,4 | 2i13quhv |
| ImageNet-A | IN-1K | 55,00 | 49,5 | 6tvoh895 | 44,13 | 44,3 | a6t2exh4 |

- RanDumb hơn NCM ở cả 8 cặp.
- So với số của bài, trong ngưỡng ±5%: RanDumb 6/8, NCM 3/8.
- ImageNet-A cao hơn bài 11-17%. Chia lại train/test với 4 seed khác, độ lệch chuẩn dưới 0,4 điểm, nên **không phải do may rủi chia**. Đổi NCM sang cosine thay đổi dưới 1,1 điểm. Chi tiết ở [report4.md](report4.md) mục 7.

### 4.2. E8 pha A: chuẩn hóa × γ (CIFAR, OAS, λ = 10⁻⁵, seed 0)

| Cấu hình | CIFAR-10, D = 25.000 | CIFAR-100, D = 25.000 | Ghi chú |
|---|---|---|---|
| **z-score + `scale`** (cấu hình code gốc) | **55,56** (xt5ndf1l) | **28,62** (jcimgvig) | Khớp số của bài (55,6 / 28,6) |
| chỉ chia 255 + `scale` | 55,48 (sa8x4qpi) | 28,45 (4pa2jdx8) | |
| L2 + `scale` | 54,65 (lkh24j88) | 28,15 (mawjpqrg) | |
| z-score + γ = 1,0 (đúng như bài ghi) | 10,09 (wd4q4rih) | 1,12 (wflh0mor) | Bằng đoán ngẫu nhiên |
| chỉ chia 255 + γ = 1,0 | 10,06 (jtua3cwm) | 1,15 (y205x9a6) | Bằng đoán ngẫu nhiên |

- Ở D = 5.000, `0,5·scale` tốt nhất với cả ba cách chuẩn hóa: CIFAR-10 khoảng 51-52%, CIFAR-100 khoảng 25-26%. `2·scale` thấp hơn mức tốt nhất khoảng 2-5,5 điểm. Số đầy đủ ở [e1-e8-A/summary.csv](e1-e8-A/summary.csv).
- **Thông điệp cho slide:** γ = 1,0 như bài ghi không chạy được với chuẩn hóa z-score. Cấu hình của code gốc (z-score + `scale`) tái lập gần như chính xác số của bài. Đây là bằng chứng thực nghiệm cho phát hiện #1 của E9.
- L2 + γ = 1,0 (cấu hình E1 của nhóm) được chạy trong E8 pha B, chưa có CSV trong repo.

---

## 5. Việc 3: lỗi trong các bản nháp slide trên repo (chỉ báo cáo, không tự sửa)

Các file sau của Huy có lỗi. **Không sửa trực tiếp.** Liệt kê lại cho Kiên để nhóm quyết định. Nếu soạn slide từ các file này thì phải dùng số đúng ở cột phải.

| File, vị trí | Đang ghi | Đúng |
|---|---|---|
| `slide_design_instructions_for_model.md` §0.1 và tài liệu [1] | Tác giả: Prabhu, Hammoud, Lim, Ghanem, Torr, Bibi | **Prabhu, Sinha, Kumaraguru, Torr, Sener, Dokania** |
| `slide_design_instructions_for_model.md` slide 15; `slide_draft_methodology_experiments.md` slide 15 | Bảng E2: bài 88,4 / 81,1 / …; nhóm 88,08 / 81,71 / …; run x06w28q2, 90n2w8g8, … | **Sai toàn bộ**, cả số và ID run. Dùng bảng ở mục 4.1. |
| `slide_design_instructions_for_model.md` slide 20 và 25 | "E2 CIFAR-100 lệch ≤ 0,5%", "15/16 run lệch < 3%" | RanDumb 6/8 và NCM 3/8 trong ±5% (mục 4.1) |
| Cả hai file, slide giao thức đánh giá | Average Incremental Accuracy `A_inc` | Độ chính xác cuối, trên mọi lớp đã thấy sau một lượt (mục 3.3) |
| Cả hai file, sơ đồ pipeline | Khối chuẩn hóa là "z-score" | E1 của nhóm dùng **L2**; code gốc dùng z-score; bài chỉ ghi "normalize" |
| `slide_design_instructions_for_model.md` slide 24 và 26 | "25K² × 8 byte ≈ 50 GB" | 625 triệu × 8 byte = **5 GB (4,66 GiB)** |
| `slide_design_instructions_for_model.md` §0.3 và slide 16 | E8 "seed 0-2", "Acc (3 seeds)"; E8-A là "OAS × λ" | E8 chỉ seed 0. E8-A là **chuẩn hóa × γ**, E8-B là **shrink × λ** |
| `slide_design_instructions_for_model.md` slide 22 | "Nhóm sử dụng λ relative → ổn định hơn" | Chưa có căn cứ; E1 dùng λ cố định. Chờ kết quả E8-B. |
| Tài liệu [11] trong cả hai file | `apd20/randumb` | `https://github.com/drimpossible/RanDumb`, commit `14a51ee` |
| `slide_design_instructions_for_model.md` slide 6 | "vượt trội tất cả SotA trên 5 datasets" | Bài tự nhận thua PEC (CIFAR-10), OnPro (CIFAR-10, Tiny), GDumb (B.1 Tiny, D CIFAR-10), VR-MCL (D Tiny) |
| Slide 3, 7 (instructions); slide 2, 6 (`slide_draft_context_contributions.md`) | RFF "miễn nhiễm với quên thảm khốc"; "lần đầu liên hệ RFF với miễn nhiễm forgetting" | Bài không tuyên bố điều này. Nếu giữ thì ghi là **nhận định của nhóm**. Novelty bài tự nêu: lần đầu dùng random features để học liên tục từ đầu, không cần học biểu diễn. |
| `slide_design_instructions_for_model.md` slide 23 | Ghép bảng kiến trúc (task-incremental, ER) với câu "thu hẹp 70-90%" | Hai số đến từ hai bảng khác nhau (Bảng 6 phải và Bảng 1). Nên tách riêng. |
| `slide_design_instructions_for_model.md` slide 21 | Bảng D = 1K / 5K / 15K có dấu "~" | Bài chỉ cho số trong Hình 3, không có bảng. Ghi rõ "ước lượng đọc từ hình". |

---

## 6. Checklist khi xong
- [ ] Đã sửa 5 lỗi ở mục 3.1 (dạng đề xuất).
- [ ] Đã sửa 4 điểm ở mục 3.2.
- [ ] Đã thêm các ý ở mục 3.3 và mục mới 3.4.
- [ ] Đã xóa ghi chú cá nhân và link ChatGPT (mục 3.5).
- [ ] Không thay đổi phần đã đúng (mục 3.6).
- [ ] Báo lại Kiên: danh sách đã sửa, chỗ bỏ qua và lý do, và lỗi trong mục 5 nếu soạn slide.
