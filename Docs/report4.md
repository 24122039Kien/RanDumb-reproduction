# Báo cáo 16/11: kết quả E2 trên Kaggle

**Người lập:** Kiên
**Ngày:** 16/11/2026
**Trạng thái `main`:** commit `456c875`, 56 test qua

---

## 1. Tóm tắt

- **E2 đã chạy xong cả 16 run** trên Kaggle (GPU T4 x2), tất cả đều được log lên W&B.
- **Kết luận chính của bài được tái lập:** RanDumb hơn NCM ở cả 8 cặp dataset × backbone, với mức chênh từ 2,7 đến 14,8 điểm phần trăm.
- **RanDumb:** 6/8 run nằm trong ±5% so với số của bài. Hai run lệch nhiều đều thuộc ImageNet-A, và đều **cao hơn** bài (+16,9% và +11,1%).
- **NCM:** chỉ 3/8 run nằm trong ±5%. Với backbone IN-21K, NCM của nhóm thấp hơn bài ở cả 4 dataset.
- Lỗi đọc file đặc trưng ở ô 5 (báo trong buổi trước) đã được sửa trên `main`. Lần chạy này dùng bản đã sửa.

---

## 2. Kết quả

Độ lệch tương đối = (của nhóm − của bài) / của bài × 100. Mỗi run chỉ có 1 seed vì E2 không có thành phần ngẫu nhiên: không có RFF, và tập chia train/test cố định.

### RanDumb (LDA + OAS trên đặc trưng ViT-B/16, λ = 0)

| Dataset | Backbone | Nhóm (%) | Bài (%) | Lệch tương đối | W&B run |
|---|---|---|---|---|---|
| CIFAR-100 | IN-21K | 88,26 | 86,8 | +1,68% | [qgyoenv0](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/qgyoenv0) |
| CIFAR-100 | IN-1K | 87,79 | 84,5 | +3,89% | [iap1teh9](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/iap1teh9) |
| CUB | IN-21K | 86,71 | 88,5 | −2,02% | [gwcehtms](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/gwcehtms) |
| CUB | IN-1K | 86,88 | 88,0 | −1,27% | [4xbdp2lr](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/4xbdp2lr) |
| ImageNet-R | IN-21K | 65,04 | 64,9 | +0,21% | [k9b69i12](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/k9b69i12) |
| ImageNet-R | IN-1K | 66,69 | 66,9 | −0,32% | [pih4rrhp](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/pih4rrhp) |
| ImageNet-A | IN-21K | 49,33 | 42,2 | **+16,90%** | [th91yx1d](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/th91yx1d) |
| ImageNet-A | IN-1K | 55,00 | 49,5 | **+11,11%** | [6tvoh895](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/6tvoh895) |

### NCM (trung bình lớp, khoảng cách Euclid)

| Dataset | Backbone | Nhóm (%) | Bài (%) | Lệch tương đối | W&B run |
|---|---|---|---|---|---|
| CIFAR-100 | IN-21K | 79,28 | 83,5 | **−5,05%** | [658h5g12](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/658h5g12) |
| CIFAR-100 | IN-1K | 82,68 | 78,3 | **+5,59%** | [u6z1f3hr](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/u6z1f3hr) |
| CUB | IN-21K | 84,02 | 86,5 | −2,87% | [gqarb26i](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/gqarb26i) |
| CUB | IN-1K | 84,21 | 84,8 | −0,70% | [o31zb7vu](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/o31zb7vu) |
| ImageNet-R | IN-21K | 50,23 | 54,8 | **−8,33%** | [2nwh32an](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/2nwh32an) |
| ImageNet-R | IN-1K | 59,30 | 62,5 | **−5,12%** | [5a1val65](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/5a1val65) |
| ImageNet-A | IN-21K | 37,27 | 41,4 | **−9,98%** | [2i13quhv](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/2i13quhv) |
| ImageNet-A | IN-1K | 44,13 | 44,3 | −0,38% | [a6t2exh4](https://wandb.ai/24122039kiki-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/randumb-reproduction-24TNT/runs/a6t2exh4) |

In đậm: lệch quá ±5%.

### RanDumb hơn NCM bao nhiêu (điểm phần trăm)

| Dataset | IN-21K: nhóm | IN-21K: bài | IN-1K: nhóm | IN-1K: bài |
|---|---|---|---|---|
| CIFAR-100 | +8,98 | +3,3 | +5,11 | +6,2 |
| CUB | +2,69 | +2,0 | +2,67 | +3,2 |
| ImageNet-R | +14,81 | +10,1 | +7,39 | +4,4 |
| ImageNet-A | +12,06 | +0,8 | +10,87 | +5,2 |

Cả 8 ô của nhóm đều dương, giống bài. Ở 6/8 ô, khoảng cách của nhóm lớn hơn của bài, chủ yếu vì NCM của nhóm thấp hơn.

---

## 3. Phân tích sai lệch

### 3.1. Sai số do kích thước tập test

Không có seed để tính khoảng tin cậy, nên dùng sai số chuẩn của tỉ lệ đúng trên tập test, √(p(1−p)/n), làm thước đo nhiễu:

| Dataset | Số ảnh test | Sai số chuẩn xấp xỉ (điểm %) |
|---|---|---|
| CIFAR-100 | 10.000 | 0,3 |
| CUB | 5.794 | 0,4 |
| ImageNet-R | 6.000 | 0,6 |
| ImageNet-A | 1.500 | 1,3 |

Các lệch ở ImageNet-A (+7,1 và +5,5 điểm cho RanDumb) lớn hơn nhiều lần mức nhiễu này, nên **không thể giải thích bằng may rủi** mà phải do khác biệt trong thiết lập.

### 3.2. ImageNet-A: RanDumb cao hơn bài 11-17%

**Giả thuyết chính: tập chia train/test khác bài.** Bài dùng tập chia của RevisitingCIL, chỉ có trên Google Drive và không được phép phân phối lại. Nhóm tự chia ngẫu nhiên 80/20 theo lớp với `SPLIT_SEED = 0` (xem [e2_datasets.md](e2_datasets.md), mục 4). ImageNet-A có số ảnh mỗi lớp rất lệch nhau và nhiều lớp chỉ có vài ảnh, nên cách chia ảnh hưởng mạnh đến độ chính xác. Chia theo lớp đảm bảo mỗi lớp đều có ảnh train, điều có thể không đúng với tập chia gốc.

Ủng hộ giả thuyết này: ImageNet-R cũng tự chia nhưng có 30.000 ảnh khá cân bằng giữa các lớp, và lệch dưới 0,4%.

**Chưa kiểm chứng.** Muốn kiểm chứng cần chạy lại với vài seed chia khác (nếu số dao động mạnh thì cách chia là nguyên nhân) hoặc nhờ Huy so tên file với tập chia của code gốc.

### 3.3. NCM với IN-21K thấp hơn bài ở cả 4 dataset

Lệch từ −0,6 đến −4,6 điểm, luôn cùng một chiều, nên có nhiều khả năng là khác biệt cách cài hơn là nhiễu. Hai giả thuyết:

1. **Độ đo khoảng cách.** Nhóm dùng Euclid. Các bài nền (SimpleCIL, RanPAC) thường dùng **cosine** cho bộ phân loại nguyên mẫu. Với đặc trưng IN-21K chưa fine-tune, độ dài vector thay đổi nhiều giữa các ảnh, nên Euclid bị ảnh hưởng còn cosine thì không.
2. **Trọng số backbone.** Nhóm dùng `vit_base_patch16_224.augreg_in21k`. Code các bài nền dùng tên cũ `vit_base_patch16_224_in21k` của timm, có thể trỏ tới bộ trọng số khác tùy phiên bản timm.

Giả thuyết 1 kiểm chứng được rất rẻ: `scripts/run_e2.py` đã có `--ncm-metric cosine`, chỉ cần chạy lại 8 run NCM trên file đặc trưng có sẵn, mỗi run vài giây.

### 3.4. CIFAR-100 với IN-1K: cả RanDumb và NCM đều cao hơn bài

RanDumb +3,9% và NCM +5,6%. Khi cả hai phương pháp cùng lệch lên, nguyên nhân nhiều khả năng nằm ở đặc trưng chứ không ở bộ phân loại: trọng số `augreg_in21k_ft_in1k` có thể khác backbone IN-1K của bài, và nhóm chuẩn hóa ảnh bằng mean/std 0,5 trong khi code gốc giữ ảnh ở [0, 1].

---

## 4. Đánh giá theo tiêu chí ±5%

| Phương pháp | Trong ±5% | Ngoài ±5% |
|---|---|---|
| RanDumb | 6/8 | ImageNet-A × 2 (cao hơn bài) |
| NCM | 3/8 | CIFAR-100 × 2, ImageNet-R × 2, ImageNet-A IN-21K |

Kết luận dùng cho deck:
- **Đã tái lập** con số RanDumb trên CIFAR-100, CUB và ImageNet-R, cùng kết luận "RanDumb hơn NCM".
- **Chưa tái lập** chính xác ImageNet-A, có nguyên nhân khả dĩ là tập chia khác.
- NCM là đường cơ sở, lệch nhiều hơn RanDumb và có giả thuyết kiểm chứng được (mục 3.3).

---

## 5. Biểu đồ W&B trông lạ: không phải lỗi

Mỗi run E2 chỉ log **một lần**, gồm 6 con số cuối cùng (`run.log(metrics)` trong `scripts/run_e2.py`). Khi một chỉ số chỉ có một giá trị, W&B không vẽ được đường theo thời gian, nên tự vẽ **một cột ngang duy nhất** cho mỗi chỉ số. Ví dụ: trang run `E2-imagenet-r-in21k-ncm` có cột `relative_deviation_pct` kéo sang trái đến −8,33.

Biểu đồ trong từng run vì vậy không có ý nghĩa. Muốn so sánh nên xem ở mức project:
- Mở trang **Runs** (dạng bảng) của project, lọc theo nhóm `E2`, rồi thêm các cột `dataset`, `backbone`, `method`, `final_accuracy`, `paper_accuracy`.
- Hoặc thêm một panel **Bar chart** trong workspace, chọn `final_accuracy` và nhóm theo `method`. Panel này dùng được làm hình cho deck.

---

## 6. Việc tiếp theo

| Việc | Người | Hạn |
|---|---|---|
| Chạy lại 8 run NCM với `--ncm-metric cosine` (mục 3.3) | Kiên | 17/11 |
| Thử 2-3 seed chia khác cho ImageNet-A (mục 3.2), nếu còn giờ GPU | Kiên | 17/11 |
| Slide E2: bảng kết quả, biểu đồ RanDumb với NCM, phân tích sai lệch | Kiên | 18/11 |
| Đối chiếu tập chia ImageNet-A với code gốc | Huy | 18/11 |
| E1: chốt chuẩn hóa và γ (report3, mục 3) | Đăng | Quá hạn |

**Lưu ý cho 2 việc đầu:** file đặc trưng nằm trong `/kaggle/working/data/features/` của phiên Kaggle vừa chạy. Nếu đã **Save Version**, thêm output của version đó làm Input cho notebook mới để khỏi trích lại khoảng 1 giờ.
