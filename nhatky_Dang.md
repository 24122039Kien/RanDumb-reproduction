# Nhật ký công việc — Đăng

> Tài liệu này ghi lại những gì đã làm, quyết định thiết kế và điều cần lưu ý.
> Cập nhật theo từng sprint.
> Cập nhật mới nhất: **14/11/2026**, xem mục **Sprint 3** cuối file.
> Những quyết định trong mục mới nhất thay thế các lựa chọn tạm ở nhật ký cũ.

---

## Sprint 0 — Khởi động (06/11/2026)

**Trạng thái: ✅ Hoàn thành**

### Đã làm

| File | Việc đã làm |
|---|---|
| `src/randumb/features.py` | Implement đầy đủ `RandomFourierFeatures` và `RPReLU` |
| `src/randumb/stats.py` | Implement đầy đủ `SufficientStats` với `update`, `finalize`, `save`, `load` |
| `src/randumb/classifier.py` | Implement đầy đủ `oas_shrink`, `fit_lda`, `predict` |
| `tests/test_features.py` | Viết Test 1 (RFF xấp xỉ kernel) và Test 6 (seed) — cả hai PASS |
| `tests/test_stats.py` | Viết Test 2 (online = batch) và Test 3 (bất biến thứ tự) — cả hai PASS |
| `tests/test_classifier.py` | Viết Test 4 (OAS khớp sklearn) và Test 5 (Mahalanobis) — cả hai PASS |

**Kết quả chạy test:**
```
9 passed, 1 warning in 7.70s
```
(9 test gồm 6 test bắt buộc + 3 test bổ sung)

**Commit:** `c822aad` — `Sprint 0/1 (Dang): implement features, stats, classifier; all 6 tests pass`

---

## Chi tiết thiết kế

### `features.py` — RandomFourierFeatures

**Công thức dùng:** dạng D chiều `sqrt(2/D) cos(ωᵀx + b)` với `b ~ U(0, 2π)`.

**Lý do chọn dạng D chiều (không phải 2D):**
- Bài báo viết φ(x) có 2D chiều (cos + sin).
- Nhưng bài cũng nói dùng RFF của scikit-learn với "25K embedding" → scikit-learn dùng dạng D chiều.
- **→ Dùng dạng D chiều để khớp con số 25K của bài. Mâu thuẫn này sẽ ghi vào phân tích sai lệch.**

**Seed:** Dùng `torch.Generator(device="cpu")` → lấy mẫu ω và b trên CPU trước → chuyển sang device.
Điều này đảm bảo cùng seed cho cùng phép chiếu trên mọi máy (CPU/GPU), nhất quán với quyết định thiết kế trong `implementation_plan.md`.

---

### `stats.py` — SufficientStats

**Công thức tích lũy theo lô:**
```
A += ΦᵀΦ
tong[c] += tổng φ của lớp c trong lô
dem[c]  += số mẫu lớp c trong lô
N       += B
---
μ_c = tong[c] / dem[c]
S   = (A - Σ_c dem[c] μ_c μ_cᵀ) / N
```

**Kiểu số:** Mặc định `float64` vì ta cộng dồn nhiều số rất nhỏ (đặc trưng RFF ~ `1/sqrt(D)`).

**Bảo vệ trường hợp lớp trống:** `dem_safe = dem.clamp(min=1.0)` tránh chia 0.

**Save/load:** Dùng `torch.save` với `weights_only=True` (tốt cho bảo mật với torch ≥ 2.0).

---

### `classifier.py` — OAS, LDA, predict

**OAS:**
```
mu    = trace(S) / p
alpha = mean(S_ij²)
rho   = min(1, (alpha + mu²) / ((n+1)(alpha - mu²/p)))
```
- Xử lý mẫu số = 0: nếu `denominator < 1e-15` hoặc NaN → `rho = 1` (co hoàn toàn về `mu·I`).
- Đây là trường hợp S = `mu·I` (ma trận bội số đơn vị), khi đó OAS không có thêm thông tin để ước lượng.

**fit_lda:**
- Giải `(cov + λI) w_c = μ_c` bằng Cholesky.
- Fallback sang `torch.linalg.lstsq` nếu ma trận không xác định dương.
- Giải một lần cho tất cả C lớp cùng lúc (rhs là ma trận `D×C`).

**predict:**
```
score_c = φᵀw_c - ½ μ_cᵀw_c
ŷ = argmax_c score_c
```
- Bỏ số hạng `φᵀS⁻¹φ` vì giống nhau ở mọi lớp.
- Triển khai vector hóa: `phi @ weights.T - 0.5 * (means * weights).sum(dim=1)`.

---

## Các điểm cần lưu ý cho sprint sau

### Điểm chưa rõ cần chốt ở Sprint 2 (E8)

| Điểm | Quyết định tạm | Cần kiểm tra |
|---|---|---|
| γ mặc định | Chưa chốt — cần chạy E8 trước E1 | Bài: 1.0; code gốc: `gamma='scale'` |
| Chuẩn hóa | Chưa chốt | z-score, L2, hay chỉ chia 255 |
| OAS: gộp chung hay từng lớp | Hiện implement gộp chung (`S` tích lũy từ tất cả lớp) | Code gốc dùng OAS từng lớp; Sprint 2 E8 sẽ so sánh |

### Mâu thuẫn trong bài báo cần ghi vào phân tích sai lệch

1. **Công thức RFF:** Bài ghi 2D chiều (cos + sin), nhưng thực tế code dùng D chiều (cos với offset). Nhóm chọn D chiều.
2. **γ:** Bài ghi 1.0, code gốc dùng `gamma='scale'`.
3. **Vị trí OAS:** Bài viết chung chung, code gốc dùng OAS từng lớp.
4. **TinyImageNet:** Bảng 3-5 ghi 11.6 còn dòng Bảng 6 ghi 11.1 (ứng với λ khác nhau).

---

## Môi trường chạy (laptop)

| Thành phần | Phiên bản |
|---|---|
| Python | 3.13.0 |
| torch | 2.13.0 (CPU) |
| numpy | 2.3.5 |
| scikit-learn | 1.8.0 |
| pytest | 9.2.0 |

> Chưa có GPU và wandb trên laptop. Run thật sẽ chạy trên Kaggle T4.

---

## Sprint 2 — Bước 1: Chốt D, kiểu số và cách chia phiên (14/11/2026)

**Nhánh làm việc:** `dang/sprint-2-config` — chứa thay đổi cấu hình, tests và nhật ký của bước này.

### Kiểm tra Đăng đã làm tới đâu

- Đã cài `features.py`, `stats.py`, `classifier.py` và các test lõi ở commit `c822aad`.
- Đã ghép với data, tracking, pipeline của Kiên và verification của Huy. `main` ở commit `3f7e19f` chứa cả ba nhánh; đã push lên GitHub trong phiên làm việc trước ngày 14/11. Hai merge commit được tạo trên máy ngày 13/11 nên GitHub hiển thị ngày đó.
- Trước bước này, toàn bộ **31 tests pass**; phần lõi và tích hợp của Đăng trong Sprint 1 đã có. Không vì vậy mà coi cả Sprint 1 đã hoàn tất: chưa có bằng chứng trong repo về run MNIST/CIFAR-10 trên dữ liệu thật kèm W&B để đạt điều kiện nghiệm thu sprint.
- `Docs/report2.md` chỉ ghi smoke test môi trường Kaggle, tốc độ phép toán và kiểm tra nguồn dataset. Đây chưa phải kết quả E1 hay E8.
- Chưa có bằng chứng Đăng đã hoàn thành E8 pha A/B hoặc các run E1. Các ô về thực nghiệm đó giữ trạng thái chưa xong.

**Phạm vi lần này:** chỉ thực hiện việc đầu của Đăng ở Sprint 2: chốt D, kiểu số và cách chia phiên, rồi đưa các mặc định tương ứng vào CLI. Chưa chạy E8/E1, chưa tối ưu bộ nhớ hay triển khai OAS theo lớp.

### Thông số đối chiếu trực tiếp với paper

Nguồn: `Randumbofficial.pdf` trong repo, **arXiv:2402.08823v3, 20/11/2024**, phần Implementation Details ở mục 3, trang 5–6; số lần khởi tạo ở Checklist câu 7, trang 15. File hiện tại là v3, không phải v1 như một số ghi chú cũ trong tài liệu nhóm.

| Thông số E1 | Giá trị chốt | Căn cứ |
|---|---|---|
| Biểu diễn | Random Fourier Features xấp xỉ RBF | Mục 2 và Implementation Details |
| Số chiều đầu ra | **25.000** cho cả 5 dataset | Paper ghi 25K embedding size |
| γ | **1.0** | Implementation Details; không thay bằng `scale` để tăng accuracy |
| λ MNIST | **1e-6** | Implementation Details |
| λ CIFAR-10 / CIFAR-100 | **1e-5** | Implementation Details |
| λ TinyImageNet / miniImageNet | **1e-4** | Implementation Details |
| Flip ngang khi train | MNIST: không; 4 dataset màu: có | Một ảnh gốc kèm bản flip, không thêm RandomCrop |
| Đầu vào | MNIST 784 chiều; ảnh màu 3.072 chiều | MNIST 28×28×1; Tiny/mini resize 32×32×3 |
| Số lượt học | Một lượt qua train | Không thêm epoch; đánh giá sau khi học hết |
| Bộ nhớ mẫu cũ của learner | Không lưu exemplars để replay | Giữ thống kê tích lũy |
| Số khởi tạo kernel | 3 | Checklist câu 7; paper không cho biết ba giá trị seed cụ thể |

**Ưu tiên yêu cầu mới của Đăng:** giữ các thông số paper đã công bố. Không áp dụng phương án cũ giảm D xuống 15.000. E8 về sau chỉ là phân tích độ nhạy, không dùng kết quả E8 để âm thầm thay γ hoặc λ của run E1 đối chiếu paper.

### Quyết định tự chốt cho phần phối hợp với Kiên

Đăng cho phép tự quyết các nội dung cần chốt với Kiên. Các lựa chọn dưới đây là quyết định thực hiện trong phiên này, **không phải xác nhận Kiên đã trao đổi hoặc đồng ý**, cũng không phải thông số tác giả đã công bố:

| Nội dung | Quyết định và giới hạn |
|---|---|
| Kiểu số | `float64` cho RFF, thống kê và phép giải. Paper không nêu dtype; chọn để ưu tiên độ chính xác số và giữ nhất quán giữa các run. Phần chuẩn hóa ảnh hiện vẫn tạo tensor `float32`, sau đó mới chuyển sang dtype RFF. Không gọi pipeline này là float64 toàn bộ từ tiền xử lý. |
| Seed cụ thể | `0, 1, 2`, theo quy ước của nhóm. Cùng số seed với paper nhưng không biết có trùng giá trị seed tác giả hay không. |
| Chuẩn hóa | Giữ `zscore` theo kênh, tính mean/std từ train và dùng lại cho test, như code nhóm đang có. Paper chỉ nói normalize, không công bố phép chuẩn hóa cụ thể; chưa thể xác nhận khớp hoàn toàn với tác giả. |
| Dạng RFF | Giữ dạng D chiều có pha ngẫu nhiên như cài đặt hiện tại để diễn giải 25K là số chiều đầu ra. Paper trình bày cos+sin ở mục 2 nhưng mô tả dùng scikit-learn ở phần thực nghiệm; ghi nhận sự không nhất quán này, không khẳng định hai cách cho cùng kết quả từng bit. |
| OAS | Giữ OAS trên covariance trong lớp gộp của code hiện tại. Paper không mô tả đủ chi tiết vị trí shrink; kiểm chứng khác biệt theo lớp thuộc bước sau. |
| Batch tính toán | 512 ảnh gốc/lô, tương ứng tối đa 1.024 ảnh sau flip. Đây là lựa chọn triển khai để tích lũy thống kê, không phải đúng giao thức một mẫu mỗi timestep của paper; test online/batch đã kiểm tra tính tương đương trên dữ liệu nhỏ. |
| Tập đánh giá | E1 dùng toàn bộ train và split test do loader hiện tại quy định; không giới hạn số mẫu, không dùng validation để chọn lại thông số E1. Nguồn/split HF, đặc biệt miniImageNet, vẫn cần kiểm chứng theo `Docs/e2_datasets.md`. |
| Phiên chạy | **1 dataset × 1 seed / 1 tiến trình**, lần lượt MNIST → CIFAR-10 → CIFAR-100 → TinyImageNet → miniImageNet; mỗi dataset chạy seed 0, 1, 2, tổng **15 run**. Thoát tiến trình giữa các run để giải phóng bộ nhớ. Không chạy đồng thời các run D=25K trên cùng thiết bị. |
| Lưu kết quả | Mỗi run ghi W&B riêng, `group=E1`, tag `lab:reproduction`, tên chứa dataset/D/seed. Không mặc định lưu ma trận thống kê lớn ra đĩa. |
| Tài nguyên | Với code hiện tại, dự trù host CPU **64 GiB RAM trở lên** cho run đầy đủ; đây là mức dự trù, chưa phải kết quả đo peak hay máy đã được cấp. Kaggle T4 chỉ là ứng viên sau khi xử lý và đo bộ nhớ. Không tự hạ D hoặc đổi dtype để vừa máy. |

Như vậy, các giá trị paper **công bố rõ** đã được chốt đúng. Những chi tiết paper bỏ ngỏ và khác biệt triển khai vẫn được ghi nhận; chưa có cơ sở tuyên bố tái lập khớp hoàn toàn hoặc sẽ đạt đúng accuracy của bài.

### Vì sao chưa chạy D = 25.000 ngay trên máy này

- Kiểm tra trực tiếp ngày 14/11: Python **3.11.6**, torch **2.11.0+cu128**, CUDA khả dụng; GPU **NVIDIA GeForce RTX 3050 6GB Laptop GPU**; RAM tổng **11,71 GiB**. Thông tin này thay thế bảng môi trường laptop cũ ở trên cho phiên hiện tại.
- Một ma trận 25.000² `float64` chiếm **5.000.000.000 byte**, khoảng **4,66 GiB**. Code hiện tại giữ `A`, covariance và nhiều ma trận tạm, chưa kể RFF và dữ liệu.
- Báo cáo của Kiên đã chỉ ra nguy cơ vượt 16 GB trên T4. Chia thành 15 tiến trình giúp giải phóng bộ nhớ giữa các run nhưng **không giải quyết peak trong một run**. Hai T4 cũng không tự gộp thành một vùng VRAM cho code một thiết bị này.
- Việc chốt cấu hình đã xong; việc đo thời gian/peak ở D lớn và xác nhận nơi chạy vẫn còn mở. Không ghi thời gian ngoại suy thành số đo thực, không đánh dấu đã chạy E1.

### Thay đổi đã thực hiện

1. `scripts/run_experiment.py`: đổi γ mặc định từ `scale` sang **1.0**; thêm λ mặc định theo từng dataset đúng bảng trên. Có thể truyền rõ `--gamma`/`--lam` cho các thực nghiệm độ nhạy riêng.
2. Thêm `--feature-dtype` mặc định `float64`, truyền vào `PipelineConfig.feature_dtype`. Trước đây CLI `--dtype` chỉ ảnh hưởng thống kê và phép giải, RFF vẫn mặc định `float32`. Cấu hình W&B nay có cả `dtype`, `feature_dtype` và giá trị λ đã resolve trước khi log.
3. `tests/test_run_experiment.py`: thêm 5 trường hợp kiểm tra cấu hình mặc định của từng dataset thực sự truyền tới pipeline, cộng 1 trường hợp đảm bảo vẫn truyền được cấu hình độ nhạy. Test dùng dữ liệu giả, không tải dataset hoặc cấp phát D=25K.
4. Đồng bộ các ô công việc đã có bằng chứng của Đăng trong `Docs/implementation_plan.md`.

Lệnh tham chiếu cho E1 (chưa chạy; dùng trên host đã kiểm tra đủ tài nguyên). Mỗi dòng dưới là seed 0; lặp riêng với `--seed 1` và `--seed 2`:

```bash
python scripts/run_experiment.py --dataset mnist --dim 25000 --gamma 1.0 --lam 1e-6 --no-flip --norm zscore --shrink oas --dtype float64 --feature-dtype float64 --batch-size 512 --device cpu --seed 0 --group E1 --tag lab:reproduction
python scripts/run_experiment.py --dataset cifar10 --dim 25000 --gamma 1.0 --lam 1e-5 --flip --norm zscore --shrink oas --dtype float64 --feature-dtype float64 --batch-size 512 --device cpu --seed 0 --group E1 --tag lab:reproduction
python scripts/run_experiment.py --dataset cifar100 --dim 25000 --gamma 1.0 --lam 1e-5 --flip --norm zscore --shrink oas --dtype float64 --feature-dtype float64 --batch-size 512 --device cpu --seed 0 --group E1 --tag lab:reproduction
python scripts/run_experiment.py --dataset tiny-imagenet --dim 25000 --gamma 1.0 --lam 1e-4 --flip --norm zscore --shrink oas --dtype float64 --feature-dtype float64 --batch-size 512 --device cpu --seed 0 --group E1 --tag lab:reproduction
python scripts/run_experiment.py --dataset mini-imagenet --dim 25000 --gamma 1.0 --lam 1e-4 --flip --norm zscore --shrink oas --dtype float64 --feature-dtype float64 --batch-size 512 --device cpu --seed 0 --group E1 --tag lab:reproduction
```

### Kiểm chứng sau thay đổi

Chạy toàn bộ tests bằng Python 3.11 đã có thư viện (lệnh `python` mặc định trên máy trỏ tới MSYS2, thiếu pytest):

```powershell
& 'C:/Users/Hai Dang/AppData/Local/Programs/Python/Python311/python.exe' -m pytest -q
```

Kết quả: **37 passed, 1 warning in 12.30s**. Warning đến từ plugin `anyio` sử dụng API pytest sẽ bị bỏ ở phiên bản sau; không có test thất bại. Đây là kiểm thử code/cấu hình trên dữ liệu giả, không phải số liệu tái lập paper.

**Điểm dừng:** đã hoàn thành đúng một bước tiếp theo — chốt và áp dụng cấu hình. Công việc sau cần xử lý điều kiện bộ nhớ và đo thực tế trước các run lớn; E8/E1 vẫn chưa thực hiện trong phiên này.

---

## Sprint 3 — E1 và E8 pha B (14/11/2026, đang thực hiện)

**Phạm vi mới do Đăng yêu cầu:** chỉ Sprint 3 của Đăng; thay thế yêu cầu trước đó làm toàn bộ các sprint. Bao gồm E1 trên 5 dataset × 3 seed và E8 pha B về shrink/λ. Các sửa lỗi và tối ưu bộ nhớ cần để chạy đúng các thực nghiệm này là phần chuẩn bị; không tính là đã hoàn thành run.

### Đồng bộ và quyết định đã được xác nhận

- Đã `fetch` GitHub và cập nhật bản làm việc từ `main` mới nhất, commit **`b57e437`**, có `Docs/report3.md` và code E2 của Kiên. Dùng repo hiện có, không tạo thêm bản clone trùng.
- Nhánh thực hiện: **`dang/sprint-3-experiments`**, tạo từ `main` mới nhất.
- **Đăng đã chọn phương án 1: γ = 1.0 + chuẩn hóa L2 cho E1**, và yêu cầu ghi rõ vào nhật ký. Quyết định này thay thế lựa chọn z-score tạm của Sprint 2. D = 25.000 và λ theo dataset tiếp tục giữ đúng paper; không chuyển γ sang `scale` như đề xuất của Kiên.
- **Giới hạn:** paper chỉ ghi normalize, không xác nhận dùng L2. L2 là lựa chọn được Đăng chốt, có động cơ từ thử nghiệm sơ bộ của Kiên; không gọi đó là thông số tác giả đã công bố. Các số 9,82%/45,51% trong báo cáo Kiên chưa có W&B, chưa dùng làm kết quả E1/E8 của Đăng.
- Giữ RFF, thống kê và phép giải `float64`; seed nhóm `0,1,2`; flip theo paper. Phần tiền xử lý vẫn tạo tensor `float32` như đã ghi ở Sprint 2.
- Trước lần xác nhận lại yêu cầu, thao tác cập nhật/tạo nhánh từng bị từ chối ở hộp quyền. Sau khi Đăng yêu cầu lại làm Sprint 3, thao tác cập nhật và tạo nhánh mới đã được cho phép và hoàn tất.

### Môi trường thực nghiệm còn cần

Máy hiện tại có RTX 3050 6 GB và RAM 11,71 GiB; chưa có credential Kaggle/W&B cho script. Đã hỏi Đăng về notebook Kaggle có GPU và Secret `WANDB_API_KEY` hoặc host khác đủ tài nguyên. Chưa nhận thông tin này nên chưa chạy full E1/E8, chưa phát sinh run hay số liệu W&B. Không lưu hoặc yêu cầu ghi khóa vào source/chat.

### Phát hiện khi kiểm tra phần lõi

Kiểm tra dữ liệu giả trên covariance `diag(1,2,4)`, n=100: OAS hiện tại cho `rho=0.14851485148514856`, nhưng khi nhân covariance với `1e-10` lại cho `rho=1.0`. Theo công thức OAS, hệ số co phải không đổi khi nhân toàn bộ covariance với một hằng dương. Nguyên nhân là điều kiện tuyệt đối `abs(denominator) < 1e-15`; điều này có thể làm sai các covariance nhỏ. Đây là lỗi được tái hiện trên dữ liệu giả, chưa phải đo tác động lên accuracy dữ liệu thật.

**Trạng thái run lúc bắt đầu:** E1 0/15 hoàn tất; E8 pha B chưa chạy. Các mục dưới sẽ cập nhật theo kết quả thực tế.

### Các việc đã làm để chuẩn bị chạy Sprint 3

1. **Áp dụng quyết định γ=1.0 + L2:** đổi chuẩn hóa mặc định của `scripts/run_experiment.py` sang `l2`. D và λ E1 giữ như Sprint 2. Các lệnh z-score trong phần nhật ký Sprint 2 chỉ là lịch sử, được thay bằng cấu hình L2 hiện tại.
2. **Sửa OAS:** bỏ ngưỡng tuyệt đối `1e-15`; viết công thức theo tỉ số `mu²/alpha` để hệ số co không bị ép thành 1 chỉ vì covariance có thang nhỏ. Tính Frobenius norm bằng reduction, không tạo `cov**2` kích thước D×D. Test đối chiếu cả covariance và ρ với `sklearn.OAS` ở ba thang 1, 1e-10, 1e10.
3. **Giảm ma trận tạm:** tích lũy `A.addmm_`; `finalize(consume=True)` dùng lại A làm covariance, hiệu chỉnh trung bình bằng addmm tại chỗ; OAS và ridge sửa trực tiếp covariance khi pipeline cho phép. Giải phóng covariance sau giải hệ. API mặc định của các hàm vẫn giữ đầu vào để tương thích với người gọi khác. Sau consume, update/finalize/save bị từ chối để tránh tái sử dụng nhầm thống kê.
4. **Cài OAS theo lớp cho E8-B:** mỗi lớp một thống kê tạm, co covariance rồi gộp theo số mẫu (gồm ảnh flip). Không giữ C ma trận covariance. Giữ trung bình lớp riêng và log `rho_per_class`; không giả vờ tồn tại một ρ duy nhất cho cả phép gộp này. Test đối chiếu với các OAS sklearn riêng trên hai lớp có kích thước/phương sai khác nhau.
5. **Cài λ tương đối:** `--lam-mode relative --lam alpha` dùng `lambda_effective = alpha * trace(S)/D`. CLI yêu cầu alpha rõ ràng, không dùng nhầm λ paper làm alpha. Log mean variance và λ thực dùng. Test tính bất biến dự đoán khi đổi thang toàn bộ đặc trưng.
6. **Lưu bằng chứng run:** `--output` ghi JSON qua file tạm rồi replace sau khi run hoàn tất; config có commit, trạng thái dirty và SHA256 source, dtype của RFF/thống kê, λ đầu vào và calibration URL. Lỗi pipeline đánh dấu W&B run thất bại. Không lưu credential vào bản ghi.
7. **Bộ chạy `scripts/run_sprint3.py`:** E1 gồm 15 run; E8-B gồm 24 run theo thiết kế bên dưới. Mỗi run một subprocess; chạy tuần tự. Mặc định chỉ in kế hoạch; `--execute` mới chạy. Có bộ lọc dataset/seed để chia phiên. Resume chỉ nhận JSON đúng source/cấu hình và có URL W&B; source hoặc cấu hình khác thì dừng, không ghi đè kết quả cũ.
8. **Bảng tổng hợp:** xuất từng run vào `summary.csv`, chênh lệch điểm phần trăm và sai lệch tương đối. `e1_aggregate.csv` chỉ tính mean/sample std khi đủ seed 0,1,2; thiếu seed thì ghi incomplete. Giữ dữ liệu các phiên trước khi chạy thêm dataset/seed.
9. **Notebook `notebooks/03_dang_sprint3.ipynb`:** clone nhánh Sprint 3, cài dependencies ghim, đọc Kaggle Secrets, chạy tests và kiểm tra memory ảnh giả D=25.000 trước khi chạy dữ liệu thật. Đã compile mọi ô code; chưa chạy notebook trên Kaggle.
10. **Tài liệu:** thêm `Docs/sprint3_Dang.md` với cấu hình, lệnh chạy, thiết kế E8-B, cách chia phiên/resume và tiêu chí nghiệm thu. Ghi trạng thái chuẩn bị trong kế hoạch; chưa đánh dấu các run thật là xong.

**Lưu ý flip:** code hiện duyệt batch ảnh gốc rồi batch ảnh flip riêng; batch 512 là tối đa 512 ảnh trong mỗi lần transform, không phải ghép 1.024 ảnh cùng lúc như mô tả tạm trong bảng Sprint 2. Tổng số mẫu tích lũy vẫn gấp đôi với dataset có flip.

### Thiết kế E8-B được chốt cho Sprint 3

- CIFAR-10 và CIFAR-100; D=5.000 và 25.000; ba chế độ `oas`, `oas_per_class`, `none`; hai quy tắc λ; seed 0: **24 run**. Đây là thiết kế của nhóm, không phải cấu hình mặc định paper; một seed chỉ cho phân tích thăm dò.
- Fixed dùng λ=1e-5. Relative dùng α=1e-5/(trace(S_ref)/5.000), S_ref lấy từ **train** của run fixed D=5.000 tương ứng dataset/shrink. Giữ nguyên α khi đổi D lên 25.000. Không chọn alpha theo accuracy test.
- Chạy relative ở D=5.000 để kiểm tra khớp mốc fixed; chạy D=25.000 để so hiệu ứng của quy tắc λ. Ghi calibration run vào config/JSON/W&B.
- RFF ở các D không đảm bảo là các tập chiều lồng nhau dù cùng seed; không diễn giải khác biệt thành tác động thuần túy của riêng λ khi chưa xem cả yếu tố D.
- E1 vẫn dùng γ=1.0, L2, D=25.000, OAS gộp, λ theo dataset, 3 seed; E8 không tự đổi cấu hình E1 sau khi xem test.

### Kiểm chứng thực tế và giới hạn

- Toàn bộ **69 tests passed, 1 warning in 8.49s** trên Python 3.11.6, torch 2.11.0+cu128. Warning là plugin anyio dùng API pytest sắp bỏ; không có test thất bại. Có kiểm tra tích hợp suite bằng subprocess giả để chứng minh chạy một lần rồi resume không chạy trùng; đây không phải thực nghiệm thật.
- Đã đo pipeline thật với **ảnh giả**, GPU RTX 3050 6 GB, D=4.096, float64, batch 128, 256 ảnh train (512 mẫu gồm flip). OAS gộp: peak allocated **0,477631 GiB**, reserved **0,566406 GiB**. OAS theo lớp: allocated **0,477662 GiB**, reserved **0,568359 GiB**.
- JSON gốc nằm ở `Docs/evidence/sprint3/memory-4096-oas.json` và `memory-4096-oas-per-class.json`, có phiên bản môi trường, thời gian, source SHA256. Phép đo được làm khi code chưa commit nên `git_dirty=true`; hash source ghi nhận đúng nội dung lúc đo.
- Chưa đo D=25.000 trên T4; không suy ra T4 chắc chắn đủ VRAM từ phép đo D=4.096. Giới hạn còn lại gồm workspace Cholesky và dữ liệu thật. Notebook có bước kiểm tra D=25.000 riêng và dừng nếu lỗi.
- **Chưa có kết quả E1 hoặc E8-B trên dữ liệu thật, chưa có URL W&B mới, chưa có bảng paper/team hoàn chỉnh.** Chưa chạy E8 pha A (ngoài phạm vi Sprint 3 lần này). Không dùng số giả để lấp bảng hoặc đánh dấu sprint đã xong.

### Việc cần Đăng cung cấp để chạy tiếp

Đã gửi câu hỏi xin **link notebook Kaggle có quyền chỉnh sửa, GPU T4 và Secret WANDB_API_KEY**, hoặc thông tin môi trường khác đủ tài nguyên. Chưa có câu trả lời ở thời điểm cập nhật này. Source/kiểm thử/notebook đã sẵn sàng cho bước chạy; trạng thái thực nghiệm vẫn là **E1 0/15, E8-B 0/24**.
