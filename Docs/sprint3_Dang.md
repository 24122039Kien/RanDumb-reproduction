# Sprint 3 của Đăng: cách chạy và trạng thái bằng chứng

**Ngày:** 14/11/2026. **Nhánh:** `dang/sprint-3-experiments`, bắt đầu từ `main` tại `b57e437`.

**Trạng thái:** đã chuẩn bị code và kiểm thử; **E1 0/15, E8-B 0/24 run thật hoàn tất**. Chờ môi trường Kaggle/máy đủ tài nguyên và quyền ghi W&B. Không lấy accuracy của dữ liệu giả hoặc thử nghiệm chưa log làm kết quả tái lập.

## Cấu hình đã được Đăng xác nhận

E1 giữ **D=25.000, γ=1.0**, RFF, một lượt học, λ theo dataset và flip theo paper. Đăng chọn **L2 theo ảnh**; paper không nêu rõ phép chuẩn hóa nên L2 là quyết định của nhóm. RFF/thống kê/phép giải dùng float64; tiền xử lý ảnh vẫn float32. Seed `0,1,2` do nhóm quy định.

| Dataset | λ | Flip | Số seed | Accuracy paper (%) |
|---|---:|---|---:|---:|
| MNIST | 1e-6 | Không | 3 | 98,3 |
| CIFAR-10 | 1e-5 | Có | 3 | 55,6 |
| CIFAR-100 | 1e-5 | Có | 3 | 28,6 |
| TinyImageNet | 1e-4 | Có | 3 | 11,6 |
| miniImageNet | 1e-4 | Có | 3 | 17,7 |

Nguồn thông số: `Randumbofficial.pdf`, mục 3, Implementation Details. TinyImageNet ở Bảng 6 có cả số 11,1; bảng trên dùng 11,6 tương ứng λ=1e-4. Dạng RFF D chiều, OAS gộp của E1 và nguồn/split dữ liệu hiện tại vẫn có các điểm cần đối chiếu như nhật ký đã ghi; không tuyên bố khớp mọi chi tiết tác giả.

## Bổ sung ngày 19/11 (Kiên)

- **E8 pha A** (chuẩn hóa × γ, 32 run) được thêm vào `run_sprint3.py` để E8 phủ đủ 4 yếu tố của Instruction mục 4.6. Chọn bằng `--group E8-A`; `--group E8` chạy cả A và B. Thiết kế và lý do: [e1_e8_fidelity.md](e1_e8_fidelity.md).
- Thêm `--dim` để chia phiên E8. Chạy `--dim 5000` trước `--dim 25000`, vì λ tương đối cần run hiệu chỉnh ở D = 5.000.
- `fit_lda(..., inplace=True)` giờ ghi thừa số Cholesky đè lên `cov` để vừa VRAM của T4 ở D = 25.000.

## Thiết kế E8 pha B

24 run = CIFAR-10/100 × D∈{5.000,25.000} × shrink∈{OAS gộp, OAS từng lớp, không OAS} × {λ cố định, λ tương đối}; seed 0. Đây là phân tích thăm dò một seed, chưa đo biến thiên qua nhiều seed.

- Cố định: λ=1e-5 ở cả hai D.
- Tương đối: λ_eff = α·trace(S)/D. S là covariance đưa vào bộ phân loại, sau shrink nếu có.
- Với mỗi dataset/shrink, hiệu chỉnh **α = 1e-5 / (trace(S_ref)/5.000)** từ covariance **train** của run fixed D=5.000. Giữ α đó khi tăng D lên 25.000. Không dùng test accuracy để chọn α.
- D=5.000 relative là kiểm tra khớp mốc fixed; D=25.000 kiểm tra tác động của quy tắc regularization. D khác nhau không đảm bảo các chiều RFF lồng nhau dù cùng seed; phải nêu giới hạn này khi diễn giải.
- OAS từng lớp xử lý từng lớp lần lượt, co covariance riêng rồi gộp theo số mẫu (bao gồm flip); không giữ mọi covariance lớp trong RAM/VRAM cùng lúc. Đây là đánh giá ở cuối luồng, không bổ sung metric thích nghi từng timestep.
- Giữ L2, γ=1.0, dtype, flip như E1. Thí nghiệm E8 pha A về γ/chuẩn hóa không được tự nhận là đã thực hiện trong Sprint 3.

## Chạy trên Kaggle

Notebook: [`03_dang_sprint3.ipynb`](../notebooks/03_dang_sprint3.ipynb).

1. Import notebook, bật GPU T4 và Internet; bật Secret `WANDB_API_KEY` có quyền ghi vào project chung. Có thể thêm `HF_TOKEN` để tải dữ liệu. Không chép key vào source.
2. Nhánh `dang/sprint-3-experiments` phải có trên GitHub trước khi ô clone chạy. Notebook in commit, cài các phiên bản ghim từ requirements rồi chạy tests.
3. Hai phép đo ảnh giả ở D=25.000 kiểm tra bộ nhớ trước. Nếu OOM thì dừng để điều tra; không tự hạ D hoặc đổi dtype. Ảnh giả ít mẫu nên chưa thay thế phép đo peak trên dataset thật.
4. Chọn `GROUP`, `DATASET`, `SEED` để chia phiên nếu cần. Mặc định `all` là toàn bộ 39 run, tuần tự, không dùng hai GPU đồng thời.
5. Mỗi run ghi JSON và W&B. Sau phiên, Save Version và giữ thư mục outputs; phiên mới phải phục hồi JSON trước khi resume. Không đổi source giữa các phiên trong cùng bộ kết quả.

Lệnh tương đương (chạy từ root repo đã cài dependencies):

```bash
# Chỉ xem kế hoạch, không tải dữ liệu/chạy thí nghiệm
python scripts/run_sprint3.py --group all

# Một phiên E1 MNIST, 3 seed
python scripts/run_sprint3.py --group E1 --dataset mnist --device cuda --out-dir outputs/sprint3 --execute

# E1 còn lại: có thể chọn dataset/seed để chia phiên
python scripts/run_sprint3.py --group E1 --device cuda --out-dir outputs/sprint3 --execute

# E8-B: tự chạy fixed trước, dùng trace train làm tham chiếu cho relative
python scripts/run_sprint3.py --group E8 --device cuda --out-dir outputs/sprint3 --execute
```

Khi resume, script kiểm tra tên, source SHA256, cấu hình, accuracy hợp lệ và URL W&B trong JSON. Nếu khác thì dừng, không bỏ qua run hoặc ghi đè kết quả cũ một cách im lặng. Đây là kiểm tra bản ghi local, không tự truy vấn server để xác nhận quyền public/trạng thái W&B; cần kiểm tra link public sau run.

## Đầu ra cần có để nghiệm thu Sprint 3

- Mỗi run: JSON cấu hình, commit/source SHA256, accuracy, covariance trace trung bình, λ thực dùng, ρ hoặc ρ từng lớp, thời gian và peak memory, URL W&B. Run relative ghi thêm URL calibration.
- `summary.csv`: từng run paper/team, chênh lệch điểm phần trăm và sai lệch tương đối. Số paper trong dòng E8 chỉ là tham chiếu E1, không phải cấu hình ablation giống nhau.
- `e1_aggregate.csv`: mean và **sample standard deviation** qua 3 seed. Nếu thiếu seed, trạng thái `incomplete`, không tính mean/std chính thức. Bảng tổng hợp giữ cả các phiên dataset/seed trước trong cùng thư mục.
- Điền số đo thật và link vào `nhatky_Dang.md`/README sau khi đủ bằng chứng. Chưa đánh dấu xong các ô thực nghiệm của Sprint 3 chỉ vì code/tests qua.

## Kiểm chứng đã thực hiện trên máy Đăng

69 tests pass, 1 warning từ plugin anyio/pytest. Notebook mới đã kiểm tra cú pháp tất cả ô code; chưa chạy trên Kaggle.

Phép đo ảnh giả: RTX 3050 6 GB, D=4.096, float64, 256 ảnh train 32×32×3 (512 mẫu tính cả flip), batch 128. Không log thành kết quả thực nghiệm E1/E8.

| Shrink | Peak allocated (GiB) | Peak reserved (GiB) | Bằng chứng |
|---|---:|---:|---|
| OAS gộp | 0,477631 | 0,566406 | [JSON](evidence/sprint3/memory-4096-oas.json) |
| OAS từng lớp | 0,477662 | 0,568359 | [JSON](evidence/sprint3/memory-4096-oas-per-class.json) |

Chưa đo D=25.000 trên T4. Không ngoại suy các số trên thành cam kết đủ VRAM/thời gian. Cận dưới cho hai ma trận D×D float64 ở D=25.000 đã khoảng 9,31 GiB, chưa tính RFF và workspace của thư viện.
