# Nhật ký công việc — Đăng

> Tài liệu này ghi lại những gì đã làm, quyết định thiết kế và điều cần lưu ý.
> Cập nhật theo từng sprint.

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
