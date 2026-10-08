# 1D-cVAE Sinh Gia Tốc Có Điều Kiện & Đánh Giá TSTR trên PPG-DaLiA

> **Tiểu luận cuối khóa: Trí tuệ nhân tạo cho IoT (AIoT)**  
> **Sinh viên**: Nguyễn Bách Tùng – MSSV: 23110166  
> **Giảng viên hướng dẫn**: ThS. Hồ Nhựt Minh  
> **Căn cứ**: Đề cương đã chỉnh sửa và Phiếu nhận xét giải trình (tháng 09/2026).

---

## 1. Giới thiệu dự án

Dự án nghiên cứu khả năng sinh dữ liệu gia tốc ba trục ($x, y, z$) cổ tay theo 8 hoạt động con người bằng mô hình **1D-cVAE (1D Conditional Variational Autoencoder)** và đánh giá chất lượng thực nghiệm qua giao thức **TSTR (Train on Synthetic, Test on Real)** trên tập dữ liệu chuẩn **PPG-DaLiA**.

### Điểm đặc trưng & Quy tắc sống còn
- **Nguồn cảm biến**: Cảm biến gia tốc cổ tay (`signal/wrist/ACC`, 32 Hz).
- **Nhãn hoạt động**: Lấy từ trường `activity` (4 Hz), đồng bộ lên 32 Hz bằng Zero-Order Hold (ZOH: `activity[floor(i/8)]`), loại mã 0, ánh xạ 1–8 → 0–7.
- **Trường `label`**: Chứa nhịp tim (ECG HR ground truth), **tuyệt đối không dùng làm nhãn HAR**.
- **Cửa sổ**: Chiều dài 128 mẫu (4.0s tại 32 Hz), bước 64 mẫu (50% chồng lấn). Chỉ giữ cửa sổ 100% đồng nhất 1 nhãn duy nhất. Không ghép nối các khoảng thời gian rời rạc.
- **Bảo toàn thành phần DC**: Không lọc thông cao 0.5 Hz, không trừ mean từng cửa sổ để giữ nguyên tư thế tĩnh và trọng lực.
- **Phân chia không rò rỉ**: Train S1–S11, Validation S12–S13, Test S14–S15. Scaler Z-Score fit **chỉ trên Train S1–S11**.
- **Kiến trúc cVAE**: 4 tầng Conv1d và 4 tầng ConvTranspose1d với `output_padding=1` ở cả 4 tầng (bắt buộc để ra đúng độ dài 128).
- **Hàm mất mát cVAE**: $\mathcal{L}_t = 0.5 \mathcal{L}_{\text{rec}} + (\beta_t / 384) \mathcal{L}_{\text{KL}}$, warm-up tuyến tính $\beta_t$ trong 25 epoch, validation tính với $\beta = 1.0$ cố định, early stopping bắt đầu sau epoch 25.
- **5 nhánh đối chứng**: TRTR, Real+ROS, Real+TradAug (jitter $\sigma=0.01$, scale $U(0.95, 1.05)$), TSTR, Real+cVAE. Tất cả dùng chung ngân sách và chọn checkpoint bằng Macro-F1 trung bình theo người trên tập Val: $(F_{S12} + F_{S13}) / 2$.
- **Đánh giá MMD² & Welch PSD**: Kernel RBF 3 băng thông $\{0.5\sigma_0, \sigma_0, 2\sigma_0\}$, Welch PSD có cờ xử lý công suất 0, gọi đúng hệ số xác định $R^2$.

---

## 2. Cấu trúc thư mục

```
CUOI_KY_IOT/
├── NguyenBachTung_1D_cVAE_PPG_DaLiA.ipynb # Jupyter Notebook tổng hợp toàn bộ mã nguồn & kết quả
├── NguyenBachTung_BaoCao_CuoiKy.docx     # Báo cáo cuối kỳ bản Word hoàn chỉnh
├── REFERENCES.md                         # Danh mục tài liệu tham khảo & đối chiếu trích dẫn chuẩn IEEE
├── configs/
│   └── config.yaml                       # Toàn bộ siêu tham số tập trung
├── src/                                  # Mã nguồn module Python
│   ├── download_dataset.py               # Tải và giải nén PPG-DaLiA tự động từ UCI
│   ├── data_loader.py                    # Tiền xử lý, ZOH, cắt cửa sổ, fit scaler
│   ├── models_cvae.py                    # 1D-cVAE và HAR Classifier chuẩn
│   ├── train_cvae.py                     # Huấn luyện cVAE, beta warm-up
│   ├── train_classifier.py               # Huấn luyện 5 nhánh phân loại
│   ├── eval_tstr.py                      # Đánh giá MMD biased, Welch PSD L1, test Macro-F1
│   └── realtime_server.py                # Máy chủ IoT suy luận thời gian thực
├── checkpoints/                          # Checkpoints mô hình và scaler
│   ├── scaler.pkl                        # Z-score scaler fit trên S1-S11
│   ├── cvae_acc_dalia_best.pth           # Trọng số cVAE tốt nhất
│   └── classifier_*_best.pth             # Trọng số các bộ phân loại
├── data/
│   └── processed_cache.npz               # Dữ liệu cache nén chạy ngay trong 0.5s
├── report_assets/figures/                # 8 hình ảnh đồ thị khoa học 300 DPI
├── results/                              # Sản phẩm bàn giao báo cáo & dữ liệu
│   ├── split_manifest.json               # Danh sách phân chia đối tượng
│   ├── window_statistics.csv             # Thống kê số cửa sổ giữ/loại
│   ├── raw_predictions.csv               # Dự đoán thô từng cửa sổ test
│   ├── metrics_summary.csv               # Tổng hợp chỉ số F1, MMD, PSD
│   └── distribution_metrics.csv          # Chỉ số MMD² và Welch PSD theo lớp
├── logs/                                 # Nhật ký huấn luyện chi tiết
├── requirements.txt                      # Danh mục thư viện phụ thuộc
└── README.md                             # Hướng dẫn chạy và tái lập
```

---

## 3. Cài đặt môi trường & Kích hoạt GPU

Máy tính của bạn có card đồ họa **NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)**. Để chạy trên GPU thay vì CPU, cài đặt PyTorch hỗ trợ CUDA:

```bash
# Cài đặt PyTorch với CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài đặt các thư viện bổ trợ
pip install -r requirements.txt
```

Kiểm tra GPU hoạt động:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 4. Hướng dẫn thực thi từng bước

### Bước 0: Tải dữ liệu PPG-DaLiA từ UCI
Chạy script tự động tải và giải nén các file `S1.pkl` ... `S15.pkl` vào thư mục `data/`:
```bash
python src/download_dataset.py
```
*(Nếu bạn đã có sẵn file trên máy, chỉ cần copy các thư mục `S1` ... `S15` hoặc các file `S*.pkl` vào thư mục `data/`)*.

### Bước 1: Tiền xử lý & Trích xuất cửa sổ
Kiểm tra pipeline dữ liệu, fit scaler trên S1–S11, lưu `checkpoints/scaler.pkl`, `results/split_manifest.json` và `results/window_statistics.csv`:
```bash
python src/data_loader.py
```

### Bước 2: Huấn luyện mô hình 1D-cVAE
Huấn luyện mạng cVAE sinh tín hiệu với lịch warm-up $\beta$ và lưu checkpoint `checkpoints/cvae_acc_dalia_best.pth`:
```bash
python src/train_cvae.py --seed 2026
```

### Bước 3: Huấn luyện 5 Nhánh Bộ Phân Loại HAR
Sinh dữ liệu tổng hợp từ prior $\mathcal{N}(0, I)$, thực hiện cân bằng quota mẫu và huấn luyện 5 nhánh:
```bash
python src/train_classifier.py --seed 2026
```

### Bước 4: Đánh giá Toàn diện (TSTR, MMD², Welch PSD)
Suy luận trên tập Test hold-out (S14, S15), tính Macro-F1 theo người, xuất `raw_predictions.csv`, đo MMD² đa băng thông và khoảng cách phổ $d_{\text{PSD}}$:
```bash
python src/eval_tstr.py
```

---

---

## 5. Kết quả Thực nghiệm Thực tế (Seed 2026 trên GPU RTX 3050 Ti)

### Bảng 1: Hiệu năng Phân loại HAR trên Tập Test (S14, S15)

| Nhánh Thực nghiệm | Macro-F1 (S14) | Macro-F1 (S15) | Macro-F1 TB ((S14+S15)/2) | Accuracy | Số bước cập nhật (Steps) | $\Delta\text{TSTR}$ | $G_{\text{aug}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. TRTR (Real Train)** | 0.6687 | 0.7171 | **0.6929** | 68.38% | 15,148 | — | — |
| **2. Real + ROS** | 0.6786 | 0.7363 | **0.7074** | 69.39% | 28,752 | — | +0.0145 (+1.45%) |
| **3. Real + TradAug** | 0.6387 | 0.6971 | **0.6679** | 63.62% | 16,772 | — | -0.0250 (-2.50%) |
| **4. TSTR (100% Synthetic)** | 0.3708 | 0.4328 | **0.4018** | 39.28% | 5,410 | **+0.2911** | -0.2911 (-29.11%) |
| **5. Real + cVAE** | 0.6122 | 0.7120 | **0.6621** | 63.90% | 10,782 | — | -0.0309 (-3.09%) |

> **Nhận xét kết quả thực nghiệm:**
> - **$\Delta\text{TSTR} = F_{\text{TRTR}} - F_{\text{TSTR}} = +0.2911$**: Bộ phân loại học trên 100% dữ liệu sinh từ cVAE đạt Macro-F1 40.18% trên người thật chưa từng thấy (S14, S15). Mức sụt giảm 29.11% phản ánh hiện tượng *mode collapse* nhẹ và sự phức tạp của tín hiệu gia tốc hoạt động đời thực (in-the-wild).
> - **$G_{\text{aug}} = -0.0309$**: Khi bổ sung mẫu sinh cVAE vào tập train thật, mô hình đạt Macro-F1 66.21%, thấp hơn TRTR 3.09%, trong khi phương pháp nội suy mẫu đơn giản ROS đạt cải thiện nhẹ (+1.45%). Kết quả này cung cấp cơ sở giải trình thực nghiệm trung thực, khách quan theo đúng tiêu chí khoa học của đề cương.

---

### Bảng 2: Độ tương đồng Phân phối MMD² (108D) và Welch PSD theo từng Lớp

| Lớp (ID) | Hoạt động | $\text{MMD}^2(\text{Test, Syn})$ | $\text{MMD}^2(\text{Test, Train})$ [Ref] | $d_{\text{PSD}}$ (L1 Shape) | $R^2$ PSD | Tổng $P_f$ (Syn) | Cờ công suất 0 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Sitting | 0.43287 | 0.06201 | 0.02614 | 0.9996 | 0.6054 | Hợp lệ |
| 1 | Stairs | 0.28603 | 0.02724 | 0.25308 | 0.9704 | 1.0389 | Hợp lệ |
| 2 | Table soccer | 0.26876 | 0.01900 | 0.28615 | 0.9659 | 0.8464 | Hợp lệ |
| 3 | Cycling | 0.66471 | 0.02568 | 0.78312 | 0.0945 | 0.6152 | Hợp lệ |
| 4 | Driving | 0.35930 | 0.02449 | 0.10416 | 0.9961 | 0.6330 | Hợp lệ |
| 5 | Lunch break | 0.12089 | 0.01442 | 0.04106 | 0.9994 | 0.6079 | Hợp lệ |
| 6 | Walking | 0.27338 | 0.02350 | 0.17644 | 0.9876 | 1.0264 | Hợp lệ |
| 7 | Working | 0.16887 | 0.01951 | 0.02733 | 0.9997 | 0.6592 | Hợp lệ |
| **TB** | **Trung bình 8 lớp** | **0.32185** | **0.02698** | **0.21218** | **0.8767** | — | **0 lỗi** |

---

## 6. Danh mục Sản phẩm Bàn giao (Mục 5.2 Đề cương)

Toàn bộ các sản phẩm cam kết trong Mục 5.2 Đề cương đã được tạo lập, nghiệm thu và lưu trữ đầy đủ:
1. **Mã nguồn đầy đủ** (5 module chuẩn):
   - `src/data_loader.py`: Xử lý ZOH, lọc cửa sổ 100% nhãn duy nhất, fit Z-score scaler trên S1–S11, lưu cache nhị phân.
   - `src/models_cvae.py`: Mạng 1D-cVAE chuẩn 4 tầng Conv1d/ConvTranspose1d (`output_padding=1`) và HARClassifier.
   - `src/train_cvae.py`: Huấn luyện cVAE, lịch warm-up $\beta_t$ trong 25 epoch, validation tính với $\beta=1.0$ cố định.
   - `src/train_classifier.py`: Huấn luyện 5 nhánh phân loại với quota mẫu cân bằng chặt chẽ, ghi nhận bước cập nhật.
   - `src/eval_tstr.py`: Đánh giá Macro-F1 theo người trên S14/S15, tính MMD² 108D đa băng thông và Welch PSD.
2. **Cấu hình tập trung**: `configs/config.yaml`.
3. **Môi trường**: `requirements.txt`.
4. **Tệp chuẩn hóa**: `checkpoints/scaler.pkl`.
5. **Checkpoints mô hình** (Lưu tại `checkpoints/`):
   - `cvae_acc_dalia_best.pth`: Trọng số cVAE tốt nhất (7.8 MB).
   - `classifier_trtr_best.pth`: Bộ phân loại TRTR (216 KB).
   - `classifier_ros_best.pth`: Bộ phân loại Real+ROS (216 KB).
   - `classifier_tradaug_best.pth`: Bộ phân loại Real+TradAug (216 KB).
   - `classifier_tstr_best.pth`: Bộ phân loại TSTR (216 KB).
   - `classifier_cvaeaug_best.pth`: Bộ phân loại Real+cVAE (216 KB).
6. **Bảng phân chia & Thống kê dữ liệu**:
   - `results/split_manifest.json`: Phân chia rõ ràng S1–S11 (train), S12–S13 (val), S14–S15 (test).
   - `results/window_statistics.csv`: Thống kê chi tiết 34,585 cửa sổ train, 6,302 cửa sổ val, 6,110 cửa sổ test.
7. **Dữ liệu dự đoán thô & Báo cáo chỉ số**:
   - `results/raw_predictions.csv`: 30,550 dòng dự đoán chi tiết từng cửa sổ test cho 5 nhánh kèm window_idx, start_sample_idx, subject_id, true_label, pred_label.
   - `results/metrics_summary.csv`: Bảng tổng kết F1 từng đối tượng, Macro-F1 trung bình, Accuracy, Actual Steps, $\Delta\text{TSTR}$, $G_{\text{aug}}$.
   - `results/distribution_metrics.csv`: Bảng chi tiết $\text{MMD}^2$, $d_{\text{PSD}}$, $R^2$, và công suất phổ từng lớp.

---

## 7. Triển khai Thực tế trên Thiết bị Phần cứng (Live Smartphone Edge Demo)

Dự án cung cấp sẵn máy chủ thời gian thực biến chiếc **Điện thoại thông minh** (iPhone / Android) thành một thiết bị cảm biến đeo IoT hoàn chỉnh với chi phí **0 VNĐ**:

```bash
python src/realtime_server.py
```

### Quy trình hoạt động:
1. Máy tính và Điện thoại kết nối chung một mạng Wi-Fi.
2. Trình duyệt điện thoại mở địa chỉ: `http://<IP_MÁY_TÍNH>:8080`.
3. Bấm **"BẮT ĐẦU ĐO GIA TỐC"**: Cảm biến gia tốc điện thoại phát luồng dữ liệu 32 Hz qua WebSocket về máy tính.
4. Máy tính chuẩn hóa Z-score từ `checkpoints/scaler.pkl`, đưa qua `HARClassifier` (checkpoint tốt nhất) và phản hồi kết quả nhận diện (Ngồi, Đi bộ, Leo cầu thang, Đạp xe...) kèm xác suất hiển thị trực tiếp trên màn hình điện thoại trong thời gian thực.
5. Hỗ trợ thêm chế độ test mẫu thật từ PPG-DaLiA và nhận dữ liệu từ app **Sensor Logger**.

---

## 8. Danh mục Tài liệu Tham khảo (References)

Toàn bộ **11 tài liệu gốc (PDF bài báo quốc tế, mã nguồn MATLAB/Python, tài liệu kỹ thuật chuẩn)** đã được tải về và lưu trữ đầy đủ trong thư mục: 👉 **[`tai_lieu_tham_khao/`](tai_lieu_tham_khao/)** ([Xem mục lục chi tiết](tai_lieu_tham_khao/README.md)).  
Chi tiết về ngữ cảnh trích dẫn học thuật, bảng đối chiếu trang trong văn bản báo cáo và định dạng BibTeX được lưu trữ tại: 👉 **[REFERENCES.md](REFERENCES.md)**.

| Mã | Tài liệu tham khảo (IEEE Format) | Vị trí trích dẫn trong Báo cáo DOCX | Tệp Minh chứng Lưu trữ (`tai_lieu_tham_khao/`) |
| :---: | :--- | :--- | :--- |
| **[1]** | A. Reiss et al., *"PPG-DaLiA,"* UCI Machine Learning Repository, 2019. | Mục 2.1 (Trang 4) | [[01] Đặc tả PPG-DaLiA (PDF)](tai_lieu_tham_khao/[01]_UCI_PPG_DaLiA_Dataset_Documentation.pdf) |
| **[2]** | A. Reiss et al., *"Deep PPG: Large-Scale Heart Rate Estimation with CNNs,"* *Sensors*, 2019. | Mục 2.1 (Trang 4) | [[02] Bài báo Reiss Sensors (PDF)](tai_lieu_tham_khao/[02]_Reiss2019_Deep_PPG_Sensors.pdf) |
| **[3]** | P. H. Charlton, *"collate_ppg_dalia_dataset.m: MATLAB Data Collation Script,"* Univ. of Cambridge, 2026. | Mục 2.2 (Trang 4) | [[03] Mã MATLAB](tai_lieu_tham_khao/[03]_Charlton2026_collate_ppg_dalia_dataset.m) \| [[03] Python](tai_lieu_tham_khao/[03]_Charlton2026_convert_subject_pickle_files_to_mat.py) |
| **[4]** | D. P. Kingma & M. Welling, *"Auto-Encoding Variational Bayes,"* ICLR, 2013. | Mục 3.2 & Mục 3.4 (Trang 8, 9) | [[04] Bài báo Kingma VAE (PDF)](tai_lieu_tham_khao/[04]_Kingma2013_Auto_Encoding_Variational_Bayes_ICLR.pdf) |
| **[5]** | K. Sohn et al., *"Learning Structured Output Representation using Deep cVAEs,"* NeurIPS, 2015. | Mục 3.2 (Trang 8) | [[05] Bài báo Sohn cVAE (PDF)](tai_lieu_tham_khao/[05]_Sohn2015_Learning_Structured_Output_Representation_cVAE_NeurIPS.pdf) |
| **[6]** | C. Esteban et al., *"Real-valued (Medical) Time Series Generation with RCGANs,"* arXiv, 2017. | Mục 1.3 (Trang 2) | [[06] Bài báo Esteban RCGAN (PDF)](tai_lieu_tham_khao/[06]_Esteban2017_Medical_Time_Series_RCGAN.pdf) |
| **[7]** | A. Gretton et al., *"A Kernel Two-Sample Test,"* *JMLR*, vol. 13, 2012. | Mục 4.4 (Trang 14) | [[07] Bài báo Gretton MMD (PDF)](tai_lieu_tham_khao/[07]_Gretton2012_Kernel_Two_Sample_Test_MMD_JMLR.pdf) |
| **[8]** | PyTorch Foundation, *"ConvTranspose1d Module Documentation,"* PyTorch Docs v2.5, 2026. | Mục 3.5 (Trang 9) | [[08] Tài liệu ConvTranspose1d (PDF)](tai_lieu_tham_khao/[08]_PyTorch_ConvTranspose1d_Official_Documentation.pdf) |
| **[9]** | Scikit-learn Developers, *"Common pitfalls in data preprocessing (Data Leakage),"* v1.5, 2026. | Mục 2.5 (Trang 5) | [[09] Tài liệu Data Leakage (PDF)](tai_lieu_tham_khao/[09]_ScikitLearn_Data_Leakage_Preprocessing_Pitfalls.pdf) |
| **[10]** | SciPy Community, *"scipy.signal.welch: Estimation of PSD,"* SciPy Reference v1.14, 2026. | Mục 4.4 (Trang 14) | [[10] Tài liệu SciPy Welch (PDF)](tai_lieu_tham_khao/[10]_SciPy_Signal_Welch_PSD_Official_Documentation.pdf) |
| **[11]** | Scikit-learn Developers, *"r2_score: Coefficient of determination,"* v1.5, 2026. | Mục 5.3 (Trang 17) | [[11] Tài liệu R2 Score (PDF)](tai_lieu_tham_khao/[11]_ScikitLearn_R2_Score_Official_Documentation.pdf) |

---
**Tác giả đồ án:** Nguyễn Bách Tùng — Hà Nội, 2026.


