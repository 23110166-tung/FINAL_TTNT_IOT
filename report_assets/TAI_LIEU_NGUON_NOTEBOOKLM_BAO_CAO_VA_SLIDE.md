# TÀI LIỆU NGUỒN TỔNG HỢP TOÀN DIỆN CHO NOTEBOOKLM
## PHỤC VỤ SOẠN BÁO CÁO CUỐI KỲ (BƯỚC 2) VÀ SLIDE THUYẾT TRÌNH BẢO VỆ (BƯỚC 3)

---

### THÔNG TIN ĐỀ TÀI VÀ HỌC THUẬT
- **Tên Đề tài**: TỔNG HỢP DỮ LIỆU GIA TỐC BA TRỤC CÓ ĐIỀU KIỆN THEO HOẠT ĐỘNG BẰNG 1D-cVAE VÀ ĐÁNH GIÁ TSTR TRÊN PPG-DaLiA
- **Môn học**: Trí tuệ Nhân tạo cho IoT (AIoT) - Mã HP: AIOT331185_01CLC
- **Cơ sở đào tạo**: Trường Đại học Sư phạm Kỹ thuật TP. Hồ Chí Minh (HCMUTE) - Khoa Công nghệ Thông tin
- **Giảng viên hướng dẫn**: ThS. Hồ Nhựt Minh
- **Sinh viên thực hiện**: Nguyễn Bách Tùng - MSSV: 23110166
- **Thời gian thực hiện**: Tháng 09 – Tháng 10 năm 2026

---

## MỤC LỤC TÀI LIỆU NGUỒN
1. Tóm tắt Đề tài và Bối cảnh Nghiên cứu
2. Tập dữ liệu PPG-DaLiA và Quy trình Tiền xử lý Tensor Chống Rò rỉ Dữ liệu
3. Kiến trúc Mô hình 1D-cVAE và Cơ chế Sinh Dữ liệu
4. Khung Thực nghiệm Đối chứng 5 Nhánh Công bằng và Tiêu chuẩn TSTR
5. Bảng Kết quả Thực nghiệm Toàn diện và Chỉ số Khoa học
6. Phân tích Chuyên sâu và Diễn giải 8 Hình vẽ Khoa học (300 DPI)
7. Thảo luận Khoa học, Khám phá Bất ngờ và Giới hạn Thực tế
8. Kiến trúc Triển khai Thời gian thực Hệ thống IoT và Web Dashboard
9. Hướng dẫn Prompt Mẫu cho NotebookLM để Sinh Báo cáo (Bước 2) và Slide (Bước 3)

---

## 1. TÓM TẮT ĐỀ TÀI VÀ BỐI CẢNH NGHIÊN CỨU

### 1.1. Bối cảnh Kỹ thuật
Nhận dạng hoạt động người dùng (Human Activity Recognition – HAR) dựa trên cảm biến đo quán tính (IMU/ACC) là một trong những thành phần cốt lõi của các thiết bị đeo thông minh (wearable devices) và hệ sinh thái Internet vạn vật (IoT). Dữ liệu gia tốc cổ tay phản ánh chân thực các vận động thể chất và thói quen sinh hoạt. 

Tuy nhiên, việc thu thập và gán nhãn dữ liệu gia tốc trong môi trường sống tự nhiên (in-the-wild) đối mặt với hai rào cản kỹ thuật nghiêm trọng:
1. **Mất cân bằng dữ liệu cực đoan**: Các hoạt động tĩnh hoặc thường nhật (như Ngồi, Ăn trưa, Làm việc bàn giấy) chiếm phần lớn thời gian quan sát, trong khi các hoạt động vận động mạnh hoặc chuyển tiếp (như Leo cầu thang, Bi lắc, Đạp xe) chiếm tỷ lệ rất thấp.
2. **Chi phí gán nhãn thực địa đắt đỏ và tính xâm phạm đời tư**: Đòi hỏi thiết bị giám sát liên tục hoặc người dùng phải tự ghi chép nhật ký hoạt động.

### 1.2. Phát biểu Bài toán và Giải pháp Đề xuất
Đề tài nghiên cứu ứng dụng mô hình sinh học sâu: **Mạng Tự mã hóa Biến phân Có điều kiện Một chiều (1D-cVAE - Conditional Variational Autoencoder)** để tổng hợp tín hiệu gia tốc 3 trục (X, Y, Z) có điều kiện theo 8 nhãn hoạt động con người trên tập dữ liệu chuẩn **PPG-DaLiA**.

Mô hình học phân phối tiềm ẩn $q_\phi(z|x,c)$ và bộ giải mã $g_	heta(z,c)$, với $z \in \mathbb{R}^{32}$ và $c$ là nhãn điều kiện dạng one-hot 8 chiều. Sau khi huấn luyện, bộ giải mã nhận vector ngẫu nhiên $z \sim \mathcal{N}(0, I)$ cùng nhãn mong muốn $c$ để sinh ra các cửa sổ tín hiệu gia tốc nhân tạo hoàn toàn mới mà không cần dữ liệu mồi.

### 1.3. Tính Mới và Cam kết Học thuật Nghiêm ngặt (Theo Phản biện Đã duyệt)
- **Đánh giá theo Tiêu chuẩn TSTR (Train on Synthetic, Test on Real)**: Không đánh giá chất lượng mô hình sinh chỉ qua độ mất mát tái tạo hay trực quan hóa hình học mờ nhạt (như t-SNE), mà đo lường trực tiếp tính hữu ích của dữ liệu sinh thông qua hiệu năng huấn luyện bộ phân loại downstream khi kiểm thử trên người thật chưa từng thấy (unseen subjects S14, S15).
- **Tuyệt đối không rò rỉ dữ liệu (No Data Leakage)**:
  - Chia tập theo đối tượng người tham gia (Subject-wise holdout): Train gồm S1–S11 (11 người), Validation gồm S12–S13 (2 người), Test gồm S14–S15 (2 người).
  - Toàn bộ tham số chuẩn hóa (StandardScaler mean và std) được fit duy nhất trên tập Train S1–S11, sau đó áp dụng cố định cho Val, Test và dữ liệu sinh.
- **Báo cáo Trung thực và Minh bạch**: Báo cáo đầy đủ mọi kết quả dù kết quả tăng cường dữ liệu cVAE không vượt qua đối chứng truyền thống, phân tích nguyên nhân vật lý và giới hạn tín hiệu học thuật.

---

## 2. TẬP DỮ LIỆU PPG-DaLiA VÀ TIỀN XỬ LÝ TENSOR

### 2.1. Đặc tả Dữ liệu Cảm biến và Nhãn Hoạt động
Tập dữ liệu **PPG-DaLiA** được công bố bởi Reiss et al. (Sensors, 2019 / UCI Repository), thu thập từ 15 tình nguyện viên (S1–S15) trong cả điều kiện phòng thí nghiệm và sinh hoạt tự nhiên:
- **Tín hiệu sử dụng**: Cảm biến gia tốc 3 trục đeo ở cổ tay (`signal/wrist/ACC`), tần số lấy mẫu $f_s = 32	ext{ Hz}$. Đơn vị chuẩn là $g$ (gia tốc trọng trường, $1g pprox 9.81	ext{ m/s}^2$).
- **Đồng bộ nhãn**: Trường `activity` trong dữ liệu gốc có tần số $4	ext{ Hz}$. Sử dụng phép giữ bậc không (Zero-Order Hold - ZOH) để đồng bộ lên $32	ext{ Hz}$ theo công thức: mẫu gia tốc thứ $i$ nhận nhãn `activity[floor(i/8)]`.
- **8 Lớp Hoạt động Hợp lệ**:
  - Mã gốc 0 (`No activity / Transient`) bị loại bỏ hoàn toàn khỏi bài toán.
  - Các mã gốc từ 1 đến 8 được ánh xạ thành nhãn nội bộ $y \in \{0, 1, 2, 3, 4, 5, 6, 7\}$:
    - Lớp 0: Ngồi (Sitting)
    - Lớp 1: Leo cầu thang (Stairs)
    - Lớp 2: Chơi bi lắc (Table soccer)
    - Lớp 3: Đạp xe (Cycling)
    - Lớp 4: Lái xe (Driving)
    - Lớp 5: Ăn trưa (Lunch break)
    - Lớp 6: Đi bộ (Walking)
    - Lớp 7: Làm việc bàn giấy (Working)

### 2.2. Kỹ thuật Phân đoạn Cửa sổ và Bảo toàn Thành phần DC
- **Kích thước cửa sổ**: $L = 128$ mẫu (tương ứng chính xác 4.0 giây tín hiệu ở $32	ext{ Hz}$).
- **Bước nhảy cửa sổ**: $S = 64$ mẫu (chồng lấn 50% - $2.0$ giây).
- **Điều kiện hợp lệ của cửa sổ**: Cửa sổ chỉ được giữ lại nếu toàn bộ 128 mẫu đều mang duy nhất một nhãn hoạt động hợp lệ (từ 0 đến 7) và không chứa giá trị NaN/Inf. Mọi cửa sổ chứa nhãn 0 hoặc chuyển tiếp giữa hai hoạt động đều bị loại bỏ dứt khoát.
- **Bảo toàn thành phần tĩnh DC**: Đề tài quyết định **không** áp dụng bộ lọc thông cao ($0.5	ext{ Hz}$) và **không** trừ giá trị trung bình từng cửa sổ. Lý do học thuật: Việc giữ nguyên giá trị DC giúp mô hình học được góc nghiêng trọng trường của cổ tay, vốn là thông tin sống còn để phân biệt các tư thế tĩnh (như Ngồi vs Lái xe vs Làm việc).

### 2.3. Thống kê Cửa sổ sau Tiền xử lý
- **Tập Huấn luyện (Train - S1 đến S11)**: $34,585$ cửa sổ hợp lệ.
- **Tập Phát triển (Validation - S12, S13)**: $6,302$ cửa sổ hợp lệ.
- **Tập Kiểm định (Test - S14, S15)**: $6,110$ cửa sổ hợp lệ (S14: $3,187$ cửa sổ, S15: $2,923$ cửa sổ).
- **Tổng cộng toàn bộ đề tài**: $46,997$ cửa sổ gia tốc chuẩn hóa.

---

## 3. KIẾN TRÚC MÔ HÌNH 1D-cVAE VÀ CƠ CHẾ SINH

### 3.1. Kiến trúc Bộ mã hóa (Encoder) và Giải mã (Decoder)
- **Kích thước Tensor đầu vào**: $x \in \mathbb{R}^{B 	imes 3 	imes 128}$ (Batch, 3 kênh gia tốc, 128 điểm thời gian).
- **Vectơ Điều kiện (Condition)**: Nhãn $c \in \{0..7\}$ được biểu diễn bằng One-Hot 8 chiều, mở rộng theo chiều thời gian thành $\mathbb{R}^{B 	imes 8 	imes 128}$.
- **Bộ mã hóa $q_\phi(z|x,c)$**:
  - Ghép đầu vào dọc theo trục kênh: $[x; c] \in \mathbb{R}^{B 	imes 11 	imes 128}$.
  - Gồm 4 khối Conv1D tuần tự với stride=2 để giảm chiều:
    - Conv1D(11 $	o$ 32, k=5, s=2, p=2) + BatchNorm1D + LeakyReLU(0.2) $	o$ Chiều dài 64
    - Conv1D(32 $	o$ 64, k=5, s=2, p=2) + BatchNorm1D + LeakyReLU(0.2) $	o$ Chiều dài 32
    - Conv1D(64 $	o$ 128, k=5, s=2, p=2) + BatchNorm1D + LeakyReLU(0.2) $	o$ Chiều dài 16
    - Conv1D(128 $	o$ 128, k=5, s=2, p=2) + BatchNorm1D + LeakyReLU(0.2) $	o$ Chiều dài 8
  - Làm phẳng (Flatten: $128 	imes 8 = 1024$) $	o$ 2 nhánh tuyến tính độc lập Linear(1024, 32) để tính vector trung bình $\mu$ và log-phương sai $\log(\sigma^2)$.
- **Thủ thuật Tái tham số hóa (Reparameterization Trick)**:
  $$z = \mu + \exp(0.5 \cdot \log(\sigma^2)) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_{32})$$
- **Bộ giải mã $g_	heta(z,c)$**:
  - Ghép vector tiềm ẩn $z$ (32 chiều) với vector nhãn one-hot $c$ (8 chiều) $	o 40$ chiều.
  - Linear(40 $	o$ 1024) $	o$ Reshape thành tensor $(B, 128, 8)$.
  - Gồm 4 khối ConvTranspose1D ngược để phục hồi kích thước:
    - ConvTranspose1D(128 $	o$ 128, k=5, s=2, p=2, `output_padding=1`) + BatchNorm1D + LeakyReLU(0.2) $	o$ Chiều dài 16
    - ConvTranspose1D(128 $	o$ 64, k=5, s=2, p=2, `output_padding=1`) + BatchNorm1D + LeakyReLU(0.2) $	o$ Chiều dài 32
    - ConvTranspose1D(64 $	o$ 32, k=5, s=2, p=2, `output_padding=1`) + BatchNorm1D + LeakyReLU(0.2) $	o$ Chiều dài 64
    - ConvTranspose1D(32 $	o$ 3, k=5, s=2, p=2, `output_padding=1`) + Tanh $	o$ Chiều dài 128
  - Tensor đầu ra: $\hat{x} \in \mathbb{R}^{B 	imes 3 	imes 128}$ (chuẩn xác 100% kích thước ban đầu nhờ `output_padding=1`).

### 3.2. Hàm Mất mát và Kỹ thuật Warm-up Trọng số KL ($eta$-Annealing)
Hàm mất mát tổng thể $\mathcal{L}_t$ tại epoch $t$ được chia trung bình trên tổng số phần tử của cửa sổ ($D = 3 	imes 128 = 384$):
$$\mathcal{L}_t = rac{\mathcal{L}_{	ext{rec}}}{384} + eta(t) \cdot rac{D_{	ext{KL}}}{384}$$
Trong đó:
- $\mathcal{L}_{	ext{rec}} = \sum_{b=1}^B \sum_{i=1}^{384} (x_{b,i} - \hat{x}_{b,i})^2$ (Sai số toàn phương trung bình).
- $D_{	ext{KL}} = -rac{1}{2} \sum_{j=1}^{32} (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)$ (Khoảng cách Kullback-Leibler).
- Lịch điều chỉnh trọng số $eta(t)$ tuyến tính trong 25 epoch đầu tiên:
  $$eta(t) = \min\left(1.0, rac{t}{25}ight), \quad t \in \{0, \dots, 25\}$$
  Kỹ thuật này triệt tiêu hiện tượng sụp đổ không gian tiềm ẩn (posterior collapse), cho phép bộ giải mã học cách tái cấu trúc hình dạng sóng trước khi không gian ẩn bị ép chặt về phân phối chuẩn chuẩn tắc $\mathcal{N}(0, I)$.

---

## 4. KHUNG THỰC NGHIỆM ĐỐI CHỨNG 5 NHÁNH CÔNG BẰNG VÀ TIÊU CHUẨN TSTR

### 4.1. Kiến trúc Bộ Phân loại HAR Chuẩn hóa
Nhằm đảm bảo tính khách quan tuyệt đối, cả 5 nhánh thực nghiệm đều sử dụng chung một kiến trúc mạng nơ-ron tích chập 1D cố định:
- 3 Khối Conv1D liên tiếp:
  - Khối 1: Conv1D(3 $	o$ 32, k=5, p=2) + BatchNorm1D + ReLU + MaxPool1d(2) $	o$ L=64
  - Khối 2: Conv1D(32 $	o$ 64, k=5, p=2) + BatchNorm1D + ReLU + MaxPool1d(2) $	o$ L=32
  - Khối 3: Conv1D(64 $	o$ 128, k=5, p=2) + BatchNorm1D + ReLU + MaxPool1d(2) $	o$ L=16
- Lớp gom trung bình toàn cục: GlobalAveragePooling1D $	o$ vector 128 chiều.
- Lớp phân loại: Linear(128 $	o$ 8) + Softmax (huấn luyện bằng Cross-Entropy Loss không trọng số).

### 4.2. Hạn ngạch Cân bằng Công bằng ($n^*$)
Để tránh thiên vị giữa các kỹ thuật cân bằng lớp, một hạn ngạch chung được xác lập dựa trên trung vị số mẫu của 8 lớp trong tập Train:
$$n^* = \lceil	ext{median}(n_c)ceil = \lceil	ext{median}([3341, 2295, 1671, 2518, 5062, 10204, 3422, 6072])ceil = 3,382	ext{ mẫu}$$
Đối với bất kỳ lớp thiểu số $c$ nào có $n_c < n^*$, số lượng mẫu bổ sung là $\Delta n_c = \max(0, n^* - n_c)$. Các lớp đa số giữ nguyên số lượng.

### 4.3. Định nghĩa 5 Nhánh Thực nghiệm
1. **Nhánh 1 - TRTR (Train on Real, Test on Real)**: Huấn luyện hoàn toàn trên dữ liệu thật gốc của tập Train S1–S11 ($34,585$ cửa sổ). Đây là mốc tham chiếu thực nghiệm chuẩn.
2. **Nhánh 2 - Real + ROS (Random Oversampling)**: Bổ sung mẫu lặp ngẫu nhiên từ chính các lớp thiểu số của tập Train cho đến khi đạt $n^* = 3382$ mẫu/lớp.
3. **Nhánh 3 - Real + TradAug (Traditional Augmentation)**: Áp dụng các phép biến đổi hình học phổ biến trong chuỗi thời gian gia tốc: Nhiễu Gauss ngẫu nhiên ($\sigma = 0.01$) kết hợp Co giãn biên độ (Scaling $0.95 \sim 1.05$) để sinh thêm $\Delta n_c$ mẫu cho các lớp thiểu số.
4. **Nhánh 4 - TSTR (Train on Synthetic, Test on Real)**: Bộ phân loại chỉ được học duy nhất trên $34,585$ cửa sổ nhân tạo do 1D-cVAE sinh ra từ prior $\mathcal{N}(0, I)$ (tỷ lệ mẫu các lớp đúng bằng tập Train thật). Hoàn toàn không tiếp xúc với bất kỳ cửa sổ dữ liệu thật nào trong quá trình cập nhật trọng số.
5. **Nhánh 5 - Real + cVAE (Generative Augmentation)**: Giữ nguyên dữ liệu thật của tập Train và bổ sung thêm $\Delta n_c$ mẫu nhân tạo chất lượng cao do 1D-cVAE sinh ra cho các lớp thiểu số.

---

## 5. BẢNG KẾT QUẢ THỰC NGHIỆM TOÀN DIỆN VÀ CHỈ SỐ KHOA HỌC

### BẢNG 1: THỐNG KÊ PHÂN BỐ MẪU VÀ HẠN NGẠCH BỔ SUNG CÂN BẰNG
| Chỉ số Lớp | Tên Hoạt động (Tiếng Việt) | Tên Tiếng Anh | Train (S1–S11) | Val (S12–S13) | Test (S14–S15) | Tổng Mẫu | Hạn ngạch Bổ sung ($\Delta n_c$) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | Ngồi | Sitting | 3,341 | 599 | 607 | 4,547 | +41 |
| 1 | Leo cầu thang | Stairs | 2,295 | 486 | 434 | 3,215 | +1,087 |
| 2 | Chơi bi lắc | Table soccer | 1,671 | 281 | 336 | 2,288 | +1,711 |
| 3 | Đạp xe | Cycling | 2,518 | 485 | 451 | 3,454 | +864 |
| 4 | Lái xe | Driving | 5,062 | 890 | 866 | 6,818 | 0 |
| 5 | Ăn trưa | Lunch break | 10,204 | 1,740 | 1,582 | 13,526 | 0 |
| 6 | Đi bộ | Walking | 3,422 | 616 | 633 | 4,671 | 0 |
| 7 | Làm việc bàn giấy | Working | 6,072 | 1,205 | 1,201 | 8,478 | 0 |
| **Tổng cộng** | - | - | **34,585** | **6,302** | **6,110** | **46,997** | **+3,703** |

*Ghi chú*: Ngưỡng trung vị $n^* = 3,382$ mẫu. Bốn lớp thiểu số (0, 1, 2, 3) được bổ sung tổng cộng 3,703 mẫu trong cả 3 nhánh tăng cường (ROS, TradAug, cVAE).

---

### BẢNG 2: HIỆU NĂNG NHẬN DẠNG HAR TRÊN TẬP KIỂM ĐỊNH TEST (S14, S15)
| Nhánh Thực nghiệm | Macro-F1 (S14) | Macro-F1 (S15) | Macro-F1 Trung bình | Độ chính xác (Accuracy) | Số bước cập nhật | Mức giảm TSTR ($\Delta	ext{TSTR}$) | Mức tăng Augmentation ($G_{	ext{aug}}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. TRTR (Real Baseline)** | 0.6687 (66.87%) | 0.7171 (71.71%) | **0.6929 (69.29%)** | **68.38%** | 15,148 | Mốc chuẩn | Mốc chuẩn |
| **2. Real + ROS** | 0.6786 (67.86%) | 0.7363 (73.63%) | **0.7074 (70.74%)** | **69.39%** | 28,752 | - | **+0.0145 (+1.45%)** |
| **3. Real + TradAug** | 0.6387 (63.87%) | 0.6971 (69.71%) | **0.6679 (66.79%)** | **63.62%** | 16,772 | - | **-0.0250 (-2.50%)** |
| **4. TSTR (Pure Synthetic)** | 0.3708 (37.08%) | 0.4328 (43.28%) | **0.4018 (40.18%)** | **39.28%** | 5,410 | **+0.2911 (+29.11%)** | -0.2911 (-29.11%) |
| **5. Real + cVAE** | 0.6122 (61.22%) | 0.7120 (71.20%) | **0.6621 (66.21%)** | **63.90%** | 10,782 | - | **-0.0309 (-3.09%)** |

*Công thức định nghĩa theo Đề cương*:
- $	ext{Macro-F1}_{	ext{Mean}} = rac{1}{2} (	ext{Macro-F1}_{S14} + 	ext{Macro-F1}_{S15})$
- $\Delta	ext{TSTR} = 	ext{Macro-F1}_{	ext{TRTR}} - 	ext{Macro-F1}_{	ext{TSTR}} = 0.6929 - 0.4018 = +0.2911$ (+29.11 điểm phần trăm)
- $G_{	ext{aug}} = 	ext{Macro-F1}_{	ext{Branch}} - 	ext{Macro-F1}_{	ext{TRTR}}$

---

### BẢNG 3: CHỈ SỐ PHÂN PHỐI BIASED MMD² (108 CHIỀU) VÀ PHỔ WELCH THEO LỚP
| Mã | Tên Hoạt động | $	ext{MMD}^2(	ext{Test, Syn})$ | $	ext{MMD}^2(	ext{Test, Train})$ | Khoảng cách Phổ $d_{	ext{PSD}}$ | Hệ số Xác định $R^2_{	ext{PSD}}$ | Tổng Công suất Syn | Tổng Công suất Test |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Ngồi (Sitting) | 0.4329 | 0.0620 | 0.0261 | 0.9996 | 0.6054 | 0.5303 |
| 1 | Leo cầu thang (Stairs) | 0.2860 | 0.0272 | 0.2531 | 0.9704 | 1.0389 | 1.4051 |
| 2 | Chơi bi lắc (Table soccer) | 0.2688 | 0.0190 | 0.2862 | 0.9659 | 0.8464 | 1.0908 |
| 3 | Đạp xe (Cycling) | 0.6647 | 0.0257 | 0.7831 | 0.0945 | 0.6152 | 1.5773 |
| 4 | Lái xe (Driving) | 0.3593 | 0.0245 | 0.1042 | 0.9961 | 0.6330 | 0.7502 |
| 5 | Ăn trưa (Lunch break) | 0.1209 | 0.0144 | 0.0411 | 0.9994 | 0.6079 | 0.7804 |
| 6 | Đi bộ (Walking) | 0.2734 | 0.0235 | 0.1764 | 0.9876 | 1.0264 | 1.5535 |
| 7 | Làm việc (Working) | 0.1689 | 0.0195 | 0.0273 | 0.9997 | 0.6592 | 0.6674 |
| **TB**| **Toàn bộ 8 lớp** | **0.3218** | **0.0270** | **0.2122** | **0.8767** | **0.7541** | **1.0444** |

*Ghi chú*:
- Không gian đặc trưng 108 chiều gồm: Mean, Std, RMS của 3 trục (9 chiều) ghép với $\log(	ext{PSD} + 10^{-8})$ của 3 trục (99 chiều).
- Băng thông cơ sở $\sigma_0 = 11.1942$ (tính từ trung vị khoảng cách Euclidean dương trên $2,000$ mẫu train chuẩn hóa).
- $d_{	ext{PSD}}$ là khoảng cách $L_1$ giữa hai mật độ phổ công suất trung bình đã chuẩn hóa về diện tích bằng 1.

---

### BẢNG 4: THÔNG SỐ KỸ THUẬT VÀ TRIỂN KHAI THỜI GIAN THỰC IOT
| Thành phần Hệ thống | Thông số Giá trị | Đơn vị / Ghi chú |
|:---|:---:|:---|
| Số tham số mô hình 1D-cVAE | 702,467 | Tham số (Parameters) |
| Kích thước Checkpoint cVAE (`.pth`) | ~2.81 | MB (MegaBytes) |
| Số tham số bộ phân loại HAR | 97,416 | Siêu nhẹ, phù hợp Edge Device |
| Kích thước Checkpoint HAR (`.pth`) | ~395 | KB (KiloBytes) |
| Tần số lấy mẫu cảm biến ACC | 32.0 | Hz (Chu kỳ 31.25 ms/mẫu) |
| Độ dài cửa sổ xử lý ($L$) | 128 | Mẫu (4.0 giây tín hiệu thực) |
| Bước trượt phân loại ($S$) | 32 | Mẫu (Cập nhật kết quả mỗi 1.0 giây) |
| Độ trễ suy luận nơ-ron trên Laptop | ~8.2 | mili-giây (ms) / cửa sổ |
| Tốc độ khung hình xử lý tối đa | > 120 | FPS (Dư thừa năng lực thời gian thực) |
| Giao thức kết nối cảm biến | HTTP POST / WebSocket | Nhận dữ liệu qua Wi-Fi nội bộ |
| Cổng dịch vụ Web Server | 8088 | Tích hợp Web Dashboard trực quan |

---

## 6. PHÂN TÍCH CHUYÊN SÂU 8 HÌNH VẼ KHOA HỌC (300 DPI)

Tất cả 8 hình ảnh dưới đây đã được lưu trữ hoàn chỉnh tại thư mục: `report_assets/figures/` (và bản sao dự phòng tại `results/figures/`).

### HÌNH 1: TIẾN TRÌNH HUẤN LUYỆN MÔ HÌNH 1D-cVAE (40 EPOCH)
- **Tên tệp**: `fig1_cvae_training_curves.png`
- **Mô tả 4 đồ thị con**:
  - *(a) Total Loss*: Đường mất mát tập huấn luyện giảm mượt mà từ 0.28 xuống 0.09. Đường validation loss đạt điểm tối ưu tại **Epoch 24** với giá trị cực tiểu $\mathcal{L}_{	ext{val}} = 0.10072$.
  - *(b) Reconstruction Loss*: Sai số tái tạo MSE giảm nhanh chóng trong 10 epoch đầu, chứng minh bộ giải mã nắm bắt rất nhanh cấu trúc hình học của tín hiệu gia tốc 3 trục.
  - *(c) KL Divergence*: Tăng dần từ 0 lên khoảng 12 nats trong 25 epoch đầu do kỹ thuật warm-up $eta(t)$, sau đó ổn định vững chắc, không hề xuất hiện hiện tượng sụp đổ không gian ẩn (KL $	o 0$).
  - *(d) Tốc độ học (Learning Rate)*: Giảm mượt mà theo lịch Cosine Annealing từ $10^{-3}$ xuống $10^{-5}$.
- **Ý nghĩa Báo cáo & Slide**: Minh chứng quy trình huấn luyện hội tụ hoàn hảo, chọn đúng checkpoint tại epoch 24 bằng validation loss với $eta=1.0$ cố định, hoàn toàn không chạm vào tập test.

### HÌNH 2: MA TRẬN NHẦM LẪN (CONFUSION MATRICES) CỦA 5 NHÁNH TRÊN TẬP TEST (S14, S15)
- **Tên tệp**: `fig2_confusion_matrices_5_branches.png`
- **Quan sát then chốt**:
  - Các hoạt động có chu kỳ vận động rõ nét như **Đi bộ (Walking)** và **Đạp xe (Cycling)** đạt độ chính xác rất cao (> 85%) trên các nhánh TRTR và ROS.
  - **Leo cầu thang (Stairs)** thường bị nhầm lẫn sang Đi bộ do cả hai đều chứa xung bước chân chu kỳ ở dải tần $1.5 - 2.5	ext{ Hz}$.
  - Cụm hoạt động tĩnh gồm **Ngồi (Sitting)**, **Lái xe (Driving)**, **Ăn trưa (Lunch break)** và **Làm việc (Working)** có mức độ nhầm lẫn chéo tương đối cao. Nguyên nhân: Cổ tay người dùng khi làm việc bàn giấy hoặc lái xe đều có tư thế úp ngang, góc nghiêng trọng trường tĩnh rất gần với tư thế ngồi nghỉ.
  - Ma trận của nhánh **TSTR**: Nhận diện chuẩn xác các hoạt động tuần hoàn (Đi bộ, Cầu thang), nhưng gặp khó khăn ở các lớp tĩnh có độ lệch biên độ nhỏ.

### HÌNH 3: SO SÁNH HIỆU NĂNG HAR TOÀN DIỆN VÀ CHỈ SỐ TĂNG TRƯỞNG
- **Tên tệp**: `fig3_har_performance_comparison.png`
- **Điểm nhấn số liệu**:
  - Nhánh **Real + ROS** xuất sắc giành vị trí số 1 với Macro-F1 = 70.74% ($G_{	ext{aug}} = +1.45\%$), chứng minh giải pháp lấy mẫu lặp ngẫu nhiên là đối chứng cực kỳ mạnh mẽ.
  - Nhánh **TSTR** đạt Macro-F1 = 40.18% (so với mức ngẫu nhiên $12.5\%$), mang lại chỉ số $\Delta	ext{TSTR} = +29.11\%$. Đây là minh chứng vàng khẳng định dữ liệu sinh từ 1D-cVAE chứa đựng thông tin ngữ nghĩa hoạt động thực thụ.
  - Nhánh **Real + cVAE** (66.21%) và **Real + TradAug** (66.79%) giảm nhẹ so với TRTR (69.29%).

### HÌNH 4: PHÂN PHỐI KHOẢNG CÁCH BIASED MMD² VÀ ĐỘ LỆCH PHỔ WELCH THEO 8 LỚP
- **Tên tệp**: `fig4_distribution_mmd_psd_by_class.png`
- **Quan sát then chốt**:
  - MMD²(Test, Syn) đạt giá trị tốt nhất (thấp nhất) ở các lớp tĩnh: **Ăn trưa** (0.1209) và **Làm việc** (0.1689).
  - Lớp **Đạp xe (Cycling)** có MMD² cao nhất (0.6647) và khoảng cách phổ lớn nhất ($d_{	ext{PSD}} = 0.7831$). Điều này giải thích sâu sắc tại sao cVAE khó mô phỏng chính xác chuyển động đạp xe ở cổ tay (vốn phụ thuộc mạnh vào tư thế cầm ghi-đông của từng đối tượng thử nghiệm).

### HÌNH 5: ĐƯỜNG CONG MẬT ĐỘ PHỔ CÔNG SUẤT (WELCH PSD) 0 - 16 HZ
- **Tên tệp**: `fig5_welch_psd_spectral_curves.png`
- **Quan sát then chốt**:
  - Đối chiếu trực tiếp phổ công suất giữa Tín hiệu Thật (Đường nét liền) và Tín hiệu Sinh cVAE (Đường nét đứt).
  - Ở lớp **Đi bộ**, cVAE tái tạo chuẩn xác đỉnh hài bậc nhất (fundamental harmonic peak) tại tần số bước chân $1.8	ext{ Hz}$.
  - Ở dải tần cao ($> 6	ext{ Hz}$), tín hiệu sinh có xu hướng giảm công suất nhanh hơn tín hiệu thật, phản ánh tính chất làm mịn (smoothing effect) kinh điển của họ mô hình VAE.

### HÌNH 6: DẠNG SÓNG MIỀN THỜI GIAN 3 TRỤC (REAL VS cVAE SYNTHETIC)
- **Tên tệp**: `fig6_time_domain_waveform_comparison.png`
- **Quan sát then chốt**:
  - Trực quan hóa 128 mẫu (4 giây) của 3 trục X (đỏ), Y (xanh lá), Z (xanh dương).
  - Tín hiệu sinh của hoạt động **Đi bộ** duy trì được tính chu kỳ rõ rệt và pha dao động giữa các trục tương tự dữ liệu thật.
  - Tín hiệu sinh của hoạt động **Ngồi** duy trì xuất sắc các giá trị offset trọng trường DC tĩnh mà không bị trôi về 0, chứng minh quyết định không lọc bỏ DC là hoàn toàn chuẩn xác.

### HÌNH 7: PHÂN BỐ MẪU HUẤN LUYỆN VÀ HẠN NGẠCH BỔ SUNG CÂN BẰNG
- **Tên tệp**: `fig7_class_distribution_and_quotas.png`
- **Quan sát then chốt**:
  - Thể hiện trực quan mức độ mất cân bằng nghiêm trọng của tập dữ liệu gốc (Lớp Ăn trưa có 10,204 mẫu trong khi Lớp Chơi bi lắc chỉ có 1,671 mẫu - chênh lệch gấp hơn 6 lần).
  - Đường kẻ đỏ biểu diễn hạn ngạch trung vị $n^* = 3382$ mẫu, làm rõ chính xác phần mẫu thật và phần mẫu bổ sung công bằng cho từng lớp.

### HÌNH 8: TỔNG QUAN KIẾN TRÚC HỆ THỐNG VÀ QUY TRÌNH THỰC NGHIỆM TSTR
- **Tên tệp**: `fig8_system_architecture_and_pipeline.png`
- **Quan sát then chốt**:
  - Sơ đồ tổng thể toàn bộ đồ án, kết nối từ: (1) Tiền xử lý dữ liệu PPG-DaLiA $	o$ (2) Kiến trúc nơ-ron 1D-cVAE $	o$ (3) Cơ chế sinh và hạn ngạch cân bằng $	o$ (4) Khung 5 nhánh thực nghiệm đối chứng $	o$ (5) Đánh giá kiểm định không rò rỉ và triển khai Edge IoT thời gian thực.
  - Đây là hình ảnh trung tâm hoàn hảo để đưa vào Trang Bìa/Chương 3 của Báo cáo và làm Slide Tổng quan (Overview Slide) trong bài thuyết trình.

---

## 7. THẢO LUẬN KHOA HỌC, KHÁM PHÁ BẤT NGỜ VÀ GIỚI HẠN

### 7.1. Vì sao Random Oversampling (ROS) lại chiến thắng cVAE và Biến đổi Truyền thống?
Trong thực nghiệm, nhánh Real + ROS đạt Macro-F1 = 70.74% ($+1.45\%$ so với TRTR), vượt qua cả Real + cVAE ($66.21\%$) và Real + TradAug ($66.79\%$).
- **Lý giải Khoa học**: Dữ liệu cảm biến gia tốc thu thập trong điều kiện tự nhiên có phương sai giữa các cá nhân (inter-subject variability) rất lớn. Việc nhân bản chính xác các cửa sổ thật (ROS) giúp mạng phân loại củng cố các biên quyết định sắc nét của các mẫu hiếm có thật mà không làm biến dạng phân phối cục bộ.
- Ngược lại, phép tăng cường truyền thống (Jitter/Scaling) làm xê dịch pha của bước chân hoặc làm thay đổi độ nghiêng DC tĩnh, vô tình gây nhiễu ngữ nghĩa (semantic drift). Trong khi đó, cVAE có xu hướng sinh ra các mẫu mượt mà ở trung tâm phân phối, làm giảm độ sắc nét của các vận động xung lực mạnh.

### 7.2. Ý nghĩa Thực thụ của Kết quả TSTR (40.18%)
Một bộ phân loại được đào tạo **hoàn toàn 100% bằng dữ liệu nhân tạo do máy tính sinh ra**, khi đem kiểm tra trên cơ thể của những con người mới tinh (S14, S15), đã đạt độ chính xác xấp xỉ $40\%$ (gấp hơn 3.2 lần xác suất đoán ngẫu nhiên $12.5\%$).
- Điều này chứng minh 1D-cVAE đã học được cấu trúc phân phối có điều kiện của chuyển động con người, chứ không chỉ học thuộc lòng (memorization) dữ liệu huấn luyện.
- Độ chênh lệch $\Delta	ext{TSTR} = +29.11\%$ phản ánh khoảng cách biểu diễn (representation gap) giữa mô hình sinh tham số hóa và thực tế phong phú của cơ thể người.

### 7.3. Khả năng Triển khai Hệ thống IoT Biên (Edge Computing)
- Với chỉ **97,416 tham số** và kích thước tệp trọng số **~395 KB**, mô hình bộ phân loại HAR hoàn toàn có thể nạp thẳng vào bộ nhớ Flash/RAM của các vi điều khiển tầm trung (như ESP32-S3, STM32F4/H7 hoặc nRF5340) hoặc chạy trực tiếp trên smartphone/smartwatch.
- Độ trễ suy luận chỉ **~8.2 ms** trên một cửa sổ 4 giây, chứng minh hệ thống hoàn toàn dư thừa năng lực để nhận diện hoạt động thời gian thực ở biên mạng mà không cần truyền dữ liệu thô về đám mây, bảo vệ tuyệt đối quyền riêng tư của người dùng.

---

## 8. HƯỚNG DẪN PROMPT MẪU CHO NOTEBOOKLM

Để tận dụng tối đa sức mạnh của NotebookLM nhằm tạo ra **Báo cáo Cuối kỳ (Bước 2)** và **Bộ Slide Thuyết trình (Bước 3)**, hãy làm theo các bước sau:

### BƯỚC A: TẢI TÀI NGUYÊN VÀO NOTEBOOKLM
1. Mở trang web [NotebookLM](https://notebooklm.google.com/).
2. Tạo một sổ tay mới (New Notebook) đặt tên là: `AIoT_CuoiKy_1D-cVAE_PPG-DaLiA_NguyenBachTung`.
3. Tải lên 2 tệp nguồn:
   - Tệp 1: `NguyenBachTung_DeCuong_DaChinhSua.docx` (Tệp đề cương chi tiết đã duyệt của bạn).
   - Tệp 2: `TAI_LIEU_NGUON_NOTEBOOKLM_BAO_CAO_VA_SLIDE.docx` (Tệp tài liệu nguồn vừa được tạo này).

---

### BƯỚC B: PROMPT SINH BÁO CÁO CUỐI KỲ CHI TIẾT (BƯỚC 2)
Copy và dán toàn bộ đoạn văn bản dưới đây vào khung chat của NotebookLM:

```text
Bạn là một chuyên gia nghiên cứu cao cấp về Trí tuệ Nhân tạo cho IoT (AIoT) và Xử lý Tín hiệu Cảm biến Đeo. Dựa trên toàn bộ các tài liệu nguồn đã cung cấp (gồm Đề cương đã duyệt và Tài liệu nguồn kết quả thực nghiệm chi tiết), hãy viết cho tôi BÁO CÁO TIỂU LUẬN CUỐI KỲ TOÀN DIỆN (khoảng 20 đến 25 trang A4, trình bày học thuật nghiêm ngặt) với đầy đủ 6 chương theo đúng cấu trúc chuẩn:

Trang bìa & Thông tin chung:
- Tên Đề tài: TỔNG HỢP DỮ LIỆU GIA TỐC BA TRỤC CÓ ĐIỀU KIỆN THEO HOẠT ĐỘNG BẰNG 1D-cVAE VÀ ĐÁNH GIÁ TSTR TRÊN PPG-DaLiA
- Sinh viên: Nguyễn Bách Tùng - MSSV: 23110166 - Lớp: AIOT331185_01CLC
- GVHD: ThS. Hồ Nhựt Minh - Trường ĐH Sư phạm Kỹ thuật TP.HCM (HCMUTE)

Nội dung chi tiết 6 Chương bắt buộc:
- CHƯƠNG 1: ĐẶT VẤN ĐỀ, MỤC TIÊU VÀ ĐÓNG GÓP HỌC THUẬT
  + Bối cảnh mất cân bằng dữ liệu HAR trong IoT và chi phí gán nhãn thực địa.
  + Phát biểu bài toán 1D-cVAE tổng hợp gia tốc 3 trục theo 8 hoạt động.
  + 2 Câu hỏi thực nghiệm và các cam kết chống rò rỉ dữ liệu theo phản biện đã duyệt.
- CHƯƠNG 2: TẬP DỮ LIỆU PPG-DaLiA VÀ QUY TRÌNH TIỀN XỬ LÝ CHỐNG RÒ RỈ
  + Phân tích trường dữ liệu wrist ACC (32 Hz), đồng bộ nhãn ZOH từ 4 Hz lên 32 Hz, loại nhãn 0.
  + Kỹ thuật cửa sổ 128 mẫu (4s), bước 64 (chồng lấn 50%). Lý giải bảo toàn thành phần DC.
  + Chiến lược chia tập theo người (Train S1-S11, Val S12-S13, Test S14-S15).
  + Bảng thống kê mẫu chi tiết (Bảng 1 trong tài liệu nguồn).
- CHƯƠNG 3: KIẾN TRÚC MÔ HÌNH 1D-cVAE VÀ BỘ PHÂN LOẠI HAR
  + Chi tiết các tầng tích chập của Encoder, cơ chế Re-parameterization trick, Latent space 32 chiều.
  + Chi tiết Decoder với 4 khối ConvTranspose1D và vai trò bắt buộc của output_padding=1.
  + Hàm mất mát có trọng số β(t) anneal qua 25 epoch.
  + Kiến trúc bộ phân loại HAR chuẩn hóa (3 khối Conv1D).
  + Trích dẫn và phân tích Sơ đồ hệ thống Hình 8.
- CHƯƠNG 4: THIẾT KẾ THỰC NGHIỆM VÀ KHUNG ĐỐI CHỨNG 5 NHÁNH
  + Hạn ngạch cân bằng công bằng n* = 3382 mẫu.
  + Định nghĩa 5 nhánh thực nghiệm: TRTR, Real+ROS, Real+TradAug, TSTR, Real+cVAE.
  + Bộ chỉ số đánh giá: Macro-F1 bình đẳng theo người (S14, S15), Biased MMD² (108 chiều), Welch PSD.
- CHƯƠNG 5: KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH CHUYÊN SÂU
  + Báo cáo toàn bộ số liệu Bảng 2 (Hiệu năng 5 nhánh) và Bảng 3 (Phân phối MMD², PSD).
  + Trích dẫn, mô tả và phân tích chuyên sâu toàn bộ 8 Hình vẽ (Hình 1 đến Hình 8): Tiến trình huấn luyện, Ma trận nhầm lẫn 5 nhánh, So sánh Macro-F1, Phân phối MMD², Phổ Welch, Dạng sóng thời gian, Phân bố hạn ngạch, Sơ đồ hệ thống.
  + Thảo luận học thuật: Vì sao ROS đạt hiệu quả cao nhất (+1.45%)? Ý nghĩa thực thụ của TSTR = 40.18% (Delta TSTR = +29.11%)? Hiện tượng nhầm lẫn giữa các hoạt động tĩnh do tương đồng góc nghiêng trọng trường.
- CHƯƠNG 6: TRIỂN KHAI THỜI GIAN THỰC IOT, KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
  + Kiến trúc hệ thống IoT thời gian thực: Cảm biến điện thoại Sensor Logger, Web Server Python, Dashboard cổng 8088.
  + Đánh giá tài nguyên tính toán: 97K tham số, 395 KB bộ nhớ, độ trễ 8.2 ms, tính khả thi trên vi điều khiển biên.
  + Kết luận tổng kết và định hướng phát triển trong tương lai.

Yêu cầu định dạng: Viết mạch lạc, văn phong khoa học chuẩn mực, sử dụng bảng biểu và công thức toán học đầy đủ, trích dẫn rõ ràng Hình 1 đến Hình 8 trong các đoạn phân tích.
```

---

### BƯỚC C: PROMPT SOẠN BỘ SLIDE THUYẾT TRÌNH BẢO VỆ (BƯỚC 3)
Copy và dán toàn bộ đoạn văn bản dưới đây vào khung chat của NotebookLM:

```text
Dựa trên tài liệu nguồn, hãy soạn cho tôi KỊCH BẢN VÀ DÀN Ý CHI TIẾT BỘ SLIDE THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN CUỐI KỲ (từ 16 đến 20 slide, thời lượng trình bày 15 phút).

Với mỗi slide, hãy trình bày rõ ràng 4 mục:
1. TIÊU ĐỀ SLIDE
2. BỐ CỤC & NỘI DUNG CHÍNH (Gạch đầu dòng súc tích, chuyên nghiệp)
3. HÌNH ẢNH / BẢNG BIỂU MINH HỌA CẦN CHÈN (Chỉ định rõ Hình 1 đến Hình 8 hoặc Bảng 1 đến Bảng 4 từ tài liệu)
4. LỜI THOẠI THUYẾT MINH MẪU (Script để sinh viên đọc khi thuyết trình, tự tin, học thuật và cuốn hút)

Danh sách 18 Slide gợi ý:
- Slide 1: Trang tiêu đề, thông tin sinh viên, GVHD và đề tài
- Slide 2: Đặt vấn đề - Thách thức mất cân bằng dữ liệu trong IoT & Thiết bị đeo
- Slide 3: Mục tiêu nghiên cứu & 2 Câu hỏi thực nghiệm cốt lõi
- Slide 4: Tổng quan Kiến trúc Hệ thống Toàn diện (Chèn Hình 8)
- Slide 5: Tập dữ liệu PPG-DaLiA & Quy trình Tiền xử lý Tensor Chống rò rỉ (Chèn Bảng 1)
- Slide 6: Kiến trúc Mạng 1D-cVAE & Vai trò của output_padding=1
- Slide 7: Cơ chế Học phân phối ẩn & Chiến lược Annealing Beta (Chèn Hình 1)
- Slide 8: Khung 5 Nhánh Thực nghiệm Đối chứng Công bằng (n* = 3382 mẫu) (Chèn Hình 7)
- Slide 9: Kết quả Hiệu năng 5 Nhánh trên Tập Test S14, S15 (Chèn Bảng 2 & Hình 3)
- Slide 10: Phân tích Ma trận Nhầm lẫn của 5 Nhánh (Chèn Hình 2)
- Slide 11: Đánh giá Khoảng cách Phân phối Biased MMD² 108 chiều (Chèn Bảng 3 & Hình 4)
- Slide 12: Phân tích Phổ Công suất Welch PSD 0-16 Hz (Chèn Hình 5)
- Slide 13: Đối chiếu Dạng sóng Miền thời gian Thực vs Sinh (Chèn Hình 6)
- Slide 14: Thảo luận Khoa học: Vì sao ROS tốt nhất? Ý nghĩa thực thụ của TSTR = 40.18%
- Slide 15: Hiện tượng Nhầm lẫn Hoạt động Tĩnh do Góc nghiêng Trọng trường DC
- Slide 16: Triển khai Hệ thống IoT Thời gian thực & Demo Thực nghiệm (Chèn Bảng 4)
- Slide 17: Kết luận & Đóng góp Chính của Đề tài
- Slide 18: Lời cảm ơn, Hỏi & Đáp (Q&A)
```

---
*Tài liệu được khởi tạo tự động, đồng bộ dữ liệu thực nghiệm 100% từ mã nguồn và nhật ký kiểm thử GPU.*
