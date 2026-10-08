# THƯ MỤC TÀI LIỆU THAM KHẢO & FILE MINH CHỨNG HỌC THUẬT
## (Official Reference Documents & Verification Files)

> **Đồ án Cuối kỳ:** Tăng cường Dữ liệu Chuỗi Thời gian Gia tốc Cổ tay bằng Mạng Sinh Biến phân có Điều kiện (1D-cVAE) cho Nhận diện Hoạt động Người (HAR) trên tập dữ liệu PPG-DaLiA  
> **Sinh viên thực hiện:** Nguyễn Bách Tùng — MSSV: 23110166  
> **Giảng viên hướng dẫn & chấm điểm:** Giảng viên Bộ môn AIoT / CNTT  
> **Kho lưu trữ GitHub:** [23110166-tung/FINAL_TTNT_IOT](https://github.com/23110166-tung/FINAL_TTNT_IOT)  
> **Thư mục lưu trữ:** `tai_lieu_tham_khao/`

---

## 1. Giới thiệu & Mục đích của Thư mục
Thư mục này chứa **đầy đủ 100% các tệp tài liệu gốc, bài báo khoa học chuẩn quốc tế (PDF), mã nguồn tiền xử lý đối sánh (MATLAB/Python), và tài liệu kỹ thuật chuẩn (PDF/HTML)** được trích dẫn từ **[1] đến [11]** trong báo cáo đồ án `NguyenBachTung_BaoCao_CuoiKy.docx` và file mã nguồn `NguyenBachTung_1D_cVAE_PPG_DaLiA.ipynb`.

Mục đích nhằm:
1. **Chứng minh tính xác thực, minh bạch tuyệt đối** của toàn bộ 11 tài liệu tham khảo: Mọi trích dẫn đều có file văn bản thực tế, tồn tại thực tế, có thể mở đọc và kiểm chứng trực tiếp.
2. **Đối chiếu chính xác từng trang, từng công thức, từng thông số kỹ thuật** được trích dẫn trong văn bản báo cáo đồ án nộp cho Giảng viên.

---

## 2. Bảng Đối chiếu Chi tiết 11 Tài liệu Tham khảo

| Mã | Tên Tệp trong Thư mục | Định dạng | Trích dẫn IEEE | Vị trí trong Báo cáo (DOCX) | Nội dung & Vai trò Khoa học Đối chiếu |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **[1]** | `[01]_UCI_PPG_DaLiA_Dataset_Documentation.pdf`<br>`[01]_UCI_PPG_DaLiA_Dataset_Documentation.html` | PDF, HTML | A. Reiss et al., *"PPG-DaLiA,"* UCI Machine Learning Repository, 2019. DOI: 10.24432/C53890 | **Mục 2.1 (Trang 4)** | Căn cứ pháp lý và thông số kỹ thuật của bộ dữ liệu chuẩn quốc tế PPG-DaLiA (8,300,000 mẫu, 15 đối tượng, Empatica E4 + RespiBAN). |
| **[2]** | `[02]_Reiss2019_Deep_PPG_Sensors.pdf`<br>`[02]_Reiss2019_Deep_PPG_Sensors_FullText.html`<br>`[02]_Reiss2019_Deep_PPG_Sensors_JATS.xml` | PDF, HTML, XML | A. Reiss et al., *"Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks,"* *Sensors*, vol. 19, no. 14, p. 3079, 2019. | **Mục 2.1 (Trang 4)** | Bài báo gốc công bố quy trình thu nghiệm thực tế 15 người, 8 hoạt động thường nhật và các tầng tiền xử lý tín hiệu quang PPG & gia tốc. |
| **[3]** | `[03]_Charlton2026_collate_ppg_dalia_dataset.m`<br>`[03]_Charlton2026_convert_subject_pickle_files_to_mat.py`<br>`[03]_Charlton2026_Script_Documentation.pdf` | MATLAB (.m), Python (.py), PDF | P. H. Charlton, *"collate_ppg_dalia_dataset.m: MATLAB Data Collation Script,"* Univ. of Cambridge, 2026. | **Mục 2.2 (Trang 4)** | Căn cứ phản biện khoa học phân biệt rạch ròi: trường `activity` (4 Hz) là nhãn HAR, còn trường `label` (0.5 Hz) là nhịp tim ECG; quy trình giữ mẫu Zero-Order Hold (ZOH). |
| **[4]** | `[04]_Kingma2013_Auto_Encoding_Variational_Bayes_ICLR.pdf` | PDF (Gốc arXiv) | D. P. Kingma & M. Welling, *"Auto-Encoding Variational Bayes,"* ICLR 2013, arXiv:1312.6114. | **Mục 3.2 & 3.4 (Trang 8, 9)** | Bài báo kinh điển đặt nền móng lý thuyết Mạng tự mã hóa biến phân (VAE) và kỹ thuật tái tham số hóa (reparameterization trick) $z = \mu + \sigma \odot \epsilon$. |
| **[5]** | `[05]_Sohn2015_Learning_Structured_Output_Representation_cVAE_NeurIPS.pdf` | PDF (Gốc NeurIPS) | K. Sohn et al., *"Learning Structured Output Representation using Deep Conditional Generative Models,"* NeurIPS 2015. | **Mục 3.2 (Trang 8)** | Bài báo nền tảng về cVAE, chứng minh công thức hàm mất mát điều kiện $L_{cVAE}$ và cách điều kiện hóa cả Encoder lẫn Decoder theo nhãn lớp $c$. |
| **[6]** | `[06]_Esteban2017_Medical_Time_Series_RCGAN.pdf` | PDF (Gốc arXiv) | C. Esteban et al., *"Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs,"* arXiv:1706.02633, 2017. | **Mục 1.3 (Trang 2)** | Luận chứng so sánh vì sao chọn cVAE thay vì GAN (tránh hiện tượng sụp đổ mode - mode collapse và mất ổn định khi sinh chuỗi thời gian y sinh). |
| **[7]** | `[07]_Gretton2012_Kernel_Two_Sample_Test_MMD_JMLR.pdf` | PDF (Gốc JMLR) | A. Gretton et al., *"A Kernel Two-Sample Test,"* *JMLR*, vol. 13, pp. 723–773, 2012. | **Mục 4.4 (Trang 14)** | Cơ sở toán học của thước đo khoảng cách phân phối MMD² (Maximum Mean Discrepancy) trên không gian RKHS với đa băng thông hàm nhân RBF Gaussian. |
| **[8]** | `[08]_PyTorch_ConvTranspose1d_Official_Documentation.pdf`<br>`[08]_PyTorch_ConvTranspose1d_Official_Documentation.html` | PDF, HTML | PyTorch Foundation, *"ConvTranspose1d Module Documentation,"* PyTorch Docs v2.5. | **Mục 3.5 (Trang 9)** | Công thức giải tích chiều dài đầu ra $L_{out}$ của tầng ConvTranspose1d, chứng minh bắt buộc phải cấu hình `output_padding=1` ở cả 4 tầng giải mã để khôi phục 128 mẫu. |
| **[9]** | `[09]_ScikitLearn_Data_Leakage_Preprocessing_Pitfalls.pdf`<br>`[09]_ScikitLearn_Data_Leakage_Preprocessing_Pitfalls.html` | PDF, HTML | Scikit-learn Developers, *"Common pitfalls in data preprocessing (Data Leakage Avoidance),"* scikit-learn docs. | **Mục 2.5 (Trang 5)** | Hướng dẫn chuẩn mực tránh rò rỉ dữ liệu (data leakage): fit Z-score scaler độc quyền trên Train fold (S1–S11), không được fit trên toàn bộ tập dữ liệu. |
| **[10]** | `[10]_SciPy_Signal_Welch_PSD_Official_Documentation.pdf`<br>`[10]_SciPy_Signal_Welch_PSD_Official_Documentation.html` | PDF, HTML | SciPy Community, *"scipy.signal.welch: Estimation of power spectral density using Welch method,"* SciPy Docs. | **Mục 4.4 (Trang 14)** | Tiêu chuẩn ước lượng mật độ phổ công suất Welch PSD (cửa sổ Hann tuần hoàn 64 điểm, fs=32 Hz) để đánh giá độ tương đồng miền tần số. |
| **[11]** | `[11]_ScikitLearn_R2_Score_Official_Documentation.pdf`<br>`[11]_ScikitLearn_R2_Score_Official_Documentation.html` | PDF, HTML | Scikit-learn Developers, *"r2_score: Coefficient of determination regression score,"* scikit-learn docs. | **Mục 5.3 (Trang 17)** | Định nghĩa toán học và công thức tính hệ số xác định phổ $R^2$ giữa mật độ phổ công suất thực tế và mẫu dữ liệu nhân tạo do cVAE sinh ra. |

---

## 3. Danh sách Tệp Chi tiết & Hướng dẫn Mở Kiểm chứng

Quý Thầy/Cô và người đọc có thể mở trực tiếp các tệp tin trong thư mục này:

1. **Các bài báo khoa học quốc tế (Định dạng PDF chuẩn):**
   - Mở trực tiếp bằng bất kỳ trình đọc PDF nào (Adobe Acrobat, Edge, Chrome, Preview).
   - Các bài báo đều giữ nguyên tiêu đề, tác giả, số DOI, trang ấn bản và các công thức toán học nguyên bản.

2. **Mã nguồn tiền xử lý dữ liệu (Định dạng MATLAB `.m` & Python `.py`):**
   - Mở bằng MATLAB hoặc trình soạn thảo mã nguồn (VS Code, Notepad, Spyder).
   - Chứa các chú thích chi tiết giải thích cấu trúc ma trận và cơ chế nội suy nhãn Zero-Order Hold.

3. **Tài liệu đặc tả kỹ thuật và API thư viện chuẩn (Định dạng PDF & HTML):**
   - Phiên bản PDF: Dễ dàng in ấn, nộp kẹp vào hồ sơ báo cáo đồ án.
   - Phiên bản HTML: Có thể mở bằng trình duyệt web để xem giao diện tương tác và tra cứu các liên kết gốc.

---
*Thư mục được biên soạn và đồng bộ tự động vào kho mã nguồn đồ án ngày 08/10/2026.*
