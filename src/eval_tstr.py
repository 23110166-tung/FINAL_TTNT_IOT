"""
Module đánh giá toàn diện TSTR, MMD² và Welch PSD trên PPG-DaLiA.
Căn cứ: Mục 4.2, 5.2 của Đề cương đã chỉnh sửa và Mục B.9, D.1 của Phiếu nhận xét.

Các quy tắc sống còn:
1. Đánh giá Macro-F1 trên tập Test (S14, S15):
   - Tính riêng cho S14, tính riêng cho S15, sau đó tính trung bình đều: (F_S14 + F_S15)/2.
   - zero_division=0 theo đúng quy định.
   - Hiệu số hiệu năng có dấu:
     Delta_TSTR = F_TRTR - F_TSTR (dương = TSTR giảm so với TRTR).
     G_aug = F_{Real+cVAE} - F_Real (dương = cVAE cải thiện).
2. Lưu kết quả dự đoán thô từng cửa sổ vào results/raw_predictions.csv.
3. Biểu diễn đặc trưng MMD² 108 chiều:
   - 9 chiều miền thời gian: mean, std, RMS của 3 trục.
   - 99 chiều miền tần số: log(PSD + 1e-8) của 3 trục (33 bins x 3 = 99).
   - Chuẩn hóa z-score đặc trưng bằng trung bình và std của train thật.
4. MMD² Kernel & Estimator:
   - Biased estimator (giữ đường chéo).
   - Kernel RBF 3 băng thông s in {0.5, 1.0, 2.0}:
     k(u,v) = (1/3) * sum_s exp(-||u-v||^2 / (2 * (s*sigma_0)^2)).
   - sigma_0 là median khoảng cách Euclidean dương trên tối đa 2000 cửa sổ train thật.
   - Cửa sổ không chồng lấn cho dữ liệu thật, m = n = min(300, N_test, N_syn, N_train).
   - Báo cáo cả MMD²(real_test, synthetic) và baseline MMD²(real_test, real_train).
5. Welch PSD:
   - fs=32, nperseg=64, noverlap=32, nfft=64, detrend=False, scaling="density", one-sided.
   - Chuẩn hóa p_f = P_f / sum(P_f) CHỈ KHI tổng công suất > 0; gắn cờ nếu công suất = 0.
   - Khoảng cách hình dạng phổ: d_PSD = sum |p_f(real) - p_f(syn)|.
   - Báo cáo kèm tổng công suất sum(P_f * delta_f) với delta_f = 0.5 Hz.
   - Hệ số xác định R² (coefficient of determination), không gọi nhầm là tương quan Pearson.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

import os
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.signal import welch
from scipy.spatial.distance import pdist, cdist
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, r2_score

from models_cvae import Conv1D_cVAE, HARClassifier
from data_loader import PPGDaLiADataPipeline

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def compute_welch_psd(signals, fs=32, nperseg=64, noverlap=32, nfft=64):
    """
    Ước lượng Welch PSD cho mảng tín hiệu (N, 3, 128).
    Trả về:
    - freqs: (33,) dải tần số 0 đến 16 Hz, bước 0.5 Hz
    - psd: (N, 3, 33) mật độ phổ công suất
    """
    # signals: (N, 3, 128)
    freqs, psd = welch(
        signals,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        scaling='density',
        axis=-1
    )
    return freqs, psd

def extract_108d_features(signals, fs=32):
    """
    Trích xuất vector đặc trưng 108 chiều cho mỗi cửa sổ (3, 128):
    - 9 chiều: mean, std, RMS của 3 trục (3 x 3 = 9)
    - 99 chiều: log(PSD + 1e-8) của 3 trục (3 trục x 33 bins = 99)
    signals: (N, 3, 128)
    """
    N = signals.shape[0]
    
    # 1. 9 đặc trưng miền thời gian
    means = np.mean(signals, axis=2)               # (N, 3)
    stds = np.std(signals, axis=2)                 # (N, 3)
    rms = np.sqrt(np.mean(signals**2, axis=2))     # (N, 3)
    time_feats = np.concatenate([means, stds, rms], axis=1) # (N, 9)
    
    # 2. 99 đặc trưng miền tần số (log-PSD)
    _, psd = compute_welch_psd(signals, fs=fs)     # (N, 3, 33)
    log_psd = np.log(psd + 1e-8)                   # (N, 3, 33)
    freq_feats = log_psd.reshape(N, 99)            # (N, 99)
    
    # Ghép thành 108 chiều
    features = np.concatenate([time_feats, freq_feats], axis=1) # (N, 108)
    return features

def rbf_multiscale_kernel(X, Y, sigma_0, scales=(0.5, 1.0, 2.0)):
    """
    Kernel RBF đa băng thông:
    k(u,v) = (1/3) * sum_{s in scales} exp(-||u-v||^2 / (2*(s*sigma_0)^2))
    """
    # Tính ma trận bình phương khoảng cách Euclidean: ||x_i - y_j||^2
    dist_sq = cdist(X, Y, metric='sqeuclidean') # (m, n)
    k_val = np.zeros_like(dist_sq, dtype=np.float64)
    for s in scales:
        gamma = 1.0 / (2.0 * ((s * sigma_0) ** 2))
        k_val += np.exp(-gamma * dist_sq)
    k_val /= len(scales)
    return k_val

def compute_biased_mmd2(X, Y, sigma_0, scales=(0.5, 1.0, 2.0)):
    """
    Ước lượng MMD² có độ chệch (Biased Estimator - giữ phần tử đường chéo):
    MMD^2 = (1/m^2) sum k(X,X) + (1/n^2) sum k(Y,Y) - (2/(m*n)) sum k(X,Y)
    """
    m = X.shape[0]
    n = Y.shape[0]
    if m == 0 or n == 0:
        return np.nan
        
    K_XX = rbf_multiscale_kernel(X, X, sigma_0, scales)
    K_YY = rbf_multiscale_kernel(Y, Y, sigma_0, scales)
    K_XY = rbf_multiscale_kernel(X, Y, sigma_0, scales)
    
    mmd2 = (np.sum(K_XX) / (m * m)) + (np.sum(K_YY) / (n * n)) - (2.0 * np.sum(K_XY) / (m * n))
    return float(mmd2)

def evaluate_all(config_path="configs/config.yaml", seeds=(2026, 2027, 2028)):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Bắt đầu đánh giá toàn diện trên thiết bị: {device}")
    
    # 1. Nạp dữ liệu
    pipeline = PPGDaLiADataPipeline(config)
    train_ds, val_ds, test_ds = pipeline.prepare_data()
    
    # Tín hiệu và nhãn tập Test
    test_signals = test_ds.signals.numpy() # (N_test, 3, 128)
    test_labels = test_ds.labels.numpy()   # (N_test,)
    test_subs = np.array(test_ds.subject_ids)
    test_starts = test_ds.start_indices.numpy()
    
    # Tín hiệu và nhãn tập Train
    train_signals = train_ds.signals.numpy()
    train_labels = train_ds.labels.numpy()
    train_starts = train_ds.start_indices.numpy()
    
    # Trích xuất tập cửa sổ không chồng lấn (stride 128 tương đương start % 128 == 0)
    train_non_overlap_mask = (train_starts % 128 == 0)
    train_signals_no = train_signals[train_non_overlap_mask]
    train_labels_no = train_labels[train_non_overlap_mask]
    
    test_non_overlap_mask = (test_starts % 128 == 0)
    test_signals_no = test_signals[test_non_overlap_mask]
    test_labels_no = test_labels[test_non_overlap_mask]
    
    # 2. Chuẩn bị đặc trưng 108 chiều và chuẩn hóa theo train thật
    print("[INFO] Trích xuất đặc trưng 108 chiều cho MMD²...")
    train_feats_raw = extract_108d_features(train_signals)
    feat_mean = np.mean(train_feats_raw, axis=0, keepdims=True)
    feat_std = np.std(train_feats_raw, axis=0, keepdims=True)
    feat_std = np.maximum(feat_std, 1e-8)
    
    # Tính sigma_0 trên tối đa 2000 cửa sổ train thật bằng seed cố định
    np.random.seed(42)
    max_train_samples = min(2000, len(train_feats_raw))
    train_sub_idx = np.random.choice(len(train_feats_raw), size=max_train_samples, replace=False)
    train_sub_norm = (train_feats_raw[train_sub_idx] - feat_mean) / feat_std
    
    pairwise_dists = pdist(train_sub_norm, metric='euclidean')
    positive_dists = pairwise_dists[pairwise_dists > 1e-6]
    if len(positive_dists) == 0:
        raise ValueError("LỖI: Không tìm thấy khoảng cách Euclidean dương trên tập train để ước lượng sigma_0!")
    sigma_0 = float(np.median(positive_dists))
    print(f"[INFO] Băng thông cơ sở MMD² sigma_0 = {sigma_0:.4f}")
    
    # 3. Nạp cVAE để sinh mẫu phục vụ đánh giá phân phối MMD và PSD
    cvae_path = config["paths"]["cvae_checkpoint"]
    cvae_cfg = config["cvae"]
    cvae = Conv1D_cVAE(
        in_channels=cvae_cfg["in_channels"],
        signal_channels=cvae_cfg["signal_channels"],
        condition_dim=cvae_cfg["condition_dim"],
        latent_dim=cvae_cfg["latent_dim"],
        seq_len=cvae_cfg["seq_len"]
    ).to(device)
    cvae_state = torch.load(cvae_path, map_location=device)
    cvae.load_state_dict(cvae_state["model_state_dict"])
    cvae.eval()
    
    # Sinh dữ liệu synthetic cho từng lớp (300 mẫu/lớp)
    syn_signals_dict = {}
    with torch.no_grad():
        for c in range(8):
            syn_c = cvae.sample_prior(num_samples=300, class_idx=c, device=device)
            syn_signals_dict[c] = syn_c.cpu().numpy()
            
    # 4. Tính toán MMD² và Welch PSD theo từng lớp
    print("\n[INFO] Đánh giá MMD² và khoảng cách phổ Welch PSD theo từng lớp:")
    print(f"{'Lớp':<6}{'MMD²(Test, Syn)':<18}{'MMD²(Test, Train)':<18}{'d_PSD (L1)':<14}{'Tổng P_f':<12}")
    
    mmd_test_syn_list = []
    mmd_test_train_list = []
    d_psd_list = []
    
    freqs, _ = compute_welch_psd(test_signals[:2])
    delta_f = 0.5 # Hz
    
    class_dist_metrics = []
    r2_psd_list = []
    class_names = config.get("label_mapping", {}).get("class_names", {})

    for c in range(8):
        # Trích xuất cửa sổ không chồng lấn
        test_c = test_signals_no[test_labels_no == c]
        train_c = train_signals_no[train_labels_no == c]
        syn_c = syn_signals_dict[c]
        
        m = min(300, len(test_c), len(syn_c), len(train_c))
        if m > 5:
            # Chọn m mẫu bằng seed cố định
            np.random.seed(42)
            idx_test = np.random.choice(len(test_c), size=m, replace=False)
            idx_train = np.random.choice(len(train_c), size=m, replace=False)
            idx_syn = np.random.choice(len(syn_c), size=m, replace=False)
            
            # Trích và chuẩn hóa đặc trưng 108D
            f_test = (extract_108d_features(test_c[idx_test]) - feat_mean) / feat_std
            f_train = (extract_108d_features(train_c[idx_train]) - feat_mean) / feat_std
            f_syn = (extract_108d_features(syn_c[idx_syn]) - feat_mean) / feat_std
            
            mmd_syn = compute_biased_mmd2(f_test, f_syn, sigma_0)
            mmd_train = compute_biased_mmd2(f_test, f_train, sigma_0)
            mmd_test_syn_list.append(mmd_syn)
            mmd_test_train_list.append(mmd_train)
        else:
            mmd_syn, mmd_train = np.nan, np.nan
            
        # Tính PSD trung bình theo lớp
        _, psd_test = compute_welch_psd(test_signals[test_labels == c])
        _, psd_syn = compute_welch_psd(syn_c)
        
        # Trung bình trên các mẫu và 3 trục
        mean_psd_test = np.mean(psd_test, axis=(0, 1)) # (33,)
        mean_psd_syn = np.mean(psd_syn, axis=(0, 1))   # (33,)
        
        power_test = float(np.sum(mean_psd_test))
        power_syn = float(np.sum(mean_psd_syn))
        zero_power_flag = (power_test <= 0 or power_syn <= 0)
        
        # Chỉ chuẩn hóa khi tổng công suất dương, cờ báo nếu bằng 0
        if not zero_power_flag:
            p_test = mean_psd_test / power_test
            p_syn = mean_psd_syn / power_syn
            d_psd = float(np.sum(np.abs(p_test - p_syn)))
            r2_psd = float(r2_score(p_test, p_syn))
            d_psd_list.append(d_psd)
            r2_psd_list.append(r2_psd)
        else:
            d_psd = np.nan
            r2_psd = np.nan
            print(f"  [CỜ CẢNH BÁO] Lớp {c} có tổng công suất bằng 0!")
            
        class_dist_metrics.append({
            "class_idx": c,
            "class_name": class_names.get(c, f"Class_{c}"),
            "mmd2_test_syn": mmd_syn,
            "mmd2_test_train_ref": mmd_train,
            "d_psd_shape_l1": d_psd,
            "r2_psd_shape": r2_psd,
            "power_syn_total": power_syn * delta_f,
            "power_test_total": power_test * delta_f,
            "zero_power_flag": zero_power_flag
        })
        print(f"{c:<6}{mmd_syn:<18.5f}{mmd_train:<18.5f}{d_psd:<14.5f}{power_syn*delta_f:<12.5f}")
        
    mean_mmd_syn = float(np.nanmean(mmd_test_syn_list))
    mean_mmd_train = float(np.nanmean(mmd_test_train_list))
    mean_d_psd = float(np.nanmean(d_psd_list))
    mean_r2_psd = float(np.nanmean(r2_psd_list))
    print(f"\n[TRUNG BÌNH 8 LỚP]")
    print(f"  MMD²(Test, Syn)  : {mean_mmd_syn:.5f}")
    print(f"  MMD²(Test, Train): {mean_mmd_train:.5f} (Mức tham chiếu)")
    print(f"  d_PSD (L1 trung bình): {mean_d_psd:.5f}")
    print(f"  R² PSD (Hệ số xác định): {mean_r2_psd:.5f}")

    # Lưu metrics phân phối theo lớp
    dist_df = pd.DataFrame(class_dist_metrics)
    dist_path = os.path.join(config["paths"]["results_dir"], "distribution_metrics.csv")
    dist_df.to_csv(dist_path, index=False)
    print(f"[INFO] Đã lưu Chỉ số phân phối MMD² & Welch PSD vào {dist_path}")
    
    # 5. Đánh giá 5 nhánh Classifier trên tập Test (S14, S15)
    print("\n[INFO] Đánh giá hiệu năng HAR trên tập Test (S14 và S15)...")
    branches = ["trtr", "ros", "tradaug", "tstr", "cvaeaug"]
    
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    all_raw_preds = []
    branch_f1_summary = {}
    
    for b in branches:
        ckpt_path = config["paths"]["classifier_checkpoints"][b]
        if not os.path.exists(ckpt_path):
            print(f"[CẢNH BÁO] Không tìm thấy checkpoint nhánh {b} tại {ckpt_path}, bỏ qua.")
            continue
            
        clf = HARClassifier(in_channels=3, num_classes=8).to(device)
        state = torch.load(ckpt_path, map_location=device)
        clf.load_state_dict(state["model_state_dict"])
        clf.eval()
        
        preds_list, targets_list, subs_list, starts_list = [], [], [], []
        with torch.no_grad():
            for sigs, labs, sids, starts in test_loader:
                sigs = sigs.to(device)
                logits = clf(sigs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                preds_list.extend(preds)
                targets_list.extend(labs.numpy())
                subs_list.extend(sids)
                starts_list.extend(starts.numpy())
                
        preds_arr = np.array(preds_list)
        targets_arr = np.array(targets_list)
        subs_arr = np.array(subs_list)
        starts_arr = np.array(starts_list)
        
        # Ghi nhận raw predictions
        for idx in range(len(preds_arr)):
            all_raw_preds.append({
                "window_idx": idx,
                "start_sample_idx": int(starts_arr[idx]),
                "subject_id": subs_arr[idx],
                "true_label": int(targets_arr[idx]),
                "pred_label": int(preds_arr[idx]),
                "branch": b,
                "seed": state.get("seed", 2026)
            })
            
        # Macro-F1 theo người: S14, S15
        mask_s14 = (subs_arr == "S14")
        mask_s15 = (subs_arr == "S15")
        
        f1_s14 = f1_score(targets_arr[mask_s14], preds_arr[mask_s14], average="macro", zero_division=0)
        f1_s15 = f1_score(targets_arr[mask_s15], preds_arr[mask_s15], average="macro", zero_division=0)
        mean_test_f1 = (f1_s14 + f1_s15) / 2.0
        acc = accuracy_score(targets_arr, preds_arr)
        
        branch_f1_summary[b] = {
            "Macro-F1_S14": f1_s14,
            "Macro-F1_S15": f1_s15,
            "Macro-F1_Mean": mean_test_f1,
            "Accuracy": acc,
            "Actual_Steps": state.get("actual_update_steps", 0)
        }
        
    # 6. Lưu file dự đoán thô (raw predictions)
    raw_df = pd.DataFrame(all_raw_preds)
    raw_path = config["paths"]["raw_predictions"]
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    raw_df.to_csv(raw_path, index=False)
    print(f"[INFO] Đã lưu Dự đoán thô từng cửa sổ vào {raw_path}")
    
    # 7. Tính Delta_TSTR và G_aug (giữ nguyên dấu)
    summary_df = pd.DataFrame(branch_f1_summary).T
    summary_df["Delta_TSTR"] = np.nan
    summary_df["G_aug"] = np.nan
    
    if "trtr" in branch_f1_summary and "tstr" in branch_f1_summary:
        f_trtr = branch_f1_summary["trtr"]["Macro-F1_Mean"]
        f_tstr = branch_f1_summary["tstr"]["Macro-F1_Mean"]
        delta_tstr = f_trtr - f_tstr
        summary_df.loc["tstr", "Delta_TSTR"] = delta_tstr
        print(f"\n[CHỈ SỐ CHÍNH]")
        print(f"  ΔTSTR = F_TRTR - F_TSTR = {f_trtr:.4f} - {f_tstr:.4f} = {delta_tstr:+.4f}")
        
    if "trtr" in branch_f1_summary:
        f_trtr = branch_f1_summary["trtr"]["Macro-F1_Mean"]
        for b in branch_f1_summary:
            if b != "trtr":
                summary_df.loc[b, "G_aug"] = branch_f1_summary[b]["Macro-F1_Mean"] - f_trtr
                
        if "cvaeaug" in branch_f1_summary:
            f_cvaeaug = branch_f1_summary["cvaeaug"]["Macro-F1_Mean"]
            g_aug = f_cvaeaug - f_trtr
            print(f"  G_aug (Real+cVAE) = F_Real+cVAE - F_Real = {f_cvaeaug:.4f} - {f_trtr:.4f} = {g_aug:+.4f}")
        
    summary_path = config["paths"]["metrics_summary"]
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path)
    print(f"[SUCCESS] Đã lưu Bảng tóm tắt chỉ số vào {summary_path}")
    print(summary_df)

if __name__ == "__main__":
    evaluate_all()
