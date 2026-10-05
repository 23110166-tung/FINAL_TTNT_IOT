"""
Module huấn luyện bộ phân loại HAR cho 5 nhánh đối chứng thực nghiệm.
Căn cứ: Mục 3.4, 4.1, 4.3 của Đề cương đã chỉnh sửa và Mục B.7, B.8 của Phiếu nhận xét.

Các quy tắc sống còn:
1. 5 nhánh thực nghiệm: TRTR, Real+ROS, Real+TradAug, TSTR, Real+cVAE.
2. Quota mẫu bổ sung:
   - n* = ceil(median(n_0, ..., n_7)) từ số lượng train thật.
   - Bổ sung delta_n_c = max(0, n* - n_c) cho lớp c.
   - ROS, TradAug và Real+cVAE có CÙNG SỐ MẪU BỔ SUNG theo lớp.
   - TSTR: số mẫu sinh mỗi lớp đúng bằng n_c của train thật.
3. TradAug: Jittering Gaussian (sigma=0.01) + Scaling Uniform(0.95, 1.05) trên không gian chuẩn hóa.
4. Checkpoint Classifier: Chọn dựa trên Macro-F1 TRUNG BÌNH ĐỀU THEO NGƯỜI trên Validation (S12-S13):
   Macro-F1_val = (F1_S12 + F1_S13) / 2. Tuyệt đối KHÔNG gộp chung.
5. Ngân sách cập nhật: Dùng chung max_epochs, batch_size, early stopping; ghi nhận ACTUAL UPDATE STEPS.
6. Sản phẩm bàn giao: Lưu đầy đủ checkpoints/classifier_tstr_best.pth và các nhánh đối chứng.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

import os
import math
import time
import json
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import f1_score

from models_cvae import Conv1D_cVAE, HARClassifier
from data_loader import PPGDaLiADataPipeline, HARDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def evaluate_classifier_per_person(model, dataloader, device):
    """
    Đánh giá Macro-F1 trung bình đều theo người trên tập Validation.
    Tách biệt tính Macro-F1 cho từng subject_id, sau đó lấy trung bình.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_subjects = []
    
    with torch.no_grad():
        for signals, labels, sub_ids, _ in dataloader:
            signals = signals.to(device)
            logits = model(signals)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_subjects.extend(sub_ids)
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_subjects = np.array(all_subjects)
    
    unique_subs = np.unique(all_subjects)
    sub_f1_scores = {}
    
    for sub in unique_subs:
        mask = (all_subjects == sub)
        sub_pred = all_preds[mask]
        sub_target = all_targets[mask]
        # zero_division=0 theo đúng quy định đề cương
        f1 = f1_score(sub_target, sub_pred, average="macro", zero_division=0)
        sub_f1_scores[sub] = f1
        
    # Trung bình đều theo người
    mean_macro_f1 = np.mean(list(sub_f1_scores.values())) if sub_f1_scores else 0.0
    return mean_macro_f1, sub_f1_scores

def build_augmented_data(train_dataset, cvae_model, device, seed=2026, config=None):
    """
    Chuẩn bị dữ liệu cho cả 5 nhánh thực nghiệm với quota chặt chẽ.
    """
    set_seed(seed)
    
    # 1. Trích xuất mảng numpy từ train_dataset
    signals_np = train_dataset.signals.numpy()   # (N, 3, 128)
    labels_np = train_dataset.labels.numpy()     # (N,)
    
    # 2. Thống kê số lượng từng lớp thật n_c
    class_counts = {c: int((labels_np == c).sum()) for c in range(8)}
    counts_list = [class_counts[c] for c in range(8)]
    # n* = ceil(median(n_c))
    median_val = np.median(counts_list)
    n_star = int(math.ceil(median_val))
    
    # Quota bổ sung delta_n_c = max(0, n* - n_c)
    supplement_quotas = {c: max(0, n_star - class_counts[c]) for c in range(8)}
    
    print("\n[INFO] Bảng quota mẫu huấn luyện (Train S1-S11):")
    print(f"{'Lớp':<6}{'Số mẫu thật (n_c)':<20}{'Quota n*':<12}{'Bổ sung (Δn_c)':<15}")
    for c in range(8):
        print(f"{c:<6}{class_counts[c]:<20}{n_star:<12}{supplement_quotas[c]:<15}")
        
    branch_data = {}
    
    # --- NHÁNH 1: TRTR (100% Real Train) ---
    branch_data["trtr"] = (signals_np.copy(), labels_np.copy())
    
    # --- NHÁNH 2: Real + ROS (Random Over-Sampling) ---
    ros_signals, ros_labels = [signals_np.copy()], [labels_np.copy()]
    for c, delta in supplement_quotas.items():
        if delta > 0:
            c_indices = np.where(labels_np == c)[0]
            sampled_idx = np.random.choice(c_indices, size=delta, replace=True)
            ros_signals.append(signals_np[sampled_idx])
            ros_labels.append(labels_np[sampled_idx])
    branch_data["ros"] = (np.concatenate(ros_signals, axis=0), np.concatenate(ros_labels, axis=0))
    
    # --- NHÁNH 3: Real + TradAug (Jittering + Scaling) ---
    trad_signals, trad_labels = [signals_np.copy()], [labels_np.copy()]
    jitter_std = config["classifier"]["tradaug"]["jitter_std"] # 0.01
    scale_min = config["classifier"]["tradaug"]["scale_min"]   # 0.95
    scale_max = config["classifier"]["tradaug"]["scale_max"]   # 1.05
    
    for c, delta in supplement_quotas.items():
        if delta > 0:
            c_indices = np.where(labels_np == c)[0]
            sampled_idx = np.random.choice(c_indices, size=delta, replace=True)
            orig_windows = signals_np[sampled_idx].copy() # (delta, 3, 128)
            
            # 1. Jittering Gaussian sigma=0.01
            noise = np.random.normal(0, jitter_std, size=orig_windows.shape).astype(np.float32)
            # 2. Scaling Uniform(0.95, 1.05) chung cho 3 trục
            scales = np.random.uniform(scale_min, scale_max, size=(delta, 1, 1)).astype(np.float32)
            
            augmented = (orig_windows + noise) * scales
            trad_signals.append(augmented)
            trad_labels.append(labels_np[sampled_idx])
    branch_data["tradaug"] = (np.concatenate(trad_signals, axis=0), np.concatenate(trad_labels, axis=0))
    
    # --- NHÁNH 4: TSTR (100% Synthetic từ Prior) ---
    # Số mẫu sinh mỗi lớp đúng bằng n_c của train thật
    cvae_model.eval()
    tstr_signals, tstr_labels = [], []
    with torch.no_grad():
        for c in range(8):
            n_samples = class_counts[c]
            if n_samples > 0:
                syn_c = cvae_model.sample_prior(num_samples=n_samples, class_idx=c, device=device)
                tstr_signals.append(syn_c.cpu().numpy())
                tstr_labels.append(np.full((n_samples,), c, dtype=np.int64))
    branch_data["tstr"] = (np.concatenate(tstr_signals, axis=0), np.concatenate(tstr_labels, axis=0))
    
    # --- NHÁNH 5: Real + cVAE (Train thật + Dữ liệu sinh từ Prior) ---
    # Số mẫu sinh bổ sung đúng bằng delta_n_c (cùng quota như ROS và TradAug)
    cvaeaug_signals, cvaeaug_labels = [signals_np.copy()], [labels_np.copy()]
    with torch.no_grad():
        for c, delta in supplement_quotas.items():
            if delta > 0:
                syn_c = cvae_model.sample_prior(num_samples=delta, class_idx=c, device=device)
                cvaeaug_signals.append(syn_c.cpu().numpy())
                cvaeaug_labels.append(np.full((delta,), c, dtype=np.int64))
    branch_data["cvaeaug"] = (np.concatenate(cvaeaug_signals, axis=0), np.concatenate(cvaeaug_labels, axis=0))
    
    return branch_data

def train_single_branch(branch_name, train_x, train_y, val_loader, config, device, seed=2026):
    """
    Huấn luyện một nhánh phân loại với cùng ngân sách cập nhật.
    Ghi nhận actual update steps và chọn checkpoint bằng trung bình đều Macro-F1 trên val.
    """
    set_seed(seed)
    clf_cfg = config["classifier"]
    batch_size = clf_cfg["batch_size"]
    lr = clf_cfg["lr"]
    weight_decay = clf_cfg["weight_decay"]
    max_epochs = clf_cfg["max_epochs"]
    patience = clf_cfg["early_stopping_patience"]
    
    # Dataset và DataLoader
    train_dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = HARClassifier(in_channels=3, num_classes=8).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    actual_update_steps = 0
    
    checkpoint_path = config["paths"]["classifier_checkpoints"][branch_name]
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    print(f"\n---> Huấn luyện nhánh: [{branch_name.upper()}] (Số mẫu: {len(train_y)})")
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            actual_update_steps += 1
            running_loss += loss.item()
            
        # Đánh giá sau mỗi epoch: Macro-F1 TRUNG BÌNH ĐỀU THEO NGƯỜI trên Validation
        val_macro_f1, sub_f1s = evaluate_classifier_per_person(model, val_loader, device)
        
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save({
                "branch": branch_name,
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "val_per_person_f1": sub_f1s,
                "actual_update_steps": actual_update_steps,
                "seed": seed
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping tại Epoch {epoch+1} (Patience={patience}).")
                break
                
    print(f"  [HOÀN THÀNH {branch_name.upper()}] Best Epoch: {best_epoch} | Best Val Macro-F1: {best_val_macro_f1:.4f} | Actual Steps: {actual_update_steps}")
    return {
        "branch": branch_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "actual_update_steps": actual_update_steps,
        "checkpoint_path": checkpoint_path
    }

def train_all_branches(config_path="configs/config.yaml", seed=2026):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Bắt đầu huấn luyện 5 nhánh phân loại với Seed={seed} trên: {device}")
    
    # 1. Nạp pipeline dữ liệu
    pipeline = PPGDaLiADataPipeline(config)
    train_ds, val_ds, test_ds = pipeline.prepare_data()
    val_loader = DataLoader(val_ds, batch_size=config["classifier"]["batch_size"], shuffle=False)
    
    # 2. Nạp checkpoint cVAE đã huấn luyện
    cvae_ckpt_path = config["paths"]["cvae_checkpoint"]
    if not os.path.exists(cvae_ckpt_path):
        raise FileNotFoundError(f"Chưa tìm thấy checkpoint cVAE tại {cvae_ckpt_path}. Hãy chạy train_cvae.py trước!")
        
    cvae_cfg = config["cvae"]
    cvae = Conv1D_cVAE(
        in_channels=cvae_cfg["in_channels"],
        signal_channels=cvae_cfg["signal_channels"],
        condition_dim=cvae_cfg["condition_dim"],
        latent_dim=cvae_cfg["latent_dim"],
        seq_len=cvae_cfg["seq_len"]
    ).to(device)
    
    cvae_state = torch.load(cvae_ckpt_path, map_location=device)
    cvae.load_state_dict(cvae_state["model_state_dict"])
    cvae.eval()
    print(f"[INFO] Đã nạp thành công mô hình cVAE từ {cvae_ckpt_path}")
    
    # 3. Tạo dữ liệu cho cả 5 nhánh
    branch_data = build_augmented_data(train_ds, cvae, device, seed=seed, config=config)
    
    # 4. Huấn luyện lần lượt 5 nhánh
    results = {}
    branches = ["trtr", "ros", "tradaug", "tstr", "cvaeaug"]
    for b in branches:
        bx, by = branch_data[b]
        res = train_single_branch(b, bx, by, val_loader, config, device, seed=seed)
        results[b] = res
        
    # 5. Lưu tóm tắt kết quả
    summary_path = os.path.join(config["paths"]["logs_dir"], f"classifier_training_summary_seed_{seed}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Hoàn thành huấn luyện cả 5 nhánh đối chứng! Tóm tắt lưu tại: {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026, help="Seed huấn luyện")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn config")
    args = parser.parse_args()
    
    train_all_branches(config_path=args.config, seed=args.seed)
