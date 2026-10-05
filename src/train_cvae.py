"""
Module huấn luyện mô hình 1D-cVAE trên tập dữ liệu PPG-DaLiA.
Căn cứ: Mục 3.3, 4.1 của Đề cương đã chỉnh sửa và Mục B.6 của Phiếu nhận xét.

Các quy tắc sống còn:
1. Chiều dữ liệu: D = 3 * 128 = 384.
2. L_rec = (1 / (B * D)) * sum ||x_i - x̂_i||^2 (chuẩn hóa chia D=384).
3. L_KL = - (1 / (2 * B)) * sum_i sum_j (1 + ell_ij - mu_ij^2 - exp(ell_ij)).
4. L_t = 0.5 * L_rec + (beta_t / D) * L_KL.
5. beta_t = beta_max * min(t / 25, 1) (t là epoch đã xong, bắt đầu từ 0; beta_max = 1.0).
6. Validation loss tính với beta = 1.0 CỐ ĐỊNH, cố định seed nhiễu val.
7. Early stopping chỉ bắt đầu SAU epoch 25 (hết warm-up), patience = 15.
8. Lưu riêng biệt reconstruction_loss, KL_loss, total_loss mỗi epoch.
9. Optimizer: AdamW (lr=1e-3, weight_decay=1e-4), CosineAnnealingLR (T_max=100, eta_min=1e-5).
10. Checkpoint tốt nhất lưu tại checkpoints/cvae_acc_dalia_best.pth.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

import os
import time
import json
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models_cvae import Conv1D_cVAE
from data_loader import get_dataloaders

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_cvae_loss(x, x_recon, mu, logvar, beta_t, D=384):
    """
    Tính loss theo đúng công thức Đề cương:
    L_rec: MSE trung bình trên toàn bộ B * D phần tử
    L_KL: Tổng KL trên dz chiều, trung bình trên B mẫu
    L_t = 0.5 * L_rec + (beta_t / D) * L_KL
    """
    B = x.size(0)
    
    # 1. L_rec: (1 / (B * D)) * sum ||x_i - x_recon_i||^2
    # F.mse_loss với reduction='mean' tính đúng bằng (1 / (B * D)) * tổng lỗi bình phương
    loss_rec = F.mse_loss(x_recon, x, reduction='mean')
    
    # 2. L_KL: -(1 / (2 * B)) * sum_i sum_j (1 + logvar_ij - mu_ij^2 - exp(logvar_ij))
    # Sum trên dz (32), chia cho B
    loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    
    # 3. L_t có trọng số beta_t / D
    total_loss = 0.5 * loss_rec + (beta_t / D) * loss_kl
    
    return total_loss, loss_rec, loss_kl

def evaluate_cvae(model, val_loader, device, D=384):
    """
    Đánh giá trên tập Validation với beta = 1.0 CỐ ĐỊNH và cố định seed nhiễu.
    """
    model.eval()
    val_total_loss = 0.0
    val_rec_loss = 0.0
    val_kl_loss = 0.0
    num_batches = 0
    
    # Cố định seed cho bước validation để giảm dao động lấy mẫu z
    torch.manual_seed(42)
    
    with torch.no_grad():
        for signals, labels, _, _ in val_loader:
            signals = signals.to(device)
            c_onehot = F.one_hot(labels, num_classes=8).float().to(device)
            
            x_recon, mu, logvar = model(signals, c_onehot)
            # Với validation, luôn dùng beta = 1.0 cố định
            loss_t, l_rec, l_kl = compute_cvae_loss(signals, x_recon, mu, logvar, beta_t=1.0, D=D)
            
            val_total_loss += loss_t.item()
            val_rec_loss += l_rec.item()
            val_kl_loss += l_kl.item()
            num_batches += 1
            
    if num_batches == 0:
        return 0.0, 0.0, 0.0
        
    return (
        val_total_loss / num_batches,
        val_rec_loss / num_batches,
        val_kl_loss / num_batches
    )

def train_cvae(config_path="configs/config.yaml", seed=2026):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    set_seed(seed)
    
    # Thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Khởi chạy huấn luyện cVAE với Seed={seed} trên thiết bị: {device}")
    if device.type == "cuda":
        print(f"       GPU: {torch.cuda.get_device_name(0)}")

    cvae_cfg = config["cvae"]
    batch_size = cvae_cfg["batch_size"]
    lr = cvae_cfg["lr"]
    weight_decay = cvae_cfg["weight_decay"]
    max_epochs = cvae_cfg["max_epochs"]
    beta_max = cvae_cfg["beta_max"]
    warmup_epochs = cvae_cfg["beta_warmup_epochs"]
    min_epoch_es = cvae_cfg["min_epoch_early_stop"]
    patience = cvae_cfg["early_stopping_patience"]
    
    D = 3 * cvae_cfg["seq_len"] # 384
    
    # Nạp dữ liệu
    print("[INFO] Đang nạp tập dữ liệu PPG-DaLiA...")
    train_loader, val_loader, _ = get_dataloaders(config_path, batch_size=batch_size)
    print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # Khởi tạo mô hình
    model = Conv1D_cVAE(
        in_channels=cvae_cfg["in_channels"],
        signal_channels=cvae_cfg["signal_channels"],
        condition_dim=cvae_cfg["condition_dim"],
        latent_dim=cvae_cfg["latent_dim"],
        seq_len=cvae_cfg["seq_len"]
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cvae_cfg["cosine_t_max"], eta_min=cvae_cfg["cosine_eta_min"])
    
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    
    logs = {
        "train_total_loss": [],
        "train_rec_loss": [],
        "train_kl_loss": [],
        "val_total_loss": [],
        "val_rec_loss": [],
        "val_kl_loss": [],
        "beta_values": []
    }
    
    checkpoint_path = config["paths"]["cvae_checkpoint"]
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"{'Epoch':<8}{'Beta':<8}{'Train Total':<14}{'Train Rec':<12}{'Train KL':<12}{'Val (Beta=1)':<14}")
    print(f"{'='*70}")
    
    for epoch in range(max_epochs):
        model.train()
        
        # 1. Tính lịch beta tuyến tính: beta_t = beta_max * min(epoch / 25, 1.0)
        beta_t = beta_max * min(epoch / warmup_epochs, 1.0)
        logs["beta_values"].append(beta_t)
        
        running_total = 0.0
        running_rec = 0.0
        running_kl = 0.0
        train_batches = 0
        
        for signals, labels, _, _ in train_loader:
            signals = signals.to(device)
            c_onehot = F.one_hot(labels, num_classes=8).float().to(device)
            
            optimizer.zero_grad()
            x_recon, mu, logvar = model(signals, c_onehot)
            
            loss_t, l_rec, l_kl = compute_cvae_loss(signals, x_recon, mu, logvar, beta_t=beta_t, D=D)
            loss_t.backward()
            optimizer.step()
            
            running_total += loss_t.item()
            running_rec += l_rec.item()
            running_kl += l_kl.item()
            train_batches += 1
            
        scheduler.step()
        
        epoch_train_total = running_total / train_batches
        epoch_train_rec = running_rec / train_batches
        epoch_train_kl = running_kl / train_batches
        
        # 2. Đánh giá Validation (luôn với beta = 1.0)
        val_total, val_rec, val_kl = evaluate_cvae(model, val_loader, device, D=D)
        
        logs["train_total_loss"].append(epoch_train_total)
        logs["train_rec_loss"].append(epoch_train_rec)
        logs["train_kl_loss"].append(epoch_train_kl)
        logs["val_total_loss"].append(val_total)
        logs["val_rec_loss"].append(val_rec)
        logs["val_kl_loss"].append(val_kl)
        
        print(f"{epoch+1:<8}{beta_t:<8.4f}{epoch_train_total:<14.5f}{epoch_train_rec:<12.5f}{epoch_train_kl:<12.5f}{val_total:<14.5f}", flush=True)
        
        # 3. Kiểm tra lưu Checkpoint và Early Stopping
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "seed": seed,
                "config": config
            }, checkpoint_path)
            print(f"  [SAVED] Checkpoint tốt nhất tại Epoch {best_epoch} (Val Loss: {best_val_loss:.5f}) -> {checkpoint_path}", flush=True)
        else:
            # Early stopping CHỈ kích hoạt sau epoch 25 (kết thúc warm-up)
            if epoch >= min_epoch_es:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[EARLY STOPPING] Kích hoạt dừng sớm tại Epoch {epoch+1} (Patience={patience}).")
                    break
                    
    print(f"\n[COMPLETE] Hoàn thành huấn luyện cVAE. Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.5f}")
    
    # Lưu nhật ký huấn luyện
    log_file = os.path.join(config["paths"]["logs_dir"], f"cvae_training_seed_{seed}.json")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    print(f"[INFO] Đã lưu log huấn luyện vào {log_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026, help="Seed huấn luyện cVAE")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn config")
    args = parser.parse_args()
    
    train_cvae(config_path=args.config, seed=args.seed)
