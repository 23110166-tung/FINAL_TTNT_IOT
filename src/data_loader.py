"""
Module tiền xử lý và nạp dữ liệu cho bài toán 1D-cVAE HAR trên PPG-DaLiA.
Căn cứ: Mục 2.1, 2.2, 2.3 của Đề cương đã chỉnh sửa và Mục B.1, B.3, B.4 của Phiếu nhận xét.

Các quy tắc sống còn được tuân thủ nghiêm ngặt:
1. Đọc đúng `signal/wrist/ACC` (32 Hz, 3 trục x,y,z).
2. Nhãn lấy từ `activity` (4 Hz), đồng bộ lên 32 Hz bằng Zero-Order Hold (floor(i/8)), KHÔNG nội suy tuyến tính.
3. Trường `label` là nhịp tim -> TUYỆT ĐỐI KHÔNG dùng làm nhãn HAR.
4. Loại bỏ hoàn toàn mã 0 (no activity / transient). Ánh xạ mã gốc 1-8 thành nội bộ 0-7.
5. Cửa sổ L=128, bước S=64 (chồng lấn 50%). Chỉ giữ cửa sổ 100% đồng nhất 1 nhãn duy nhất.
6. KHÔNG xóa đoạn rồi nối thời gian rời rạc trước khi chia cửa sổ (tránh tạo bước nhảy giả).
7. Bảo toàn thành phần một chiều (DC component): KHÔNG lọc thông cao 0.5 Hz, KHÔNG trừ mean từng cửa sổ.
8. Scaler z-score fit DUY NHẤT trên tập train (S1-S11), lưu scaler.pkl và áp dụng cố định cho Val/Test.
9. Đóng gói trả về tensor tín hiệu (B,3,128), nhãn (B,), kèm subject_id và start_sample_index.
10. Xuất split_manifest.json và window_statistics.csv theo quy định sản phẩm bàn giao.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import glob
import pickle
import json
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def locate_subject_file(raw_dir, subject_id):
    """
    Tìm file .pkl của subject trong các cấu trúc thư mục có thể có:
    - data/S1.pkl
    - data/S1/S1.pkl
    - data/PPG_FieldStudy/S1/S1.pkl
    """
    candidates = [
        os.path.join(raw_dir, f"{subject_id}.pkl"),
        os.path.join(raw_dir, subject_id, f"{subject_id}.pkl"),
        os.path.join(raw_dir, "PPG_FieldStudy", subject_id, f"{subject_id}.pkl"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
            
    # Tìm kiếm đệ quy nếu không thấy trực tiếp
    matches = glob.glob(os.path.join(raw_dir, "**", f"{subject_id}.pkl"), recursive=True)
    if matches:
        return matches[0]
        
    raise FileNotFoundError(f"Không tìm thấy tệp {subject_id}.pkl trong thư mục {raw_dir}")

class HARDataset(Dataset):
    """
    Dataset PyTorch đóng gói (signals, labels, subject_ids, start_sample_indices)
    """
    def __init__(self, signals, labels, subject_ids, start_indices):
        self.signals = torch.from_numpy(signals).float()       # (N, 3, 128)
        self.labels = torch.from_numpy(labels).long()          # (N,)
        self.subject_ids = subject_ids                         # list of str, length N
        self.start_indices = torch.from_numpy(start_indices).long() # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.signals[idx],
            self.labels[idx],
            self.subject_ids[idx],
            self.start_indices[idx]
        )

class PPGDaLiADataPipeline:
    def __init__(self, config=None, config_path="configs/config.yaml"):
        if config is None:
            self.config = load_config(config_path)
        else:
            self.config = config
            
        self.raw_dir = self.config["dataset"]["raw_dir"]
        self.window_size = self.config["dataset"]["window_size"]      # 128
        self.window_stride = self.config["dataset"]["window_stride"]  # 64
        self.valid_raw_codes = set(self.config["dataset"]["valid_raw_codes"]) # {1..8}
        self.raw_to_internal = {int(k): int(v) for k, v in self.config["label_mapping"]["raw_to_internal"].items()}
        
        self.train_subjects = self.config["split"]["train_subjects"]
        self.val_subjects = self.config["split"]["val_subjects"]
        self.test_subjects = self.config["split"]["test_subjects"]
        
        self.scaler = None
        self.window_stats_records = []

    def extract_subject_windows(self, subject_id):
        """
        Nạp một file subject .pkl và trích xuất các cửa sổ gia tốc hợp lệ.
        """
        file_path = locate_subject_file(self.raw_dir, subject_id)
        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        # 1. Trích xuất ACC cổ tay (32 Hz, 3 trục)
        # Cấu trúc PPG-DaLiA: data['signal']['wrist']['ACC']
        acc = data["signal"]["wrist"]["ACC"]  # (N_samples, 3)
        num_acc_samples = len(acc)

        # 2. Trích xuất nhãn hoạt động (4 Hz)
        activity = data["activity"].flatten() # (M_samples,)

        # 3. Đồng bộ nhãn lên 32 Hz bằng Zero-Order Hold (ZOH): activity[floor(i / 8)]
        # Kiểm tra độ dài tương thích
        max_valid_acc_idx = min(num_acc_samples, len(activity) * 8)
        acc = acc[:max_valid_acc_idx]
        
        # Mẫu ACC thứ i nhận nhãn activity[floor(i / 8)]
        sample_indices = np.arange(max_valid_acc_idx)
        act_indices = sample_indices // 8
        aligned_labels = activity[act_indices]

        # 4. Phân chia cửa sổ và kiểm tra độ thuần nhãn
        windows = []
        window_labels = []
        start_indices = []
        
        num_total_windows = 0
        num_kept_windows = 0
        num_dropped_code0 = 0
        num_dropped_mixed = 0
        
        kept_per_class = {c: 0 for c in range(8)}

        for start in range(0, max_valid_acc_idx - self.window_size + 1, self.window_stride):
            num_total_windows += 1
            end = start + self.window_size
            win_acc = acc[start:end]         # (128, 3)
            win_labels = aligned_labels[start:end] # (128,)

            # Kiểm tra hữu hạn (không NaN/Inf)
            if not np.isfinite(win_acc).all():
                continue

            # Kiểm tra xem có chứa mã 0 không
            if (win_labels == 0).any():
                num_dropped_code0 += 1
                continue

            # Kiểm tra xem có đồng nhất 1 nhãn duy nhất không
            unique_labels = np.unique(win_labels)
            if len(unique_labels) != 1:
                # Cửa sổ chuyển tiếp (trộn giữa 2 hoạt động) -> LOẠI
                num_dropped_mixed += 1
                continue

            raw_code = int(unique_labels[0])
            if raw_code not in self.valid_raw_codes:
                num_dropped_code0 += 1
                continue

            # Ánh xạ nhãn sang nội bộ 0-7
            internal_label = self.raw_to_internal[raw_code]
            
            # Lưu cửa sổ gia tốc: chuyển trục thành (3, 128)
            # Giữ nguyên DC component (không lọc HPF, không trừ mean)
            windows.append(win_acc.T)        # (3, 128)
            window_labels.append(internal_label)
            start_indices.append(start)
            
            num_kept_windows += 1
            kept_per_class[internal_label] += 1

        # Ghi nhận thống kê cửa sổ cho đối tượng này
        self.window_stats_records.append({
            "subject_id": subject_id,
            "total_windows_extracted": num_total_windows,
            "kept_windows": num_kept_windows,
            "dropped_code_0": num_dropped_code0,
            "dropped_mixed_transitions": num_dropped_mixed,
            **{f"class_{c}_kept": kept_per_class[c] for c in range(8)}
        })

        if len(windows) == 0:
            return (
                np.empty((0, 3, self.window_size), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                [],
                np.empty((0,), dtype=np.int64)
            )

        return (
            np.array(windows, dtype=np.float32),
            np.array(window_labels, dtype=np.int64),
            [subject_id] * len(windows),
            np.array(start_indices, dtype=np.int64)
        )

    def fit_scaler(self, train_signals):
        """
        Ước lượng mean và std của 3 trục (x, y, z) DUY NHẤT trên tập train S1-S11.
        train_signals: (N_train, 3, 128)
        """
        # Gộp tất cả mẫu thời gian của từng trục
        # Chuyển (N, 3, 128) -> (N*128, 3)
        flat_signals = train_signals.transpose(0, 2, 1).reshape(-1, 3)
        mean = np.mean(flat_signals, axis=0) # (3,)
        std = np.std(flat_signals, axis=0)   # (3,)
        std = np.maximum(std, 1e-8)          # Tránh chia cho 0

        self.scaler = {
            "mean": mean,
            "std": std
        }

        scaler_path = self.config["paths"]["scaler"]
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
            
        print(f"[INFO] Đã fit và lưu Scaler vào {scaler_path}")
        print(f"       Mean per axis (x,y,z): {mean}")
        print(f"       Std  per axis (x,y,z): {std}")

    def apply_scaler(self, signals):
        """
        Chuẩn hóa z-score cho tensor signals (N, 3, 128) dùng scaler đã fit.
        """
        if self.scaler is None:
            raise ValueError("Scaler chưa được khởi tạo hoặc fit!")
            
        mean = self.scaler["mean"].reshape(1, 3, 1)
        std = self.scaler["std"].reshape(1, 3, 1)
        return (signals - mean) / std

    def prepare_data(self):
        cache_file = os.path.join("data", "processed_cache.npz")
        scaler_path = self.config["paths"]["scaler"]
        
        # Nếu đã có cache và scaler, nạp ngay lập tức (dưới 1 giây)
        if os.path.exists(cache_file) and os.path.exists(scaler_path):
            print(f"[INFO] Nạp dữ liệu tiền xử lý từ cache: {cache_file}")
            cache = np.load(cache_file, allow_pickle=True)
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
                
            train_dataset = HARDataset(
                cache["train_sigs_norm"],
                cache["train_labs"],
                cache["train_subs"].tolist(),
                cache["train_starts"]
            )
            val_dataset = HARDataset(
                cache["val_sigs_norm"],
                cache["val_labs"],
                cache["val_subs"].tolist(),
                cache["val_starts"]
            )
            test_dataset = HARDataset(
                cache["test_sigs_norm"],
                cache["test_labs"],
                cache["test_subs"].tolist(),
                cache["test_starts"]
            )
            return train_dataset, val_dataset, test_dataset

        print("[INFO] Bắt đầu xử lý dữ liệu PPG-DaLiA từ các tệp gốc...")
        
        # 1. Trích xuất cửa sổ cho từng tập
        def process_subject_list(subjects):
            all_signals, all_labels, all_sub_ids, all_starts = [], [], [], []
            for s in subjects:
                print(f"  -> Xử lý người tham gia: {s}")
                sigs, labs, sids, starts = self.extract_subject_windows(s)
                if len(labs) > 0:
                    all_signals.append(sigs)
                    all_labels.append(labs)
                    all_sub_ids.extend(sids)
                    all_starts.append(starts)
            return (
                np.concatenate(all_signals, axis=0),
                np.concatenate(all_labels, axis=0),
                all_sub_ids,
                np.concatenate(all_starts, axis=0)
            )

        print("[INFO] Trích xuất tập Train (S1-S11)...")
        train_sigs, train_labs, train_subs, train_starts = process_subject_list(self.train_subjects)

        print("[INFO] Trích xuất tập Validation (S12-S13)...")
        val_sigs, val_labs, val_subs, val_starts = process_subject_list(self.val_subjects)

        print("[INFO] Trích xuất tập Test (S14-S15)...")
        test_sigs, test_labs, test_subs, test_starts = process_subject_list(self.test_subjects)

        # 2. Fit scaler trên Train và chuẩn hóa
        self.fit_scaler(train_sigs)
        train_sigs_norm = self.apply_scaler(train_sigs)
        val_sigs_norm = self.apply_scaler(val_sigs)
        test_sigs_norm = self.apply_scaler(test_sigs)

        # 3. Lưu split manifest
        manifest = {
            "train": {
                "subjects": self.train_subjects,
                "num_windows": len(train_labs),
                "class_counts": {int(c): int((train_labs == c).sum()) for c in range(8)}
            },
            "validation": {
                "subjects": self.val_subjects,
                "num_windows": len(val_labs),
                "class_counts": {int(c): int((val_labs == c).sum()) for c in range(8)}
            },
            "test": {
                "subjects": self.test_subjects,
                "num_windows": len(test_labs),
                "class_counts": {int(c): int((test_labs == c).sum()) for c in range(8)}
            }
        }
        manifest_path = self.config["paths"]["split_manifest"]
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Đã lưu Split Manifest vào {manifest_path}")

        # 4. Lưu window statistics
        stats_df = pd.DataFrame(self.window_stats_records)
        stats_path = self.config["paths"]["window_statistics"]
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        stats_df.to_csv(stats_path, index=False)
        print(f"[INFO] Đã lưu Thống kê cửa sổ vào {stats_path}")

        # 5. Lưu cache nhị phân nén để tăng tốc các lần nạp sau
        print(f"[INFO] Đang lưu cache nén vào {cache_file}...")
        np.savez_compressed(
            cache_file,
            train_sigs_norm=train_sigs_norm,
            train_labs=train_labs,
            train_subs=np.array(train_subs),
            train_starts=train_starts,
            val_sigs_norm=val_sigs_norm,
            val_labs=val_labs,
            val_subs=np.array(val_subs),
            val_starts=val_starts,
            test_sigs_norm=test_sigs_norm,
            test_labs=test_labs,
            test_subs=np.array(test_subs),
            test_starts=test_starts
        )
        print(f"[SUCCESS] Đã lưu cache thành công vào {cache_file}!")

        # 6. Tạo các PyTorch Dataset
        train_dataset = HARDataset(train_sigs_norm, train_labs, train_subs, train_starts)
        val_dataset = HARDataset(val_sigs_norm, val_labs, val_subs, val_starts)
        test_dataset = HARDataset(test_sigs_norm, test_labs, test_subs, test_starts)

        return train_dataset, val_dataset, test_dataset

def get_dataloaders(config_path="configs/config.yaml", batch_size=None, num_workers=0):
    config = load_config(config_path)
    if batch_size is None:
        batch_size = config["cvae"]["batch_size"]
        
    pipeline = PPGDaLiADataPipeline(config)
    train_ds, val_ds, test_ds = pipeline.prepare_data()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("[TEST] Kiểm tra nạp dữ liệu độc lập...")
    try:
        pipeline = PPGDaLiADataPipeline()
        tr, va, te = pipeline.prepare_data()
        print(f"[SUCCESS] Train: {len(tr)} cửa sổ | Val: {len(va)} cửa sổ | Test: {len(te)} cửa sổ")
    except Exception as e:
        print(f"[NOTE] Chưa tìm thấy dữ liệu raw: {e}")
