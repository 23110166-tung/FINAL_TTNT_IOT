"""
Module tải và giải nén tập dữ liệu PPG-DaLiA từ UCI Machine Learning Repository.
Tự động trích xuất các tệp S1.pkl ... S15.pkl vào thư mục data/.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import time
import zipfile
import urllib.request
import yaml

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def download_and_extract():
    config = load_config()
    raw_dir = config["dataset"]["raw_dir"]
    url = config["dataset"]["download_url"]
    os.makedirs(raw_dir, exist_ok=True)
    
    zip_path = os.path.join(raw_dir, "ppg_dalia.zip")
    
    # Kiểm tra xem các file S1.pkl ... S15.pkl đã tồn tại chưa
    all_exist = True
    for i in range(1, 16):
        s_file1 = os.path.join(raw_dir, f"S{i}.pkl")
        s_file2 = os.path.join(raw_dir, f"S{i}", f"S{i}.pkl")
        s_file3 = os.path.join(raw_dir, "PPG_FieldStudy", f"S{i}", f"S{i}.pkl")
        if not (os.path.exists(s_file1) or os.path.exists(s_file2) or os.path.exists(s_file3)):
            all_exist = False
            break
            
    if all_exist:
        print("[INFO] Toàn bộ dữ liệu S1.pkl ... S15.pkl đã sẵn sàng trong thư mục", raw_dir)
        return

    # Nếu chưa có zip thì tải về
    if not os.path.exists(zip_path):
        print(f"[INFO] Bắt đầu tải PPG-DaLiA từ: {url}")
        print(f"[INFO] Lưu tệp nén vào: {zip_path}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        
        t0 = time.time()
        with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB
            last_report = 0
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                mb_downloaded = downloaded / (1024 * 1024)
                
                # Báo cáo mỗi 50 MB
                if mb_downloaded - last_report >= 50:
                    last_report = mb_downloaded
                    elapsed = time.time() - t0
                    speed = mb_downloaded / max(elapsed, 0.001)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"  -> Đã tải: {mb_downloaded:.1f} MB / {total_size/(1024*1024):.1f} MB ({percent:.1f}%) - Tốc độ: {speed:.2f} MB/s")
                    else:
                        print(f"  -> Đã tải: {mb_downloaded:.1f} MB - Tốc độ: {speed:.2f} MB/s")
                        
        print(f"[SUCCESS] Tải hoàn tất trong {time.time() - t0:.1f} giây. Kích thước: {os.path.getsize(zip_path)/(1024*1024):.1f} MB")
    else:
        print(f"[INFO] Đã tìm thấy tệp {zip_path}, tiến hành giải nén...")

    # Giải nén
    print(f"[INFO] Đang giải nén các tệp S1.pkl ... S15.pkl...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for name in namelist:
            if name.endswith(".pkl") and any(f"S{i}.pkl" in name for i in range(1, 16)):
                print(f"  -> Trích xuất: {name}")
                zf.extract(name, raw_dir)
                
    print("[SUCCESS] Hoàn thành giải nén dữ liệu PPG-DaLiA vào thư mục", raw_dir)

if __name__ == "__main__":
    download_and_extract()
