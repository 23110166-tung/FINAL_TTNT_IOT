"""
Module Máy chủ Nhận diện Hoạt động Thời gian thực (Real-time Phone HAR Server)
Biến điện thoại thông minh thành cảm biến đeo IoT qua kết nối Wi-Fi cục bộ.

Tính năng:
1. Web App trực tiếp trên trình duyệt điện thoại (không cần cài app).
2. Tương thích ứng dụng Sensor Logger qua HTTP POST endpoint /sensor_logger.
3. Hỗ trợ chế độ Test mẫu thật từ PPG-DaLiA Test set.
4. Xử lý chuẩn hóa Z-score từ checkpoints/scaler.pkl và suy luận trên GPU/CPU.
5. Giao diện trực quan trên điện thoại: Tên hoạt động, độ tin cậy, biểu đồ sóng 3 trục.
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
import pickle
import socket
import asyncio
import numpy as np
import torch
import torch.nn.functional as F
from aiohttp import web

# Nạp mô hình HARClassifier từ models_cvae
from models_cvae import HARClassifier

CLASS_INFO = {
    0: {"name": "Ngồi (Sitting)", "icon": "🪑", "en": "Sitting"},
    1: {"name": "Leo cầu thang (Stairs)", "icon": "🪜", "en": "Stairs"},
    2: {"name": "Bi lắc (Table soccer)", "icon": "⚽", "en": "Table soccer"},
    3: {"name": "Đạp xe (Cycling)", "icon": "🚴", "en": "Cycling"},
    4: {"name": "Lái xe (Driving)", "icon": "🚗", "en": "Driving"},
    5: {"name": "Nghỉ trưa / Ăn (Lunch break)", "icon": "🥪", "en": "Lunch break"},
    6: {"name": "Đi bộ (Walking)", "icon": "🚶", "en": "Walking"},
    7: {"name": "Làm việc văn phòng (Working)", "icon": "💻", "en": "Working"}
}

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

class RealtimeHARPipeline:
    def __init__(self, checkpoint_path="checkpoints/classifier_ros_best.pth", scaler_path="checkpoints/scaler.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AI CORE] Khởi tạo thiết bị suy luận: {self.device}")
        
        # 1. Nạp Z-score scaler
        with open(scaler_path, "rb") as f:
            scaler_data = pickle.load(f)
            self.mean = np.array(scaler_data["mean"], dtype=np.float32) # (3,)
            self.std = np.array(scaler_data["std"], dtype=np.float32)   # (3,)
        print(f"[AI CORE] Scaler nạp thành công: Mean={self.mean}, Std={self.std}")
        
        # 2. Nạp mô hình HARClassifier
        self.model = HARClassifier(in_channels=3, num_classes=8).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"[AI CORE] Đã nạp checkpoint: {checkpoint_path} (Best Val F1={ckpt.get('best_val_macro_f1', 0):.4f})")
        
        # 3. Bộ đệm vòng 128 mẫu (4 giây tại 32 Hz)
        self.buffer_size = 128
        self.buffer = [] # danh sách các [ax, ay, az]
        self.last_pred = {"class_id": 0, "confidence": 0.0, "probs": [0.0]*8}
        
    def add_sample(self, ax, ay, az):
        """Thêm 1 mẫu gia tốc (đơn vị g) vào bộ đệm vòng"""
        self.buffer.append([ax, ay, az])
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
    def is_ready(self):
        return len(self.buffer) >= self.buffer_size
        
    def predict(self):
        if not self.is_ready():
            return None
            
        # 1. Chuyển thành numpy array (128, 3) -> (3, 128)
        raw_window = np.array(self.buffer, dtype=np.float32).T # (3, 128)
        
        # 2. Chuẩn hóa Z-score: (x - mean) / std cho từng trục
        norm_window = (raw_window - self.mean[:, None]) / self.std[:, None]
        
        # 3. Đưa vào tensor PyTorch: (1, 3, 128)
        tensor = torch.from_numpy(norm_window).unsqueeze(0).to(self.device)
        
        # 4. Suy luận
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
            
        self.last_pred = {
            "class_id": pred_id,
            "class_name": CLASS_INFO[pred_id]["name"],
            "icon": CLASS_INFO[pred_id]["icon"],
            "confidence": round(confidence * 100, 1),
            "probabilities": [round(float(p) * 100, 1) for p in probs]
        }
        return self.last_pred

# Khởi tạo pipeline toàn cục
har_pipeline = RealtimeHARPipeline()

# Trang giao diện HTML tương tác cao cho trình duyệt điện thoại
HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>AIoT HAR - Cảm biến Điện thoại Thời gian thực</title>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-card: rgba(22, 30, 49, 0.75);
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.35);
            --success: #10b981;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --border: rgba(255, 255, 255, 0.08);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg-primary); color: var(--text-main); min-height: 100vh; padding: 16px; display: flex; flex-direction: column; align-items: center; }
        .container { width: 100%; max-width: 480px; display: flex; flex-direction: column; gap: 16px; }
        
        .header { text-align: center; padding: 12px 0 6px 0; }
        .header h1 { font-size: 1.25rem; font-weight: 700; color: #60a5fa; letter-spacing: -0.5px; }
        .header p { font-size: 0.8rem; color: var(--text-sub); margin-top: 4px; }
        
        .card { background: var(--bg-card); backdrop-filter: blur(12px); border: 1px solid var(--border); border-radius: 18px; padding: 18px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
        
        .status-badge { display: inline-flex; align-items: center; gap: 6px; font-size: 0.75rem; padding: 4px 10px; border-radius: 20px; background: rgba(255,255,255,0.05); }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; }
        .dot.connected { background: var(--success); box-shadow: 0 0 10px var(--success); }
        
        .activity-display { text-align: center; padding: 18px 10px; }
        .activity-icon { font-size: 4rem; filter: drop-shadow(0 6px 16px rgba(0,0,0,0.4)); animation: pulse 2s infinite ease-in-out; }
        .activity-name { font-size: 1.5rem; font-weight: 800; margin-top: 10px; color: #fff; }
        .confidence-pill { display: inline-block; margin-top: 8px; font-size: 0.85rem; font-weight: 600; padding: 4px 14px; border-radius: 20px; background: var(--accent-glow); color: #93c5fd; border: 1px solid rgba(147,197,253,0.3); }
        
        .buffer-progress { width: 100%; height: 6px; background: rgba(255,255,255,0.08); border-radius: 10px; overflow: hidden; margin-top: 14px; }
        .buffer-bar { height: 100%; width: 0%; background: var(--accent); transition: width 0.2s ease; }
        
        .acc-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 6px; }
        .acc-box { background: rgba(0,0,0,0.25); padding: 10px 8px; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.04); }
        .acc-label { font-size: 0.7rem; color: var(--text-sub); text-transform: uppercase; font-weight: 700; }
        .acc-val { font-size: 1.1rem; font-weight: 700; font-family: monospace; margin-top: 4px; }
        .val-x { color: #f87171; }
        .val-y { color: #4ade80; }
        .val-z { color: #60a5fa; }
        
        .chart-box { height: 70px; width: 100%; margin-top: 12px; }
        canvas { width: 100%; height: 100%; }
        
        .probs-list { display: flex; flex-direction: column; gap: 8px; margin-top: 6px; }
        .prob-item { display: flex; flex-direction: column; gap: 3px; font-size: 0.75rem; }
        .prob-header { display: flex; justify-content: space-between; }
        .prob-bar-bg { width: 100%; height: 5px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; }
        .prob-bar-fill { height: 100%; width: 0%; background: #64748b; border-radius: 4px; transition: width 0.3s ease, background-color 0.3s ease; }
        .prob-bar-fill.active { background: #3b82f6; box-shadow: 0 0 8px rgba(59,130,246,0.6); }
        
        .btn-group { display: flex; flex-direction: column; gap: 10px; margin-top: 8px; }
        button { width: 100%; padding: 14px; border-radius: 14px; border: none; font-size: 0.95rem; font-weight: 700; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .btn-start { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; box-shadow: 0 4px 18px rgba(37,99,235,0.4); }
        .btn-start:active { transform: scale(0.98); }
        .btn-test { background: rgba(255,255,255,0.08); color: var(--text-main); border: 1px solid var(--border); font-size: 0.8rem; padding: 10px; }
        .btn-preset { background: rgba(255,255,255,0.06); color: var(--text-main); border: 1px solid var(--border); border-radius: 10px; padding: 10px 8px; font-size: 0.78rem; font-weight: 600; cursor: pointer; text-align: left; transition: all 0.2s ease; display: flex; align-items: center; gap: 6px; }
        .btn-preset:hover { background: rgba(59, 130, 246, 0.2); border-color: #3b82f6; transform: translateY(-1px); }
        .btn-preset:active { transform: scale(0.97); }
        
        @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.06); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AIoT HAR - PHẦN CỨNG ĐIỆN THOẠI</h1>
            <p>Hệ thống Nhận diện Hoạt động 3 Trục Thời gian thực (32 Hz)</p>
            <div style="margin-top: 8px;">
                <span class="status-badge">
                    <span class="dot" id="ws-dot"></span>
                    <span id="ws-status">Đang kết nối...</span>
                </span>
            </div>
        </div>
        
        <!-- Hoạt động hiện tại -->
        <div class="card activity-display">
            <div class="activity-icon" id="act-icon">⏳</div>
            <div class="activity-name" id="act-name">Chờ cảm biến...</div>
            <div class="confidence-pill" id="act-conf">Độ tin cậy: --%</div>
            
            <div class="buffer-progress">
                <div class="buffer-bar" id="buf-bar"></div>
            </div>
            <div style="font-size: 0.65rem; color: var(--text-sub); margin-top: 6px;" id="buf-text">Đang lấp đầy bộ đệm (0 / 128 mẫu)</div>
        </div>
        
        <!-- Gia tốc 3 trục -->
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #cbd5e1;">GIA TỐC THỜI GIAN THỰC (g)</span>
                <span style="font-size: 0.7rem; color: var(--text-sub);" id="sample-rate-text">~32 Hz</span>
            </div>
            <div class="acc-grid">
                <div class="acc-box">
                    <div class="acc-label">Trục X</div>
                    <div class="acc-val val-x" id="val-x">0.00</div>
                </div>
                <div class="acc-box">
                    <div class="acc-label">Trục Y</div>
                    <div class="acc-val val-y" id="val-y">0.00</div>
                </div>
                <div class="acc-box">
                    <div class="acc-label">Trục Z</div>
                    <div class="acc-val val-z" id="val-z">0.00</div>
                </div>
            </div>
            
            <div class="chart-box">
                <canvas id="accCanvas"></canvas>
            </div>
        </div>
        
        <!-- Nút điều khiển -->
        <div class="btn-group">
            <button class="btn-start" id="btn-toggle">
                <span>⚡ BẮT ĐẦU ĐO GIA TỐC (TRÌNH DUYỆT)</span>
            </button>
            <button class="btn-test" id="btn-test-sample">
                <span>🎲 Thử nghiệm Nạp mẫu Ngẫu nhiên (PPG-DaLiA)</span>
            </button>
        </div>

        <!-- Chế độ Kiểm thử Chuẩn 8 Hoạt động -->
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size: 0.82rem; font-weight: 700; color: #60a5fa;">🎯 TEST ĐỐI CHỨNG 8 HOẠT ĐỘNG (S14-S15)</span>
            </div>
            <div style="font-size: 0.7rem; color: var(--text-sub); margin-bottom: 10px;">
                Bấm trực tiếp từng hoạt động để nạp cửa sổ tín hiệu người thật vào AI:
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">
                <button class="btn-preset" onclick="testPreset(0)">🪑 0. Ngồi (Sitting)</button>
                <button class="btn-preset" onclick="testPreset(1)">🪜 1. Cầu thang (Stairs)</button>
                <button class="btn-preset" onclick="testPreset(2)">⚽ 2. Bi lắc (Table soccer)</button>
                <button class="btn-preset" onclick="testPreset(3)">🚴 3. Đạp xe (Cycling)</button>
                <button class="btn-preset" onclick="testPreset(4)">🚗 4. Lái xe (Driving)</button>
                <button class="btn-preset" onclick="testPreset(5)">🥪 5. Ăn trưa (Lunch break)</button>
                <button class="btn-preset" onclick="testPreset(6)">🚶 6. Đi bộ (Walking)</button>
                <button class="btn-preset" onclick="testPreset(7)">💻 7. Làm việc (Working)</button>
            </div>
        </div>

        <!-- Hướng dẫn cầm điện thoại thực tế -->
        <div class="card" style="font-size: 0.74rem; color: #cbd5e1; line-height: 1.45;">
            <div style="font-weight: 700; color: #f59e0b; margin-bottom: 6px;">💡 MẸO CẦM ĐIỆN THOẠI TEST ĐẠT CHUẨN:</div>
            <p>• <b>Đi bộ (Walking)</b>: Cầm điện thoại buông thõng tay xuống hông hoặc đút túi quần, <b>bước đi liên tục trong ít nhất 5 giây</b> (để cửa sổ 4.0s gom đủ chu kỳ bước chân).</p>
            <p style="margin-top: 4px;">• <b>Ngồi yên / Làm việc</b>: Đặt điện thoại nằm trên bàn hoặc cầm gõ bàn phím.</p>
            <p style="margin-top: 4px;">• <b>Leo cầu thang</b>: Bước đi dồn dập có lực nhún hoặc đi cầu thang thật.</p>
            <p style="margin-top: 4px;">• <b>Bi lắc / Thể thao</b>: Lắc cổ tay qua lại liên tục với biên độ nhanh.</p>
        </div>
        
        <!-- Xác suất 8 lớp -->
        <div class="card">
            <div style="font-size: 0.8rem; font-weight: 700; margin-bottom: 8px; color: #cbd5e1;">PHÂN PHỐI XÁC SUẤT 8 HOẠT ĐỘNG</div>
            <div class="probs-list" id="probs-container"></div>
        </div>
    </div>

    <script>
        const CLASS_NAMES = [
            "Ngồi (Sitting)", "Leo cầu thang (Stairs)", "Bi lắc (Table soccer)", 
            "Đạp xe (Cycling)", "Lái xe (Driving)", "Ăn trưa (Lunch break)", 
            "Đi bộ (Walking)", "Làm việc VP (Working)"
        ];
        
        // Khởi tạo thanh xác suất
        const probsContainer = document.getElementById('probs-container');
        CLASS_NAMES.forEach((name, i) => {
            const item = document.createElement('div');
            item.className = 'prob-item';
            item.innerHTML = `
                <div class="prob-header">
                    <span style="color:#cbd5e1;">${name}</span>
                    <span id="prob-val-${i}" style="font-family:monospace; color:#94a3b8;">0%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" id="prob-bar-${i}"></div>
                </div>
            `;
            probsContainer.appendChild(item);
        });

        // Thiết lập WebSocket
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        let socket;
        let isMeasuring = false;
        let sampleCount = 0;
        let lastSampleTime = performance.now();
        
        // Bộ đệm vẽ biểu đồ
        const historyLength = 60;
        const waveX = new Array(historyLength).fill(0);
        const waveY = new Array(historyLength).fill(0);
        const waveZ = new Array(historyLength).fill(0);
        
        const canvas = document.getElementById('accCanvas');
        const ctx = canvas.getContext('2d');
        
        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth * window.devicePixelRatio;
            canvas.height = canvas.parentElement.clientHeight * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        function drawWaveform() {
            const w = canvas.parentElement.clientWidth;
            const h = canvas.parentElement.clientHeight;
            ctx.clearRect(0, 0, w, h);
            
            // Vẽ đường 0g ở giữa
            ctx.strokeStyle = 'rgba(255,255,255,0.08)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
            ctx.stroke();
            
            function plotLine(data, color) {
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.8;
                ctx.beginPath();
                const step = w / (historyLength - 1);
                for (let i = 0; i < historyLength; i++) {
                    // map +/-2g to height
                    const y = (h / 2) - (data[i] / 2.0) * (h / 2);
                    if (i === 0) ctx.moveTo(0, y);
                    else ctx.lineTo(i * step, y);
                }
                ctx.stroke();
            }
            
            plotLine(waveX, '#f87171');
            plotLine(waveY, '#4ade80');
            plotLine(waveZ, '#60a5fa');
            
            requestAnimationFrame(drawWaveform);
        }
        drawWaveform();

        function connectWS() {
            socket = new WebSocket(wsUrl);
            socket.onopen = () => {
                document.getElementById('ws-dot').className = 'dot connected';
                document.getElementById('ws-status').textContent = 'Đã kết nối Server AI';
            };
            socket.onclose = () => {
                document.getElementById('ws-dot').className = 'dot';
                document.getElementById('ws-status').textContent = 'Mất kết nối. Đang thử lại...';
                setTimeout(connectWS, 2000);
            };
            socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'prediction') {
                    document.getElementById('act-icon').textContent = data.icon;
                    document.getElementById('act-name').textContent = data.class_name;
                    document.getElementById('act-conf').textContent = `Độ tin cậy: ${data.confidence}%`;
                    
                    if (data.latest_acc) {
                        document.getElementById('val-x').textContent = data.latest_acc[0].toFixed(2);
                        document.getElementById('val-y').textContent = data.latest_acc[1].toFixed(2);
                        document.getElementById('val-z').textContent = data.latest_acc[2].toFixed(2);
                    }
                    if (data.waveform && data.waveform.length > 0) {
                        data.waveform.forEach(pt => {
                            waveX.shift(); waveX.push(pt[0]);
                            waveY.shift(); waveY.push(pt[1]);
                            waveZ.shift(); waveZ.push(pt[2]);
                        });
                    }
                    
                    data.probabilities.forEach((p, idx) => {
                        const bar = document.getElementById(`prob-bar-${idx}`);
                        const val = document.getElementById(`prob-val-${idx}`);
                        bar.style.width = `${p}%`;
                        val.textContent = `${p}%`;
                        if (idx === data.class_id) {
                            bar.className = 'prob-bar-fill active';
                            val.style.color = '#60a5fa';
                            val.style.fontWeight = 'bold';
                        } else {
                            bar.className = 'prob-bar-fill';
                            val.style.color = '#94a3b8';
                            val.style.fontWeight = 'normal';
                        }
                    });
                } else if (data.type === 'sensor_update') {
                    if (data.acc) {
                        document.getElementById('val-x').textContent = data.acc[0].toFixed(2);
                        document.getElementById('val-y').textContent = data.acc[1].toFixed(2);
                        document.getElementById('val-z').textContent = data.acc[2].toFixed(2);
                        waveX.shift(); waveX.push(data.acc[0]);
                        waveY.shift(); waveY.push(data.acc[1]);
                        waveZ.shift(); waveZ.push(data.acc[2]);
                    }
                    const pct = Math.min(100, Math.round((data.count / 128) * 100));
                    document.getElementById('buf-bar').style.width = `${pct}%`;
                    document.getElementById('buf-text').textContent = pct < 100 ? `Đang lấp đầy cửa sổ 4.0s: ${data.count} / 128 mẫu` : `Cửa sổ đầy (128 mẫu) - Đang suy luận liên tục`;
                } else if (data.type === 'buffer_status') {
                    const pct = Math.min(100, Math.round((data.count / 128) * 100));
                    document.getElementById('buf-bar').style.width = `${pct}%`;
                    if (pct < 100) {
                        document.getElementById('buf-text').textContent = `Đang lấp đầy cửa sổ 4.0s: ${data.count} / 128 mẫu`;
                    } else {
                        document.getElementById('buf-text').textContent = `Cửa sổ đầy (128 mẫu) - Đang suy luận liên tục`;
                    }
                }
            };
        }
        connectWS();

        // Xử lý sự kiện cảm biến gia tốc
        function handleMotion(event) {
            let acc = event.accelerationIncludingGravity || event.acceleration;
            if (!acc || acc.x === null) return;
            
            // Đơn vị của DeviceMotion là m/s^2. Quy đổi về g: 1g = 9.80665 m/s^2
            const ax = acc.x / 9.80665;
            const ay = acc.y / 9.80665;
            const az = acc.z / 9.80665;
            
            document.getElementById('val-x').textContent = ax.toFixed(2);
            document.getElementById('val-y').textContent = ay.toFixed(2);
            document.getElementById('val-z').textContent = az.toFixed(2);
            
            // Cập nhật mảng sóng
            waveX.shift(); waveX.push(ax);
            waveY.shift(); waveY.push(ay);
            waveZ.shift(); waveZ.push(az);
            
            // Gửi mẫu qua WebSocket
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({
                    type: 'sensor_sample',
                    ax: ax, ay: ay, az: az,
                    timestamp: performance.now()
                }));
            }
            
            sampleCount++;
        }

        // Bật tắt đo cảm biến
        const btnToggle = document.getElementById('btn-toggle');
        btnToggle.addEventListener('click', async () => {
            if (!isMeasuring) {
                // Yêu cầu quyền trên iOS 13+
                if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
                    try {
                        const permission = await DeviceMotionEvent.requestPermission();
                        if (permission !== 'granted') {
                            alert('Quyền truy cập cảm biến bị từ chối!');
                            return;
                        }
                    } catch (err) {
                        console.error(err);
                    }
                }
                
                window.addEventListener('devicemotion', handleMotion);
                isMeasuring = true;
                btnToggle.innerHTML = '<span>🛑 DỪNG ĐO CẢM BIẾN</span>';
                btnToggle.style.background = '#dc2626';
            } else {
                window.removeEventListener('devicemotion', handleMotion);
                isMeasuring = false;
                btnToggle.innerHTML = '<span>⚡ BẮT ĐẦU ĐO GIA TỐC</span>';
                btnToggle.style.background = 'linear-gradient(135deg, #2563eb, #1d4ed8)';
            }
        });

        // Nút nạp mẫu test thật từ PPG-DaLiA
        document.getElementById('btn-test-sample').addEventListener('click', () => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({ type: 'inject_test_sample' }));
            }
        });

        function testPreset(classId) {
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({ type: 'inject_test_sample', class_id: classId }));
            }
        }
    </script>
</body>
</html>
"""

# Quản lý các kết nối WebSocket
connected_websockets = set()

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    connected_websockets.add(ws)
    print(f"[SERVER] Thiết bị kết nối thành công: {request.remote}")
    
    sample_counter = 0
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                
                if data.get("type") == "sensor_sample":
                    ax = float(data.get("ax", 0.0))
                    ay = float(data.get("ay", 0.0))
                    az = float(data.get("az", 0.0))
                    
                    har_pipeline.add_sample(ax, ay, az)
                    sample_counter += 1
                    
                    # Cập nhật trạng thái bộ đệm
                    if sample_counter % 8 == 0:
                        await ws.send_json({
                            "type": "buffer_status",
                            "count": len(har_pipeline.buffer)
                        })
                        
                    # Cứ mỗi 16 mẫu (~0.5 giây), thực hiện suy luận 1 lần
                    if sample_counter % 16 == 0 and har_pipeline.is_ready():
                        pred = har_pipeline.predict()
                        if pred:
                            await ws.send_json({
                                "type": "prediction",
                                **pred
                            })
                            print(f"[{time.strftime('%H:%M:%S')}] Nhận diện: {pred['icon']} {pred['class_name']} ({pred['confidence']}%)")
                            
                elif data.get("type") == "inject_test_sample":
                    # Sinh mẫu giả lập từ một hoạt động trong test cache để kiểm tra
                    cache_path = "data/processed_cache.npz"
                    if os.path.exists(cache_path):
                        cache = np.load(cache_path)
                        test_sigs = cache["test_sigs_norm"]
                        test_labs = cache["test_labs"]
                        
                        target_c = data.get("class_id", None)
                        if target_c is not None and 0 <= int(target_c) <= 7:
                            c_indices = np.where(test_labs == int(target_c))[0]
                            rand_idx = int(np.random.choice(c_indices)) if len(c_indices) > 0 else np.random.randint(0, len(test_labs))
                        else:
                            rand_idx = np.random.randint(0, len(test_labs))
                            
                        true_c = int(test_labs[rand_idx])
                        
                        # Đưa vào mô hình suy luận
                        tensor = torch.from_numpy(test_sigs[rand_idx]).unsqueeze(0).to(har_pipeline.device)
                        with torch.no_grad():
                            logits = har_pipeline.model(tensor)
                            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                            pred_c = int(np.argmax(probs))
                            
                        unnorm = test_sigs[rand_idx] * har_pipeline.std[:, None] + har_pipeline.mean[:, None]
                        latest_acc = [round(float(unnorm[0, -1]), 2), round(float(unnorm[1, -1]), 2), round(float(unnorm[2, -1]), 2)]
                        waveform = [[round(float(unnorm[0, i]), 2), round(float(unnorm[1, i]), 2), round(float(unnorm[2, i]), 2)] for i in range(0, 128, 3)]
                        
                        pred_res = {
                            "class_id": pred_c,
                            "class_name": f"{CLASS_INFO[pred_c]['name']} [Mẫu Thật Lớp {true_c}]",
                            "icon": CLASS_INFO[pred_c]["icon"],
                            "confidence": round(float(probs[pred_c]) * 100, 1),
                            "probabilities": [round(float(p) * 100, 1) for p in probs],
                            "latest_acc": latest_acc,
                            "waveform": waveform
                        }
                        await ws.send_json({"type": "buffer_status", "count": 128})
                        await ws.send_json({"type": "prediction", **pred_res})
                        print(f"[TEST MẪU CHUẨN] Yêu cầu lớp {target_c} -> Nhãn thật: {true_c} ({CLASS_INFO[true_c]['en']}) -> AI Dự đoán: {pred_c} ({pred_res['confidence']}%)")
                        
    finally:
        connected_websockets.remove(ws)
        print(f"[SERVER] Thiết bị đã ngắt kết nối: {request.remote}")
    return ws

async def http_sensor_logger_handler(request):
    """
    Endpoint nhận dữ liệu từ ứng dụng 'Sensor Logger' qua HTTP Push.
    Hỗ trợ cả POST (dữ liệu cảm biến) và GET (kiểm tra kết nối từ trình duyệt).
    """
    if request.method == "GET":
        return web.json_response({
            "status": "ready",
            "message": "Endpoint /sensor_logger đang hoạt động sẵn sàng nhận HTTP POST từ app Sensor Logger!",
            "target_url": f"http://{get_local_ip()}:8088/sensor_logger"
        })
        
    try:
        text = await request.text()
        if not text or not text.strip():
            print(f"[SENSOR LOGGER] Nhận kết nối test ping từ {request.remote}!")
            return web.json_response({"status": "ok", "message": "Test ping received successfully!"})
            
        try:
            body = json.loads(text)
        except Exception:
            print(f"[SENSOR LOGGER] Dữ liệu không phải JSON từ {request.remote}: {text[:80]}")
            return web.json_response({"status": "ok", "message": "Received text payload"})
            
        if isinstance(body, dict):
            payload = body.get("payload", body.get("data", []))
        elif isinstance(body, list):
            payload = body
        else:
            payload = []
            
        samples_added = 0
        last_acc = [0.0, 0.0, 0.0]
        
        for item in payload:
            name = item.get("name", "").lower()
            if "acc" in name or not name: # "accelerometer", "totalacceleration", v.v.
                vals = item.get("values")
                if isinstance(vals, dict):
                    raw_x = float(vals.get("x", 0.0))
                    raw_y = float(vals.get("y", 0.0))
                    raw_z = float(vals.get("z", 0.0))
                elif isinstance(vals, (list, tuple)) and len(vals) >= 3:
                    raw_x = float(vals[0])
                    raw_y = float(vals[1])
                    raw_z = float(vals[2])
                else:
                    continue
                    
                # Tự động nhận diện đơn vị: nếu độ lớn > 4.0 tức là m/s^2, chia 9.80665 để ra g
                mag = (raw_x**2 + raw_y**2 + raw_z**2)**0.5
                scale = 9.80665 if mag > 4.0 else 1.0
                
                ax = raw_x / scale
                ay = raw_y / scale
                az = raw_z / scale
                
                har_pipeline.add_sample(ax, ay, az)
                samples_added += 1
                last_acc = [ax, ay, az]
                
        if samples_added > 0:
            print(f"[SENSOR LOGGER] Nạp {samples_added} mẫu từ {request.remote} | Bộ đệm: {len(har_pipeline.buffer)}/128")
                
        # Broadcast sensor_update và mẫu mới nhất đến web dashboard
        for ws in list(connected_websockets):
            try:
                await ws.send_json({
                    "type": "sensor_update",
                    "acc": [round(float(last_acc[0]), 2), round(float(last_acc[1]), 2), round(float(last_acc[2]), 2)],
                    "count": len(har_pipeline.buffer)
                })
            except Exception:
                pass
                
        # Suy luận khi đủ dữ liệu
        if har_pipeline.is_ready():
            pred = har_pipeline.predict()
            if pred:
                pred["latest_acc"] = [round(float(last_acc[0]), 2), round(float(last_acc[1]), 2), round(float(last_acc[2]), 2)]
                pred["waveform"] = [[round(float(s[0]), 2), round(float(s[1]), 2), round(float(s[2]), 2)] for s in har_pipeline.buffer[-40:]]
                t_str = time.strftime('%H:%M:%S')
                print(f"[{t_str}] [DỰ ĐOÁN] {pred['icon']} {pred['class_name']} ({pred['confidence']}%) | Acc: [{last_acc[0]:.2f}, {last_acc[1]:.2f}, {last_acc[2]:.2f}]g")
                
                # Gửi thông báo đến tất cả web client đang mở
                for ws in list(connected_websockets):
                    try:
                        await ws.send_json({"type": "prediction", **pred})
                    except Exception:
                        pass
                        
        return web.json_response({
            "status": "ok",
            "samples_processed": samples_added,
            "buffer_count": len(har_pipeline.buffer),
            "latest_prediction": har_pipeline.last_pred.get("class_name") if har_pipeline.is_ready() else "Buffering..."
        })
    except Exception as e:
        print(f"[CẢNH BÁO /sensor_logger] Lỗi xử lý: {e}")
        return web.json_response({"error": str(e)}, status=400)

async def index_handler(request):
    return web.Response(text=HTML_PAGE, content_type="text/html")

def main():
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/sensor_logger", http_sensor_logger_handler)
    app.router.add_post("/sensor_logger", http_sensor_logger_handler)
    
    def find_free_port(start=8088):
        for p in range(start, start + 50):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', p)) != 0:
                    return p
        return start
        
    local_ip = get_local_ip()
    port = find_free_port(start=8088)
    
    print("\n" + "="*65)
    print("      🚀 MÁY CHỦ THỜI GIAN THỰC AIoT HAR SẴN SÀNG CHẠY! 🚀")
    print("="*65)
    print(f"\n[BƯỚC 1] Đảm bảo Điện thoại và Máy tính cùng kết nối MỘT mạng Wi-Fi.")
    print(f"\n[BƯỚC 2] Mở trình duyệt trên điện thoại (Safari / Chrome) và truy cập:")
    print(f"\n      👉 http://{local_ip}:{port}")
    print(f"\n      (Hoặc thử ngay trên máy tính: http://localhost:{port})")
    print("\n[BƯỚC 3] Trên màn hình điện thoại:")
    print("   1. Bấm nút: '⚡ BẮT ĐẦU ĐO GIA TỐC'")
    print("   2. Cầm điện thoại ngồi yên (Sitting), đứng dậy đi bộ (Walking),")
    print("      hoặc lắc mô phỏng đạp xe / leo cầu thang...")
    print("   3. Kết quả nhận diện sẽ hiển thị ngay trên điện thoại & màn hình này!")
    print("="*65 + "\n")
    
    web.run_app(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
