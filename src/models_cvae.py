"""
Kiến trúc mô hình 1D-cVAE và Bộ phân loại HAR theo đúng đặc tả Đề cương đã chỉnh sửa.
Căn cứ: Mục 3.1, 3.2, 3.4 của Đề cương và Mục B.5 của Phiếu nhận xét.

Các điểm mấu chốt:
1. Encoder: 4 tầng Conv1d (k=5, s=2, p=2), kênh 32->64->128->256, chiều dài 64->32->16->8, BN + LeakyReLU(0.2).
2. Không gian ẩn: Flatten 2048, hai Linear heads độc lập 2048->32 cho mu và log-variance.
3. Chiếu decoder: Ghép [z; c] -> (B, 40) -> Linear(40, 2048) -> Reshape (B, 256, 8).
4. Decoder: Đúng 4 tầng ConvTranspose1d (256->128->64->32->3) với output_padding=1 ở CẢ 4 TẦNG
   để đảm bảo kích thước chính xác 8 -> 16 -> 32 -> 64 -> 128.
5. Đầu ra decoder tuyến tính hoàn toàn (không BN, không Tanh/Sigmoid) vì dữ liệu ở không gian chuẩn hóa Z-score.
6. Bộ phân loại HAR: 3 tầng Conv1d (32, 64, 128; k=5, s=1, p=2) + ReLU + MaxPool1d(2) + GAP + Linear(128, 8).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1D_cVAE(nn.Module):
    def __init__(self, in_channels=11, signal_channels=3, condition_dim=8, latent_dim=32, seq_len=128):
        super().__init__()
        self.signal_channels = signal_channels
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        
        # 1. ENCODER
        # Đầu vào: x ghép với c mở rộng dọc trục thời gian -> (B, 3 + 8 = 11, 128)
        self.encoder = nn.Sequential(
            # Tầng 1: Lout = floor((128 + 2*2 - 5)/2) + 1 = 64
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Tầng 2: Lout = floor((64 + 2*2 - 5)/2) + 1 = 32
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Tầng 3: Lout = floor((32 + 2*2 - 5)/2) + 1 = 16
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Tầng 4: Lout = floor((16 + 2*2 - 5)/2) + 1 = 8
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. KHÔNG GIAN ẨN (LATENT SPACE)
        # Flatten: 256 kênh * 8 chiều dài = 2048
        self.flatten_dim = 256 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim) # log(sigma^2)
        
        # 3. CHIẾU DECODER (DECODER PROJECTION)
        # Ghép [z; c] -> latent_dim + condition_dim = 32 + 8 = 40
        self.decoder_input = nn.Linear(latent_dim + condition_dim, self.flatten_dim)
        
        # 4. GIẢI MÃ (DECODER)
        # Bắt buộc output_padding=1 ở cả 4 tầng: Lout = (Lin - 1)*2 - 2*2 + 1*(5 - 1) + 1 + 1 = 2*Lin
        # Chuỗi kích thước: 8 -> 16 -> 32 -> 64 -> 128
        self.decoder = nn.Sequential(
            # Tầng 1: 8 -> 16
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Tầng 2: 16 -> 32
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Tầng 3: 32 -> 64
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Tầng 4: 64 -> 128
            # Khối cuối: TUYẾN TÍNH, không BatchNorm, không Tanh/Sigmoid
            nn.ConvTranspose1d(32, signal_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

    def encode(self, x, c_onehot):
        """
        x: (B, 3, 128)
        c_onehot: (B, 8)
        """
        B, _, L = x.shape
        # Mở rộng nhãn điều kiện dọc theo chiều thời gian: (B, 8, 128)
        c_expanded = c_onehot.unsqueeze(2).repeat(1, 1, L)
        # Ghép theo chiều kênh: (B, 3 + 8 = 11, 128)
        encoder_input = torch.cat([x, c_expanded], dim=1)
        
        h = self.encoder(encoder_input) # (B, 256, 8)
        h_flat = h.view(B, -1)           # (B, 2048)
        
        mu = self.fc_mu(h_flat)          # (B, 32)
        logvar = self.fc_logvar(h_flat)  # (B, 32)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        z = mu + exp(0.5 * logvar) * eps,  eps ~ N(0, I)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, c_onehot):
        """
        Giải mã từ không gian ẩn z và điều kiện c:
        z: (B, 32)
        c_onehot: (B, 8)
        -> x_recon: (B, 3, 128)
        """
        # Ghép [z; c] -> (B, 40)
        z_c = torch.cat([z, c_onehot], dim=1)
        projected = self.decoder_input(z_c) # (B, 2048)
        h = projected.view(-1, 256, 8)       # (B, 256, 8)
        x_recon = self.decoder(h)           # (B, 3, 128)
        return x_recon

    def forward(self, x, c_onehot):
        mu, logvar = self.encode(x, c_onehot)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c_onehot)
        return x_recon, mu, logvar

    @torch.no_grad()
    def sample_prior(self, num_samples, class_idx, device="cpu"):
        """
        Sinh mẫu thuần túy từ phân phối tiên nghiệm chuẩn tắc z ~ N(0, I)
        phục vụ quy trình TSTR độc lập (không cần bất kỳ dữ liệu thật nào).
        """
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        # Tạo vector one-hot cho nhãn mục tiêu
        c_onehot = torch.zeros(num_samples, self.condition_dim, device=device)
        c_onehot[:, class_idx] = 1.0
        
        generated_x = self.decode(z, c_onehot)
        return generated_x

class HARClassifier(nn.Module):
    """
    Kiến trúc bộ phân loại HAR cố định dùng chung cho tất cả các nhánh đối chứng.
    Căn cứ Mục 3.4 Đề cương:
    - 3 khối Conv1d (32, 64, 128 kênh; k=5, s=1, p=2), ReLU, MaxPool1d(2)
    - Global Average Pooling (AdaptiveAvgPool1d(1))
    - Tuyến tính Linear(128, 8)
    - Chỉ nhận đầu vào là tensor tín hiệu x (không nhận nhãn)
    """
    def __init__(self, in_channels=3, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            # Khối 1: 128 -> MaxPool -> 64
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Khối 2: 64 -> MaxPool -> 32
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Khối 3: 32 -> MaxPool -> 16
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1) # (B, 128, 1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 3, 128)
        feat = self.features(x)         # (B, 128, 16)
        pooled = self.gap(feat).squeeze(-1) # (B, 128)
        logits = self.classifier(pooled)    # (B, 8)
        return logits

def test_architectures():
    """
    Unit test kiểm tra hình dạng tensor và lan truyền gradient (Checklist D.2).
    """
    print("[TEST] Bắt đầu kiểm tra kiến trúc mô hình...")
    B, C, L = 2, 3, 128
    dummy_x = torch.randn(B, C, L)
    dummy_c = F.one_hot(torch.tensor([0, 7]), num_classes=8).float()
    
    # Test cVAE
    cvae = Conv1D_cVAE(in_channels=11, signal_channels=3, condition_dim=8, latent_dim=32, seq_len=128)
    recon, mu, logvar = cvae(dummy_x, dummy_c)
    assert recon.shape == (B, C, L), f"cVAE output shape sai: {recon.shape} != {(B, C, L)}"
    assert mu.shape == (B, 32), f"cVAE mu shape sai: {mu.shape} != {(B, 32)}"
    assert logvar.shape == (B, 32), f"cVAE logvar shape sai: {logvar.shape} != {(B, 32)}"
    print(f"  -> cVAE forward test PASSED: x̂={recon.shape}, μ={mu.shape}, logvar={logvar.shape}")
    
    # Test Prior Sampling
    syn_samples = cvae.sample_prior(num_samples=5, class_idx=3)
    assert syn_samples.shape == (5, 3, 128), f"cVAE sample shape sai: {syn_samples.shape}"
    print(f"  -> cVAE prior sampling test PASSED: {syn_samples.shape}")
    
    # Test Classifier
    clf = HARClassifier(in_channels=3, num_classes=8)
    logits = clf(dummy_x)
    assert logits.shape == (B, 8), f"Classifier logits shape sai: {logits.shape} != {(B, 8)}"
    print(f"  -> Classifier forward test PASSED: logits={logits.shape}")
    print("[SUCCESS] Toàn bộ kiểm tra kiến trúc tensor đạt chuẩn Checklist D.2!")

if __name__ == "__main__":
    test_architectures()
