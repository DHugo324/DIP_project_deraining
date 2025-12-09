import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

# ==========================================
# 1. SSIM Loss (結構相似性)
# ==========================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

# ==========================================
# 2. Edge Loss (邊緣感知損失 - 你的原創加分項)
# ==========================================
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # 定義 Laplacian Kernel 來提取邊緣
        k = torch.Tensor([[0.05, 0.25, 0.05], 
                          [0.25, -1.2, 0.25], 
                          [0.05, 0.25, 0.05]]).unsqueeze(0).unsqueeze(0)
        
        # 擴展到 3 個 RGB 通道
        self.kernel = k.expand(3, 1, 3, 3)
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        # 確保 kernel 在正確的設備上 (GPU/CPU)
        if x.is_cuda:
            self.kernel = self.kernel.cuda(x.get_device())
        self.kernel = self.kernel.type_as(x)
        
        # 提取邊緣特徵圖
        x_edge = F.conv2d(x, self.kernel, padding=1, groups=3)
        y_edge = F.conv2d(y, self.kernel, padding=1, groups=3)
        
        # 計算邊緣圖之間的差異
        return self.loss(x_edge, y_edge)

# ==========================================
# 3. Typhoon Hybrid Loss (總損失函數)
# ==========================================
class TyphoonLoss(nn.Module):
    def __init__(self, w_ssim=0.6, w_l1=0.35, w_edge=0.05):
        super(TyphoonLoss, self).__init__()
        self.ssim_module = SSIM()
        self.l1_loss = nn.L1Loss()
        self.edge_loss = EdgeLoss()
        
        # 權重設定
        self.w_ssim = w_ssim
        self.w_l1 = w_l1
        self.w_edge = w_edge

    def forward(self, output, target):
        # 1. SSIM Loss (1 - SSIM 因為我們希望 SSIM 越高越好，但 Loss 要越低越好)
        loss_ssim = 1 - self.ssim_module(output, target)
        
        # 2. L1 Loss (像素級差異)
        loss_l1 = self.l1_loss(output, target)
        
        # 3. Edge Loss (邊緣銳利度)
        loss_edge = self.edge_loss(output, target)
        
        # 總損失
        total_loss = (self.w_ssim * loss_ssim) + (self.w_l1 * loss_l1) + (self.w_edge * loss_edge)
        
        return total_loss

# 簡單測試用
if __name__ == "__main__":
    criterion = TyphoonLoss()
    x = torch.randn(4, 3, 256, 256).cuda() # 模擬 output
    y = torch.randn(4, 3, 256, 256).cuda() # 模擬 target (ground truth)
    loss = criterion(x, y)
    print(f"Loss value: {loss.item()}")
    print("TyphoonLoss is ready to use!")