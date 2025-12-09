import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from model import AttentionUNet
from dataset import TyphoonDataset, get_transforms
from loss import TyphoonLoss
import os
from datetime import datetime

# ================= 設定區域 =================
BATCH_SIZE = 4       # 顯卡記憶體如果不夠，改成 2 或 1
LEARNING_RATE = 1e-4 # 學習率
EPOCHS = 20          # 最少大約跑 10-20 epoch 就會有效果了
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 你的資料路徑 (請修改這裡!)
TYPHOON_DIR = "RAIN_Dataset/typhoon"  # 你生成的颱風圖
CLEAN_DIR = "RAIN_Dataset/norain"       # 原始的乾淨圖
# ===========================================

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. 準備資料
    dataset = TyphoonDataset(TYPHOON_DIR, CLEAN_DIR, transform=get_transforms())
    # 這裡 num_workers 設為 0 在 Windows 上最穩，不會報錯
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 

    # 2. 準備模型、Loss、優化器
    model = AttentionUNet().to(DEVICE)
    criterion = TyphoonLoss().to(DEVICE) # 使用你的 Hybrid Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. 開始訓練
    print("Start Training...")
    min_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # 使用 tqdm 顯示進度條
        loop = tqdm(loader, leave=True)
        
        for batch_idx, (imgs, targets) in enumerate(loop):
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            # 計算 Ground Truth 的雨/霧層 (Input - Clean)
            # 這裡假設 imgs 是颱風圖，targets 是乾淨圖
            noise_target = imgs - targets 

            # 模型現在要預測的是 "雜訊"
            noise_pred = model(imgs)

            # Loss 是算 "預測的雜訊" 跟 "真的雜訊" 差多少
            loss = criterion(noise_pred, noise_target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新進度條
            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # 每個 Epoch 結束後，如果 Loss 變低就存檔
        avg_loss = epoch_loss / len(loader)
        if avg_loss < min_loss:
            min_loss = avg_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/Attention_UNet_model{datetime.now().strftime('%Y%m%d_%H')}.pth")
            print(f"✅ Model saved! (Loss: {avg_loss:.4f})")

if __name__ == "__main__":
    train()