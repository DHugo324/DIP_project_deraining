import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F

# 引入你的模型架構
from model import AttentionUNet 

# ================= 設定 =================
MODEL_PATH = "checkpoints/Attention_UNet_model.pth"        # 訓練好的權重檔
INPUT_FOLDER = "RAIN_Dataset/test/typhoon" # 測試圖片來源
OUTPUT_FOLDER = "results/Attention_UNet"  # 結果輸出位置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =======================================

def inference():
    # 1. 建立輸出資料夾
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 2. 載入模型
    print(f"Loading model from {MODEL_PATH}...")
    model = AttentionUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() 

    # 3. 開始推論
    # 這裡我們只跑前 10 張來做 Demo
    files = sorted(os.listdir(INPUT_FOLDER))[:10] 
    print(f"Processing {len(files)} images...")

    with torch.no_grad():
        for filename in files:
            # 檢查副檔名
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            img_path = os.path.join(INPUT_FOLDER, filename)
            img = Image.open(img_path).convert("RGB")
            
            # 智慧調整尺寸
            w, h = img.size
            
            # 算出最接近且是 16 倍數的長寬 (為了配合 U-Net 4層 Pooling)
            # 這樣可以保持圖片比例大致不變，且不會報錯
            new_w = (w // 16) * 16
            new_h = (h // 16) * 16
            
            # 1. 轉成 Tensor 並暫時 Resize 成符合網路的尺寸
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
            
            # 使用 interpolate 來調整大小 (Bilinear 插值)
            input_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            # 2. 模型預測 (現在模型吐出來的是 "雨+霧" 的雜訊層)
            noise_pred = model(input_tensor)
            output_tensor = input_tensor - noise_pred
            
            # 3. 轉回「原始」圖片大小
            output_tensor = F.interpolate(output_tensor, size=(h, w), mode='bilinear', align_corners=False)
            
            # 限制數值在 0~1 之間 (非常重要，因為相減可能會小於 0)
            output_tensor = torch.clamp(output_tensor, 0, 1)
            
            # 轉回圖片存檔
            to_pil = transforms.ToPILImage()
            result_img = to_pil(output_tensor.squeeze(0).cpu())
            
            save_path = os.path.join(OUTPUT_FOLDER, filename)
            result_img.save(save_path)
            print(f"Saved: {save_path} (Size: {w}x{h})")

    print("Inference done! Aspect ratios preserved.")

if __name__ == "__main__":
    inference()