import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random

class TyphoonDataset(Dataset):
    def __init__(self, typhoon_dir, clean_dir, transform=None):
        self.typhoon_dir = typhoon_dir
        self.clean_dir = clean_dir
        # 注意：我們不依賴外部傳入的 transform，而是在 __getitem__ 內處理
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}

        self.typhoon_imgs = sorted([
            f for f in os.listdir(typhoon_dir) 
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])
        
        self.clean_imgs = sorted([
            f for f in os.listdir(clean_dir) 
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])

        if len(self.typhoon_imgs) == 0:
            raise RuntimeError(f"Found 0 images in {typhoon_dir}")

        assert len(self.typhoon_imgs) == len(self.clean_imgs), \
            f"錯誤：颱風圖 ({len(self.typhoon_imgs)}) 與原始圖 ({len(self.clean_imgs)}) 數量不符！"

    def __len__(self):
        return len(self.typhoon_imgs)

    def __getitem__(self, index):
        typhoon_path = os.path.join(self.typhoon_dir, self.typhoon_imgs[index])
        clean_path = os.path.join(self.clean_dir, self.clean_imgs[index])

        try:
            # 1. 讀取圖片 (確保是 RGB，避免變紅色)
            img_typhoon = Image.open(typhoon_path).convert("RGB")
            img_clean = Image.open(clean_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {typhoon_path}")
            raise e

        # 2. 同步隨機裁切 (Synchronized Random Crop)
        crop_size = 256
        w, h = img_typhoon.size

        # A. 如果圖片太小，先放大
        if w < crop_size or h < crop_size:
            img_typhoon = TF.resize(img_typhoon, (crop_size, crop_size))
            img_clean = TF.resize(img_clean, (crop_size, crop_size))
            w, h = img_typhoon.size 

        # B. 產生隨機座標
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        
        # 對兩張圖做一模一樣的裁切
        img_typhoon = TF.crop(img_typhoon, i, j, crop_size, crop_size)
        img_clean = TF.crop(img_clean, i, j, crop_size, crop_size)

        # C. 轉成 Tensor
        img_typhoon = TF.to_tensor(img_typhoon)
        img_clean = TF.to_tensor(img_clean)

        return img_typhoon, img_clean

# 這裡回傳 None，因為我們在 __getitem__ 裡處理好了
def get_transforms():
    return None