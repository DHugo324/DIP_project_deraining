import cv2
import numpy as np
import os
import random

def set_global_seed(seed_value=42):
    """
    固定所有隨機套件的種子，確保實驗可重現
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    print(f"🔒 Random seed set to: {seed_value}")

def get_random_typhoon_params():
    """
    隨機產生颱風參數，讓每張圖的風雨程度都不一樣
    """
    params = {}
    
    # 1. 光線變暗 (Darkness): 模擬陰天或暴雨天色
    # Gamma > 1 會變暗。範圍設在 1.2 到 1.6 之間
    params['gamma'] = random.uniform(1.2, 1.6)
    
    # 2. 霧氣濃度 (Haze): 模擬水氣
    # t 是透射率，越低霧越濃。範圍設在 0.5 (濃霧) 到 0.85 (輕霧)
    params['transmission'] = random.uniform(0.5, 0.85)
    
    # 3. 雨條密度 (Rain Density): 數值越小雨越大
    # 範圍 0.95 (大雨) ~ 0.99 (中雨)
    params['rain_density'] = random.uniform(0.95, 0.99)
    
    # 4. 風向與風速 (雨條角度與長度)
    # 角度: 模擬強風吹拂，設定在 120度 ~ 150度 之間 (或是負的 -30 ~ -60)
    params['angle'] = random.uniform(120, 150)
    # 長度: 風越大雨條拉越長，範圍 20 ~ 50 pixel
    params['length'] = int(random.uniform(20, 50))
    
    return params

def add_dynamic_typhoon_effect(img, params):
    img = img.astype(np.float32) / 255.0
    row, col, ch = img.shape

    # --- 1. 調整亮度 (Darkness) ---
    img = np.power(img, params['gamma'])

    # --- 2. 加入雨條 (Rain Streaks) ---
    noise = np.random.uniform(0, 1, (row, col))
    drops = np.zeros_like(noise)
    drops[noise > params['rain_density']] = 1

    # 建立動態模糊核 (Motion Blur Kernel)
    angle = params['angle']
    length = params['length']
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(length))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (length, length))
    motion_blur_kernel = motion_blur_kernel / length

    # 模糊化產生雨條
    rain_layer = cv2.filter2D(drops, -1, motion_blur_kernel)
    
    # 隨機調整雨條亮度 (有些雨條比較亮，有些比較暗)
    rain_intensity = random.uniform(2.5, 4.5) 
    cv2.GaussianBlur(rain_layer, (3, 3), 0, dst=rain_layer) # 稍微模糊一點比較自然
    rain_layer = rain_layer * rain_intensity

    # --- 3. 加入霧氣 (Haze) ---
    A = 1.0  # 大氣光 (白色)
    t = params['transmission']
    
    # Haze Model: I = J*t + A*(1-t)
    hazed_img = img * t + A * (1 - t)
    
    # --- 4. 合成 (霧圖 + 雨條) ---
    # 將單通道雨條轉為三通道
    rain_layer_3ch = np.dstack((rain_layer, rain_layer, rain_layer))
    
    # 疊加
    final_img = hazed_img + rain_layer_3ch

    # 確保數值範圍
    final_img = np.clip(final_img, 0, 1)
    
    return (final_img * 255).astype(np.uint8)

def process_dataset(source_folder, target_folder, seed=42):
    """
    批次處理整個資料夾
    """
    set_global_seed(seed)

    # 建立輸出資料夾 (如果不存在)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created output directory: {target_folder}")

    # 取得所有圖片檔案
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]
    files.sort()

    total = len(files)
    print(f"Found {total} images. Starting processing...")

    for i, filename in enumerate(files):
        img_path = os.path.join(source_folder, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Skipping corrupt file: {filename}")
            continue

        # 每一張圖都重新隨機產生參數
        params = get_random_typhoon_params()
        
        # 處理圖片
        typhoon_img = add_dynamic_typhoon_effect(img, params)
        
        # 存檔
        filename = f"typhoon_{i:03d}.png"
        output_path = os.path.join(target_folder, filename)
        cv2.imwrite(output_path, typhoon_img)

        # 顯示進度
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total}: {filename}")

    print("Done! All images processed.")

# ==========================================
# 設定路徑並執行
# ==========================================
if __name__ == "__main__":
    # 修改這裡: 你的 norain 資料夾路徑
    input_dir = r"RAIN_Dataset/norain" 
    
    # 修改這裡: 你想輸出的颱風圖片資料夾
    output_dir = r"RAIN_Dataset/typhoon"
    
    process_dataset(input_dir, output_dir)