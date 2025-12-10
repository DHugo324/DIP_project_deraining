import cv2
import numpy as np
import os
import random
import re
import shutil

def set_global_seed(seed_value=42):
    """固定所有隨機套件的種子"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    print(f"Random seed set to: {seed_value}")

def get_random_typhoon_params():
    """隨機產生一組颱風參數"""
    params = {}
    params['gamma'] = random.uniform(1.2, 1.6)      # 變暗
    params['transmission'] = random.uniform(0.5, 0.85) # 霧濃度
    params['rain_density'] = random.uniform(0.95, 0.99) # 雨密度
    params['angle'] = random.uniform(120, 150)      # 風向
    params['length'] = int(random.uniform(20, 50))  # 雨條長度
    return params

def add_dynamic_typhoon_effect(img, params):
    """將颱風效果應用到圖片上"""
    img = img.astype(np.float32) / 255.0
    row, col, ch = img.shape

    # 1. 調整亮度
    img = np.power(img, params['gamma'])

    # 2. 準備雨層
    noise = np.random.uniform(0, 1, (row, col))
    drops = np.zeros_like(noise)
    drops[noise > params['rain_density']] = 1

    angle = params['angle']
    length = params['length']
    
    # 計算旋轉矩陣來產生斜雨
    rotation_matrix = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(length))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, rotation_matrix, (length, length))
    motion_blur_kernel = motion_blur_kernel / length

    rain_layer = cv2.filter2D(drops, -1, motion_blur_kernel)
    
    # 稍微模糊雨條並增強亮度
    rain_intensity = random.uniform(2.5, 4.5) 
    cv2.GaussianBlur(rain_layer, (3, 3), 0, dst=rain_layer)
    rain_layer = rain_layer * rain_intensity
    rain_layer_3ch = np.dstack((rain_layer, rain_layer, rain_layer))

    # 3. 隨機合成 (先雨後霧 or 先霧後雨)
    A = 1.0
    t = params['transmission']

    if 'order_seed' not in params:
        params['order_seed'] = random.random()

    if params['order_seed'] > 0.5:
        # 模式 A: 先霧後雨
        hazed_img = img * t + A * (1 - t)
        final_img = hazed_img + rain_layer_3ch
    else:
        # 模式 B: 先雨後霧
        rained_scene = img + rain_layer_3ch
        final_img = rained_scene * t + A * (1 - t)

    final_img = np.clip(final_img, 0, 1)
    return (final_img * 255).astype(np.uint8)

# ==========================================
# 自然排序函式
# ==========================================
def natural_sort_key(s):
    """
    將字串中的數字部分解析為整數，實現自然排序
    例如: ['1.png', '10.png', '2.png'] -> ['1.png', '2.png', '10.png']
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# ==========================================
# 功能 1: 指定範圍生成 (Range Selection)
# ==========================================
def process_dataset_range(source_folder, target_folder, start_idx=0, end_idx=None, seed=42):
    """
    批次處理指定範圍的圖片
    :param start_idx: 從第幾張開始 (包含)
    :param end_idx: 到第幾張結束 (不包含)，若為 None 則跑到最後
    """
    set_global_seed(seed)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]
    # 使用自然排序
    files.sort(key=natural_sort_key)

    # 切片取得指定範圍的檔案
    target_files = files[start_idx:end_idx]
    total = len(target_files)
    
    print(f"Source: {len(files)} images found.")
    print(f"Target Range: {start_idx} to {end_idx if end_idx is not None else len(files)}")
    print(f"Processing {total} images...")

    for i, filename in enumerate(target_files):
        img_path = os.path.join(source_folder, filename)
        img = cv2.imread(img_path)
        
        if img is None: continue

        params = get_random_typhoon_params()
        typhoon_img = add_dynamic_typhoon_effect(img, params)
        
        output_filename = f"{filename.split('.')[0]}.png"
        output_path = os.path.join(target_folder, output_filename)
        cv2.imwrite(output_path, typhoon_img)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total}: {filename}")

# ==========================================
# 功能 2: 一對多固定特效 (One-to-Many Fixed Effects)
# ==========================================
def process_one_to_many(source_folder, target_folder, start_idx=0, end_idx=None, m_effects=3, seed=42):
    """
    挑選前 n 張照片，每張照片都產生相同的 m 種隨機特效
    """
    set_global_seed(seed) # 確保那 m 組特效每次跑都一樣

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 1. 先產生 m 組固定的隨機參數 (Template)
    print(f"Generating {m_effects} fixed effect templates...")
    effect_templates = []
    for i in range(m_effects):
        params = get_random_typhoon_params()
        # 這裡強制給一個隨機數，確保這組特效的合成順序固定
        params['order_seed'] = random.random() 
        effect_templates.append(params)

    # 2. 取得前 n 張圖片
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]
    # 使用自然排序
    files.sort(key=natural_sort_key)

    target_files = files[start_idx:end_idx]

    print(f"Processing {len(target_files)} images, each with {m_effects} variations...")

    for i, filename in enumerate(target_files):
        img_path = os.path.join(source_folder, filename)
        img = cv2.imread(img_path)
        
        if img is None: continue

        # 對每一張圖，套用那 m 組固定的參數
        for j, params in enumerate(effect_templates):
            typhoon_img = add_dynamic_typhoon_effect(img.copy(), params) # 用 copy 避免汙染原圖
            
            # 檔名範例: image01_v0.png, image01_v1.png
            output_filename = f"{start_idx + i * m_effects + j + 1}.png" 
            output_path = os.path.join(target_folder, output_filename)
            cv2.imwrite(output_path, typhoon_img)
            
        print(f"✅ Generated variations for: {filename}")

# ==========================================
# 執行區域
# ==========================================
if __name__ == "__main__":
    input_dir = r"dataset/test/norain"
    output_dir = r"dataset/test/typhoon"
    flag = True  # 是否刪除舊資料夾並重新建立

    # --- 清理舊資料夾 ---
    if flag and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # --- 模式選擇 ---
    mode = "all" # 改成 "range" 或 "one_to_many" 來切換功能

    if mode == "all":
        process_dataset_range(input_dir, output_dir, start_idx=0, end_idx=75)
        process_one_to_many(input_dir, output_dir, start_idx=75, end_idx=100, m_effects=5)

    elif mode == "range":
        # 用法：只跑第 0 到第 50 張
        process_dataset_range(input_dir, output_dir, start_idx=0, end_idx=50)

    elif mode == "one_to_many":
        # 用法：取前 50 張圖片，每張都產生 3 種一模一樣的颱風效果
        # 結果會有 50 * 3 = 150 張圖
        process_one_to_many(input_dir, output_dir, start_idx=0, end_idx=50, m_effects=3)