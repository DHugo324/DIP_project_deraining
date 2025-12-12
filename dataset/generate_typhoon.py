import cv2
import numpy as np
import os
import random
import re
import shutil

# ==========================================
# Configuration & Hyperparameters
# ==========================================
SEED = 42
INPUT_DIR = r"dataset/test/norain"
OUTPUT_DIR = r"dataset/test/typhoon"

# Generation Mode: 'random_range', 'one_to_many', or 'mixed'
MODE = "mixed" 

def set_global_seed(seed_value=42):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    print(f"Random seed set to: {seed_value}")

def get_random_typhoon_params():
    """Generates random parameters for typhoon simulation."""
    params = {}
    params['gamma'] = random.uniform(1.2, 1.6)          # Luminance drop (Darkness)
    params['transmission'] = random.uniform(0.5, 0.85)  # Haze density
    params['rain_density'] = random.uniform(0.95, 0.99) # Drop density
    params['angle'] = random.uniform(120, 150)          # Wind direction (Slant rain)
    params['length'] = int(random.uniform(20, 50))      # Rain streak length
    return params

def natural_sort_key(s):
    """Sorts strings naturally (e.g., 1.png, 2.png, 10.png)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def add_dynamic_typhoon_effect(img, params):
    """Applies the physics-based typhoon effect to a single image."""
    img = img.astype(np.float32) / 255.0
    row, col, _ = img.shape

    # 1. Adjust Luminance (Gamma Correction)
    img = np.power(img, params['gamma'])

    # 2. Generate Rain Layer
    noise = np.random.uniform(0, 1, (row, col))
    drops = np.zeros_like(noise)
    drops[noise > params['rain_density']] = 1

    # Create slant rain using rotation matrix
    angle, length = params['angle'], params['length']
    rotation_matrix = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(length))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, rotation_matrix, (length, length))
    motion_blur_kernel = motion_blur_kernel / length

    rain_layer = cv2.filter2D(drops, -1, motion_blur_kernel)
    
    # Gaussian blur and intensity scaling
    rain_intensity = random.uniform(2.5, 4.5)
    cv2.GaussianBlur(rain_layer, (3, 3), 0, dst=rain_layer)
    rain_layer = rain_layer * rain_intensity
    rain_layer_3ch = np.dstack((rain_layer, rain_layer, rain_layer))

    # 3. Composite Layers (Random Order: Rain-then-Haze or Haze-then-Rain)
    A = 1.0
    t = params['transmission']
    if 'order_seed' not in params:
        params['order_seed'] = random.random()

    if params['order_seed'] > 0.5:
        # Order A: Haze first, then Rain
        hazed_img = img * t + A * (1 - t)
        final_img = hazed_img + rain_layer_3ch
    else:
        # Order B: Rain first, then Haze
        rained_scene = img + rain_layer_3ch
        final_img = rained_scene * t + A * (1 - t)

    final_img = np.clip(final_img, 0, 1)
    return (final_img * 255).astype(np.uint8)

def process_dataset(source_folder, target_folder, mode='random_range', start_idx=0, end_idx=None, m_effects=3):
    """
    Main processing function to handle both generation modes.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith(valid_exts)], key=natural_sort_key)
    target_files = files[start_idx:end_idx]
    
    print(f"[{mode.upper()}] Processing {len(target_files)} images...")

    # Pre-generate templates if in one_to_many mode
    effect_templates = []
    if mode == 'one_to_many':
        print(f"Generating {m_effects} fixed templates...")
        for _ in range(m_effects):
            p = get_random_typhoon_params()
            p['order_seed'] = random.random()
            effect_templates.append(p)

    for i, filename in enumerate(target_files):
        img_path = os.path.join(source_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image '{img_path}'. Skipping.")
            continue

        if mode == 'random_range':
            # 1-to-1: Unique random parameters per image
            params = get_random_typhoon_params()
            out_img = add_dynamic_typhoon_effect(img, params)
            save_name = f"{filename.split('.')[0]}.png"
            cv2.imwrite(os.path.join(target_folder, save_name), out_img)

        elif mode == 'one_to_many':
            # 1-to-M: Fixed templates applied to image
            for j, params in enumerate(effect_templates):
                out_img = add_dynamic_typhoon_effect(img.copy(), params)
                # Naming convention: index based (e.g., 100.png, 101.png...)
                save_idx = start_idx + i * m_effects + j + 1
                cv2.imwrite(os.path.join(target_folder, f"{save_idx}.png"), out_img)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(target_files)}")

if __name__ == "__main__":
    # Clean output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    set_global_seed(SEED)

    # Execution Logic
    if MODE == "mixed":
        # Example: First 75 images random, next 25 images using templates
        process_dataset(INPUT_DIR, OUTPUT_DIR, mode='random_range', start_idx=0, end_idx=75)
        # Note: adjust output filename logic in function if indices overlap matters
        process_dataset(INPUT_DIR, OUTPUT_DIR, mode='one_to_many', start_idx=75, end_idx=100, m_effects=5)
    
    elif MODE == "random_range":
        process_dataset(INPUT_DIR, OUTPUT_DIR, mode='random_range')
        
    elif MODE == "one_to_many":
        process_dataset(INPUT_DIR, OUTPUT_DIR, mode='one_to_many', m_effects=3)
        
    print("Dataset generation complete.")