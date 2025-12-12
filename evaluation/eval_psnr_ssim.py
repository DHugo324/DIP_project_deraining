import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_psnr_ssim(gt_dir, res_dir, suffix='.png'):
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*' + suffix)))
    assert gt_files, f'No images found in {gt_dir}'

    psnrs = []
    ssims = []

    for gt_path in gt_files:
        name = os.path.basename(gt_path)
        res_path = os.path.join(res_dir, name)

        if not os.path.exists(res_path):
            print(f'[Skip] {name}: result not found in {res_dir}')
            continue

        # 讀圖並正規化到 [0,1]
        gt  = imread(gt_path).astype(np.float32) / 255.0
        res = imread(res_path).astype(np.float32) / 255.0

        # 如果大小不一致，先做對齊（取共同最小範圍）
        H = min(gt.shape[0], res.shape[0])
        W = min(gt.shape[1], res.shape[1])
        gt  = gt[:H, :W, ...]
        res = res[:H, :W, ...]

        data_range = 1.0
        psnr = peak_signal_noise_ratio(gt, res, data_range=data_range)

        # skimage >= 0.19 用 channel_axis，舊版用 multichannel=True
        try:
            ssim = structural_similarity(gt, res, data_range=data_range, channel_axis=-1)
        except TypeError:
            ssim = structural_similarity(gt, res, data_range=data_range, multichannel=True)

        psnrs.append(psnr)
        ssims.append(ssim)

    print('----------------------------------------')
    print(f'GT dir   : {gt_dir}')
    print(f'Res dir  : {res_dir}')
    print(f'Images evaluated: {len(psnrs)}')
    print(f'Mean PSNR: {np.mean(psnrs):.4f} dB')
    print(f'Mean SSIM: {np.mean(ssims):.4f}')
    print('----------------------------------------')
    return psnrs, ssims

if __name__ == '__main__':
    # path setting
    gt_dir  = 'Rtest/target'
    defect_dir = 'Rtest/input'
    res_dir = 'Rresult'

    psnr_before, ssim_before = evaluate_psnr_ssim(gt_dir, defect_dir)
    psnr_after, ssim_after = evaluate_psnr_ssim(gt_dir, res_dir)

    # -------- 畫 PSNR / SSIM 分布 --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(psnr_before, bins=20, color='blue', alpha=0.6, label="before")
    axes[0].hist(psnr_after, bins=20, color='red', alpha=0.6, label="after")
    axes[0].set_title('PSNR Distribution')
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    axes[1].hist(ssim_before, bins=20, color='green', alpha=0.6, label="before")
    axes[1].hist(ssim_after, bins=20, color='red', alpha=0.6, label="after")
    axes[1].set_title('SSIM Distribution')
    axes[1].set_xlabel('SSIM')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    # -------------------------------------------
