# DIP Project: Deraining Enhancement for Typhoon Conditions

## Overview

This project focuses on enhancing images captured under typhoon-like weather conditions. In addition to heavy rain streaks, typhoons often introduce low illumination, atmospheric haze, and wind-driven rain, which jointly reduce image visibility and make restoration more difficult than ordinary single-image deraining.

Most existing deraining models are trained for moderate rainfall and may not generalize well to these extreme conditions. To address this problem, this project improves both the restoration model and the training data.

## Project Approach

The project is based on **MPRNet**, a multi-stage image restoration model. We integrate **SPA-Net's four-directional recurrent spatial attention mechanism** into **MPRNet's Supervised Attention Module (SAM)**.

The modified module uses a four-directional **Identity Recurrent Neural Network (IRNN)** to scan image features upward, downward, leftward, and rightward. This helps the model capture the continuity and directionality of rain streaks while retaining MPRNet's original multi-stage supervision structure.

## Synthetic Typhoon Dataset

Because public deraining datasets rarely contain real typhoon scenes, this project also provides a program for generating synthetic typhoon images from clean outdoor images.

The generation process simulates three major characteristics of typhoon weather:

- **Low illumination:** Gamma correction is used to simulate dark and overcast conditions.
- **Atmospheric haze:** Variable fog density is added to reproduce reduced visibility caused by moisture and light scattering.
- **Wind-driven rain:** Rain streaks are rotated and motion-blurred to simulate heavy rain affected by strong winds.

This customized dataset allows the model to learn image degradation patterns that approximate the selected visual characteristics of typhoon conditions.

## Experiments

The experiments compare four configurations to examine the effects of the improved spatial attention module and the Synthetic Typhoon Dataset:

| Model | Attention Module | Training Dataset |
|---|---|---|
| Model 0 | Original SAM | Original MPRNet Training Set |
| Model 1 | Original SAM | Synthetic Typhoon Training Set |
| Model 2 | Improved SAM | Original MPRNet Training Set |
| Model 3 | Improved SAM | Synthetic Typhoon Training Set |

All models were trained for **250 epochs** and evaluated using **PSNR** and **SSIM** on two test sets.

### Results on Rain100L

| Model | Mean PSNR (dB) | Mean SSIM |
|---|---:|---:|
| Model 0 | **29.2745** | **0.9041** |
| Model 1 | 14.6876 | 0.7217 |
| Model 2 | 29.0507 | 0.8992 |
| Model 3 | 12.3738 | 0.6074 |

On Rain100L, Model 0 achieves the highest average PSNR and SSIM. Model 2 produces results similar to Model 0, indicating that the improved SAM alone does not improve the average PSNR or SSIM on this test set. Models trained on the Synthetic Typhoon Training Set perform worse on Rain100L because of the differences between the training and test data characteristics.

### Results on the Synthetic Typhoon Test Set

| Model | Mean PSNR (dB) | Mean SSIM |
|---|---:|---:|
| Model 0 | 23.7074 | 0.8157 |
| Model 1 | 30.7584 | 0.9406 |
| Model 2 | 23.5406 | 0.7978 |
| Model 3 | **31.0129** | **0.9421** |

On the Synthetic Typhoon Test Set, Models 1 and 3 perform substantially better than Models 0 and 2. Because both models were trained on the Synthetic Typhoon Training Set, the results indicate that training-data adaptation is the primary factor behind the performance improvement. Model 3 slightly outperforms Model 1, suggesting that the improved SAM provides an additional but limited improvement on this test set.

Overall, the experiments reveal a clear domain trade-off: training on the Synthetic Typhoon Training Set improves performance on the Synthetic Typhoon Test Set but reduces performance on Rain100L.

## Repository Contents

- `MPRNet_improve/`: MPRNet implementation with the modified spatial attention module
- `dataset/`: Synthetic Typhoon Dataset generation program
- `evaluation/`: Image restoration evaluation and result analysis

## References

This project is based on the following works:

- **MPRNet:** Multi-Stage Progressive Image Restoration
- **SPA-Net:** Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset
