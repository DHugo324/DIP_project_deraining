# Synthetic Typhoon Dataset Generator

## 1. Overview
This project provides a Python-based pipeline to generate a **Synthetic Typhoon Dataset** from standard outdoor images. The goal is to facilitate robustness testing and domain adaptation for computer vision models in extreme weather conditions.

The generator simulates three core physical characteristics of a typhoon:
1.  **Luminance Drop:** Simulating overcast/dark sky conditions using Gamma correction.
2.  **Atmospheric Haze:** Simulating low visibility due to moisture.
3.  **Dynamic Rain Streaks:** Simulating heavy, slanted rain driven by strong winds using motion blur and rotation matrices.

## 2. Prerequisites

### System Requirements
* Python 3.8+
* OS: Windows / Linux / macOS

### Installation
Install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

Key dependencies: opencv-python, numpy

## 3. Usage

1.  **Prepare Data:** Place your clean source images in a folder (e.g., `dataset/test/norain`).
2.  **Configure:** Open `generate_typhoon.py` and modify the configuration section at the top:

    ```python
    INPUT_DIR = r"dataset/test/norain"
    OUTPUT_DIR = r"dataset/test/typhoon"
    MODE = "mixed"  # Options: 'random_range', 'one_to_many', 'mixed'
    ```

3.  **Run the Script:**
    ```bash
    python generate_typhoon.py
    ```

### Generation Modes
* **Random Range (`random_range`):** Applies unique, randomized weather parameters to each image. Best for training data diversity.
* **One-to-Many (`one_to_many`):** Generates fixed "Typhoon Templates" (e.g., specific wind speed/rain density) and applies them to multiple images. Best for consistency testing.

## 4. Hyperparameters

The simulation is controlled by the following randomization ranges to ensure physical realism:

| Parameter | Range / Value | Description |
| :--- | :--- | :--- |
| **Gamma** | `1.2` - `1.6` | Controls image darkness (higher = darker). |
| **Transmission ($t$)** | `0.5` - `0.85` | Atmospheric visibility (lower = more haze). |
| **Rain Density** | `0.95` - `0.99` | Threshold for generating rain drops. |
| **Wind Angle** | `120°` - `150°` | Rotational angle for rain streaks (simulating wind). |
| **Streak Length** | `20` - `50` pixels | Length of motion blur for rain streaks. |
| **Composition Order** | Random (50/50) | Swaps between *Rain-then-Haze* and *Haze-then-Rain*. |

## 5. Experiment Structure

The default configuration in `main` demonstrates a mixed generation strategy:

* **Part 1 (Indices 0-75):** Processed using **Random Mode** to create diverse training samples.
* **Part 2 (Indices 75-100):** Processed using **One-to-Many Mode** (5 variations per image) to create a set for stability testing.

The global random seed is set to `42` to ensure that the dataset generation is fully reproducible.