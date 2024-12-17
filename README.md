# SVD Image Compression and Evaluation Framework

This project provides tools for **image compression** using different variants of Singular Value Decomposition (SVD):
1. **Truncated SVD** (TSVD)
2. **Compressed SVD**
3. **Randomized SVD** (RSVD), with optional power iterations to improve accuracy.

The code supports both **grayscale** and **colored** image compression while measuring computational time, Mean Squared Error (MSE), and compression ratios for different numbers of singular values.

## Table of Contents
1. [Setup and Installation](#setup-and-installation)
2. [Dependencies](#dependencies)
3. [Code Overview](#code-overview)
4. [How to Use](#how-to-use)
   - Grayscale Image Compression
   - Colored Image Compression
5. [Results and Outputs](#results-and-outputs)
6. [Evaluation Metrics](#evaluation-metrics)

---

## Setup and Installation
Clone the repository and install required Python libraries:
```bash
# Clone the repository
git clone https://github.com/your-repository/svd-compression.git
cd svd-compression

# Install dependencies
pip install -r requirements.txt
```

---

## Dependencies
The following libraries are required for the project:
```bash
numpy
matplotlib
pandas
opencv-python
Pillow
```
Install them with:
```bash
pip install numpy matplotlib pandas opencv-python Pillow
```

---

## Code Overview
The project consists of:

### 1. **`svd.py`**
This module implements different variants of SVD:
- `trunc_svd`: Truncated SVD for grayscale images
- `compressed_svd`: Compressed SVD for grayscale images
- `randomized_svd`: Randomized SVD with power iteration for grayscale images
- `trunc_svd_colored`: Truncated SVD for colored images
- `compressed_svd_colored`: Compressed SVD for colored images
- `randomized_svd_colored`: Randomized SVD with power iteration for colored images

### 2. **`evaluation.py`**
This module provides tools to evaluate the compression results:
- `measure_computational_time`: Measure execution time of compression algorithms
- `mse_frobenius`: Compute the Mean Squared Error (MSE) for grayscale images
- `mse_frobenius_colored`: Compute MSE for colored images
- `calculate_compression_ratio`: Calculate the compression ratio
- `save_compressed_image`: Save the compressed image to file

### 3. **Main Script**
The main script performs compression experiments for both **grayscale** and **colored** images, evaluates performance, and visualizes results.

---

## How to Use

### Grayscale Image Compression
The script compresses a grayscale image using **Truncated SVD**, **Compressed SVD**, and **Randomized SVD** (with varying power iterations). Results include:
- Computational time
- MSE error
- Compression ratio

#### Steps
1. Place your grayscale image in the working directory (e.g., `alone.jpg`).
2. Run the main script:
```python
python main.py
```
3. Outputs:
   - Compressed images saved under `image_results/`
   - Performance metrics saved under `evaluation_results/`
   - Plots generated for time taken, MSE, and compression ratios

#### Code Snippet (Grayscale)
```python
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import svd
evaluation

image = Image.open("alone.jpg")
gray_image = image.convert('L')
image_matrix = np.array(gray_image)/255

singular_values = [1, 5, 10, 50, 100, 200, 300, 400, 500]
ts_results = []
for k in singular_values:
    comp_time, comp_image = evaluation.measure_computational_time(svd.trunc_svd, image_matrix, k)
    mse = evaluation.mse_frobenius(image_matrix, comp_image)
    print(f"k={k}, Time={comp_time}, MSE={mse}")
```

### Colored Image Compression
The script compresses colored images using **Truncated SVD**, **Compressed SVD**, and **Randomized SVD** with power iterations.

#### Steps
1. Place your colored image in the working directory (e.g., `alone.jpg`).
2. Run the main script:
```python
python main.py
```
3. Outputs:
   - Compressed images saved under `image_results/`
   - Evaluation results stored under `evaluation_results/`
   - Plots for **computational time**, **MSE**, and **compression ratio**

#### Code Snippet (Colored)
```python
image_bgr = cv2.imread('alone.jpg')
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0

singular_values = [1, 5, 10, 50, 100, 200]
ts_results = []
for k in singular_values:
    comp_time, comp_image = evaluation.measure_computational_time(svd.trunc_svd_colored, image, k)
    mse = evaluation.mse_frobenius_colored(image, comp_image)
    print(f"k={k}, Time={comp_time}, MSE={mse}")
```

---

## Results and Outputs
### Generated Outputs
- **Compressed Images**: Stored in the `image_results/` directory
- **Evaluation Metrics**: Saved as CSV or plots in `evaluation_results/`
- **Plots**: Performance comparisons:
   - Computational Time vs Singular Values
   - MSE Error vs Singular Values
   - Compression Ratio vs Singular Values

### Example Plots
- **Computational Time**
- **MSE** (Mean Squared Error)
- **Compression Ratio**

---

## Evaluation Metrics
1. **Computational Time**: Measures how long it takes to perform the compression for different SVD variants.
2. **MSE (Mean Squared Error)**: Measures reconstruction error between the original and compressed images.
3. **Compression Ratio**: Compression efficiency based on the size of the original and compressed images.

---

## Example Outputs
### Console Log:
```plaintext
This is the start of the truncated svd
Running the truncated svd for 50 singular value
Time=0.123s, MSE=0.0023, Compression Ratio=0.25

This is the start of the randomized svd with q=2
Running for k=50 singular value
Time=0.089s, MSE=0.0021, Compression Ratio=0.22
```

### Output Directory Structure:
```
project/
|-- image_results/
|   |-- gray_alone.jpg
|   |-- tsvd_50.jpg
|   |-- rsvd_q=2_50.jpg
|-- evaluation_results/
|   |-- time_taken_plot.png
|   |-- mse_plot.png
|   |-- compression_ratio_plot.png
```

---

## Author
Desmond Kofi Boateng

---

## License
This project is licensed under the MIT License. Feel free to use and modify it!
