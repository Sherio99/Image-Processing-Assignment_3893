# 📸 Image Processing Assignment - Complete Tasks Overview

This project covers multiple fundamental image processing operations using Python and libraries like OpenCV, NumPy, and PIL. The tasks include:

- **Intensity Level Reduction (Quantization)**
- **Mean (Average) Filtering**
- **Image Rotation**
- **Spatial Resolution Reduction (Block Averaging)**

Each operation is performed on a color image, with visual comparison and output image saving.

---

## 📦 Registration Details

```
Name     : DIAS B.R.S.T  
Reg No.  : EG/2020/3893  

```
---

## 📦 Required Libraries

```python
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
```

---

## 📂 Folder Structure

```
project/
├── Data/
│   └── Input_Image.jpg
├── outputs/
│   ├── task1/   ← Quantized Images
│   ├── task2/   ← Mean Filtered Images
│   ├── task3/   ← Rotated Images
│   └── task4/   ← Block Averaged Images
```

---

## 🖼️ Task 1: Intensity Level Reduction (Quantization)

**Goal**: Reduce the number of intensity levels in a color image to powers of 2 (2, 4, ..., 256).

```python
def reduce_intensity_levels(img_array, levels):
    assert (levels & (levels - 1)) == 0 and levels <= 256
    factor = 256 // levels
    quantized_img = (img_array // factor) * factor
    return quantized_img.astype(np.uint8)
```

**Example**:

```python
desired_levels = 16
reduced_img_array = reduce_intensity_levels(img_array, desired_levels)
```

**Output**:  
Saved in: `outputs/task1/reduced_16_levels.jpg`

---

## 🧼 Task 2: Mean Filtering (Smoothing)

**Goal**: Smooth the image using average filter with kernels of 3×3, 10×10, and 20×20.

```python
def apply_average_filter(img_array, kernel_size):
    return cv2.blur(img_array, (kernel_size, kernel_size))
```

**Example**:

```python
kernel_sizes = [3, 10, 20]
for k in kernel_sizes:
    smooth_img = apply_average_filter(img_array, k)
```

**Output**:  
Saved as:
- `outputs/task2/average_3x3.jpg`
- `outputs/task2/average_10x10.jpg`
- `outputs/task2/average_20x20.jpg`

---

## 🔄 Task 3: Image Rotation

**Goal**: Rotate an image around its center by 45° and 90°.

```python
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
```

**Example**:

```python
rotated_45 = rotate_image(img, 45)
rotated_90 = rotate_image(img, 90)
```

**Output**:  
Saved as:
- `outputs/task3/rotated_45.jpg`
- `outputs/task3/rotated_90.jpg`

---

## 🟩 Task 4: Spatial Resolution Reduction (Block Averaging)

**Goal**: Apply non-overlapping block averaging to reduce resolution while preserving color information.

```python
def block_average_color(img_array, block_size):
    h, w, c = img_array.shape
    new_img = img_array.copy()
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = img_array[i:i+block_size, j:j+block_size, :]
            avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
            new_img[i:i+block_size, j:j+block_size, :] = avg_color
    return new_img
```

**Example**:

```python
block_sizes = [3, 5, 7]
for bsize in block_sizes:
    reduced = block_average_color(img_array, bsize)
```

**Output**:  
Saved as:
- `outputs/task4/block_3x3.jpg`
- `outputs/task4/block_5x5.jpg`
- `outputs/task4/block_7x7.jpg`

---

## ⚠️ Note on Path Syntax (Windows Users)

If you're using Windows-style paths, avoid syntax warnings like `\A` by using:

```python
# Use raw string
image_path = r'D:\ABC\Sem 7\Image Processing Assignment\Data\Input_Image.jpg'
# OR escape backslashes
image_path = 'D:\\ABC\\Sem 7\\Image Processing Assignment\\Data\\Input_Image.jpg'
```

---

## 📊 Summary Table

| Task     | Description                                  | Output Directory         |
|----------|----------------------------------------------|--------------------------|
| Task 1   | Intensity Quantization                       | `outputs/task1/`         |
| Task 2   | Mean Smoothing (3x3, 10x10, 20x20)            | `outputs/task2/`         |
| Task 3   | Image Rotation (45°, 90°)                    | `outputs/task3/`         |
| Task 4   | Block Averaging (3x3, 5x5, 7x7)               | `outputs/task4/`         |

---
