import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aso.jpg")
image = cv2.imread(IMG)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

equalized_gray = cv2.equalizeHist(gray)


yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
equalized_color = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


min_val = np.min(gray)
max_val = np.max(gray)
linear_contrast = ((gray - min_val) / (max_val - min_val)
                   * 255).astype(np.uint8)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_gray = clahe.apply(gray)


def plot_hist(image, title):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")


plt.figure(figsize=(15, 12))

# Original grayscale
plt.subplot(3, 3, 1), plt.imshow(
    gray, cmap="gray"), plt.title("Original Grayscale")
plt.subplot(3, 3, 2), plt.imshow(equalized_gray,
                                 cmap="gray"), plt.title("Equalized (Gray)")
plt.subplot(3, 3, 3), plt.imshow(
    equalized_color[:, :, ::-1]), plt.title("Equalized (Color - YUV)")

# Contrast adjustments
plt.subplot(3, 3, 4), plt.imshow(linear_contrast,
                                 cmap="gray"), plt.title("Linear Contrast Stretching")
plt.subplot(3, 3, 5), plt.imshow(clahe_gray, cmap="gray"), plt.title(
    "CLAHE (Adaptive Equalization)")

# Histograms
plt.subplot(3, 3, 6), plot_hist(gray, "Original Histogram")
plt.subplot(3, 3, 7), plot_hist(equalized_gray, "Equalized Histogram")
plt.subplot(3, 3, 8), plot_hist(clahe_gray, "CLAHE Histogram")

plt.tight_layout()
plt.show()
