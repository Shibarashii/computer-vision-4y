import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.feature import hog
from skimage import exposure

# Load image
IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiles.png")
img = cv2.imread(IMG)
if img is None:
    raise FileNotFoundError(f"Image not found at {IMG}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Harris Corner ---
harris_img = img.copy()
gray_float = np.float32(gray)
dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
harris_img[dst > 0.01 * dst.max()] = [0, 0, 255]

# --- SIFT ---
sift_img = img.copy()
sift = cv2.SIFT_create()
kp_sift = sift.detect(gray, None)
sift_img = cv2.drawKeypoints(sift_img, kp_sift, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# --- ORB ---
orb_img = img.copy()
orb = cv2.ORB_create()
kp_orb = orb.detect(gray, None)
orb_img = cv2.drawKeypoints(orb_img, kp_orb, None,
                            color=(0, 255, 0), flags=0)

# --- HOG (with visualization using scikit-image) ---
hog_features, hog_img = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=True
)

# Rescale intensity for better contrast
hog_img = exposure.rescale_intensity(hog_img, in_range=(0, 10))

# Convert BGR → RGB for matplotlib
harris_img = cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB)
sift_img = cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB)
orb_img = cv2.cvtColor(orb_img, cv2.COLOR_BGR2RGB)

# Plot results
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(harris_img)
axs[0].set_title("Harris Corners")
axs[1].imshow(sift_img)
axs[1].set_title("SIFT Keypoints")
axs[2].imshow(orb_img)
axs[2].set_title("ORB Keypoints")
axs[3].imshow(hog_img, cmap='gray')
axs[3].set_title("HOG Visualization")
for ax in axs:
    ax.axis("off")

plt.show()
