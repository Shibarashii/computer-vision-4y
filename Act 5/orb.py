import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Load images ---
img1_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "eiffel1.png")
img2_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "eiffel2.png")

# For feature extraction
img1_gray = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2_gray = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# For visualization (color)
img1_color = cv2.imread(img1_path, cv2.IMREAD_COLOR)
img2_color = cv2.imread(img2_path, cv2.IMREAD_COLOR)

# --- SIFT ---
sift = cv2.SIFT_create()
kp1_sift, des1_sift = sift.detectAndCompute(img1_gray, None)
kp2_sift, des2_sift = sift.detectAndCompute(img2_gray, None)

# Keypoints visualization
sift_kp_img1 = cv2.drawKeypoints(
    img1_color, kp1_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
sift_kp_img2 = cv2.drawKeypoints(
    img2_color, kp2_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Matching
bf_sift = cv2.BFMatcher()
matches_sift = bf_sift.knnMatch(des1_sift, des2_sift, k=2)
good_sift = [m for m, n in matches_sift if m.distance < 0.7 * n.distance]
img_sift_match = cv2.drawMatches(
    img1_color, kp1_sift, img2_color, kp2_sift, good_sift, None, flags=2)


# --- ORB ---
orb = cv2.ORB_create()
kp1_orb, des1_orb = orb.detectAndCompute(img1_gray, None)
kp2_orb, des2_orb = orb.detectAndCompute(img2_gray, None)

# Keypoints visualization
orb_kp_img1 = cv2.drawKeypoints(
    img1_color, kp1_orb, None, color=(0, 255, 0), flags=0)
orb_kp_img2 = cv2.drawKeypoints(
    img2_color, kp2_orb, None, color=(0, 255, 0), flags=0)

# Matching
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(des1_orb, des2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)
img_orb_match = cv2.drawMatches(
    img1_color, kp1_orb, img2_color, kp2_orb, matches_orb[:50], None, flags=2)

# --- Plot everything together ---
plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(sift_kp_img1, cv2.COLOR_BGR2RGB))
plt.title(f"SIFT Keypoints (Image 1: {len(kp1_sift)})")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(sift_kp_img2, cv2.COLOR_BGR2RGB))
plt.title(f"SIFT Keypoints (Image 2: {len(kp2_sift)})")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(orb_kp_img1, cv2.COLOR_BGR2RGB))
plt.title(f"ORB Keypoints (Image 1: {len(kp1_orb)})")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(orb_kp_img2, cv2.COLOR_BGR2RGB))
plt.title(f"ORB Keypoints (Image 2: {len(kp2_orb)})")

plt.tight_layout()
plt.show()

# --- Show matches separately ---
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_sift_match, cv2.COLOR_BGR2RGB))
plt.title(f"SIFT Matches ({len(good_sift)} good)")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_orb_match, cv2.COLOR_BGR2RGB))
plt.title(f"ORB Matches ({len(matches_orb[:50])} shown)")

plt.tight_layout()
plt.show()
