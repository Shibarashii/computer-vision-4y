import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walter.jpg")
image = cv2.imread(IMG)


def apply_blur(image, blur_type='box', kernel_size=25):
    if blur_type == 'box':
        return cv2.blur(image, (kernel_size, kernel_size))
    elif blur_type == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif blur_type == 'median':
        return cv2.medianBlur(image, kernel_size)
    else:
        raise ValueError(
            "Invalid blur type. Choose from 'box', 'gaussian', or 'median'.")


def apply_sharpening(image, sharpen_type='emboss'):
    if sharpen_type == 'emboss':
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        return cv2.filter2D(image, -1, kernel)
    else:
        raise ValueError("Currently only 'emboss' sharpening is supported.")


def apply_edge_detection(image, edge_type='sobel'):
    if edge_type == 'sobel':
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobel_x, sobel_y)
        # Convert to uint8 after clamping the values
        return np.uint8(np.clip(edges, 0, 255))
    elif edge_type == 'canny':
        return cv2.Canny(image, 100, 200)
    elif edge_type == 'gaussian':
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 100, 200)
        return edges
    else:
        raise ValueError(
            "Invalid edge detection type. Choose from 'sobel', 'canny', or 'gaussian'.")


def main():
    image = cv2.imread(IMG)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blurring filters
    box_blurred = apply_blur(image, 'box')  # Color blur
    gaussian_blurred = apply_blur(image, 'gaussian')  # Color blur
    median_blurred = apply_blur(image, 'median')  # Color blur

    # Apply sharpening filter
    embossed_image = apply_sharpening(image, 'emboss')  # Color sharpening

    # Apply edge detection filters
    sobel_edges = apply_edge_detection(image, 'sobel')  # Color edge detection
    canny_edges = apply_edge_detection(image, 'canny')  # Color edge detection
    gaussian_edges = apply_edge_detection(
        image, 'gaussian')  # Color edge detection

    # Plot the results
    plt.figure(figsize=(12, 12))

    # Original color image
    plt.subplot(3, 3, 1)
    # Convert from BGR to RGB for display
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # Blurred images
    plt.subplot(3, 3, 2)
    # Color version of box blur
    plt.imshow(cv2.cvtColor(box_blurred, cv2.COLOR_BGR2RGB))
    plt.title("Box Blur")
    plt.axis('off')

    plt.subplot(3, 3, 3)
    # Color version of gaussian blur
    plt.imshow(cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Blur")
    plt.axis('off')

    plt.subplot(3, 3, 4)
    # Color version of median blur
    plt.imshow(cv2.cvtColor(median_blurred, cv2.COLOR_BGR2RGB))
    plt.title("Median Blur")
    plt.axis('off')

    # Embossed image
    plt.subplot(3, 3, 5)
    # Color version of emboss sharpening
    plt.imshow(cv2.cvtColor(embossed_image, cv2.COLOR_BGR2RGB))
    plt.title("Embossed Sharpening")
    plt.axis('off')

    # Edge detection images
    plt.subplot(3, 3, 6)
    # Color version of sobel edge detection
    plt.imshow(cv2.cvtColor(sobel_edges, cv2.COLOR_BGR2RGB))
    plt.title("Sobel Edge Detection")
    plt.axis('off')

    plt.subplot(3, 3, 7)
    # Color version of canny edge detection
    plt.imshow(cv2.cvtColor(canny_edges, cv2.COLOR_BGR2RGB))
    plt.title("Canny Edge Detection")
    plt.axis('off')

    plt.subplot(3, 3, 8)
    # Color version of gaussian edge detection
    plt.imshow(cv2.cvtColor(gaussian_edges, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Edge Detection")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
