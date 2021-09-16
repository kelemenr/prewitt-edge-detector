import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.util import img_as_float
from scipy import ndimage
import sys
import cv2
import os


def prewitt_filter(img):
    # Sobel kernels g_x, g_y
    g_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    g_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Convolve matrix with kernel g_x, g_y
    g_x = ndimage.filters.convolve(img, g_x)
    g_y = ndimage.filters.convolve(img, g_y)

    # Calculate magnitude g
    g = np.sqrt(g_x ** 2 + g_y ** 2)
    g = g / g.max() * 255

    # Calculate slope of the gradient (gradient's direction)
    theta = np.arctan2(g_y, g_x)

    return g, theta


def non_maximum_suppression(img, gradient):
    suppressed_image = np.zeros(img.shape)
    a = gradient * 180.0 / 3.14159  # convert to degrees
    # Replace negative values
    a[a < 0] += 180

    # Identify edge direction using the angle matrix
    for i in range(len(a[:, 0]) - 1):
        for j in range(len(a[0, :]) - 1):
            u, v = 255, 255
            if (0 <= a[i, j] < 22.5) or (157.5 <= a[i, j] <= 180):  # 0°
                u, v = img[i, j + 1], img[i, j - 1]
            elif 67.5 <= a[i, j] < 112.5:  # 90°
                u, v = img[i + 1, j], img[i - 1, j]
            elif 112.5 <= a[i, j] < 157.5:  # 135°
                u, v = img[i - 1, j - 1], img[i + 1, j + 1]
            elif 22.5 <= a[i, j] < 67.5:  # 45°
                u, v = img[i + 1, j - 1], img[i - 1, j + 1]

            # Check if the pixel in the direction has a higher \
            # intensity than the current pixel
            if (img[i, j] >= u) and (img[i, j] >= v):
                suppressed_image[i, j] = img[i, j]
            else:
                suppressed_image[i, j] = 0

    return suppressed_image


def get_edges(img):
    # 1. Emphasising edges
    gradient, theta = prewitt_filter(img)

    # 2. Edge thinning
    suppressed_image = non_maximum_suppression(gradient, theta)

    return gradient, theta, suppressed_image


def main():
    try:
        path = sys.argv[1]
        file_name = os.path.basename(path)
        image = io.imread(str(path), as_gray=True)
    except:
        raise SystemExit(f"Usage: {sys.argv[0]} <image_path>")

    image = img_as_float(image)

    grad, t, suppressed = get_edges(image)

    cv2.imwrite("output/prewitt_" + file_name, grad)
    cv2.imwrite("output/non_max_" + file_name, suppressed)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image', fontsize=16)

    ax2.imshow(grad, cmap='gray')
    ax2.set_title('Prewitt Filter', fontsize=16)

    ax3.imshow(suppressed, cmap='gray')
    ax3.set_title('Non-Maximum Suppression', fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()
