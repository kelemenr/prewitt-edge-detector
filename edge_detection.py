import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from scipy import ndimage
from scipy.ndimage.filters import convolve
import os


def get_gaussian_kernel(size, sigma=1.0):
    size = int(size) // 2
    x, y = np.meshgrid(np.arange(-size, size + 1),
                       np.arange(-size, size + 1))

    # The equation for a Gaussian filter kernel
    n = 1 / (2.0 * np.pi * sigma ** 2)
    H = np.exp(-((x ** 2 + y ** 2) / (
            2.0 * sigma ** 2))) * n
    return H


def sobel_filter(img):
    # Sobel kernels Gx, Gy
    Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Convolve matrix with kernel Gx, Gy
    Gx = ndimage.filters.convolve(img, Gx)
    Gy = ndimage.filters.convolve(img, Gy)

    # Calculate magnitude G
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    G = G / G.max() * 255

    # Calculate slope of the gradient (gradient's direction)
    theta = np.arctan2(Gy, Gx)

    return G, theta


def non_maximum_suppression(img, gradient):
    suppressed_image = np.zeros(img.shape)
    A = gradient * 180.0 / 3.14159  # convert to degrees
    # Replace negative values
    A[A < 0] += 180

    # Identify edge direction using the angle matrix
    for i in range(len(A[:, 0]) - 1):
        for j in range(len(A[0, :]) - 1):
            u, v = 255, 255
            if (0 <= A[i, j] < 22.5) or (157.5 <= A[i, j] <= 180):  # 0°
                u, v = img[i, j + 1], img[i, j - 1]
            elif 67.5 <= A[i, j] < 112.5:  # 90°
                u, v = img[i + 1, j], img[i - 1, j]
            elif 112.5 <= A[i, j] < 157.5:  # 135°
                u, v = img[i - 1, j - 1], img[i + 1, j + 1]
            elif 22.5 <= A[i, j] < 67.5:  # 45°
                u, v = img[i + 1, j - 1], img[i - 1, j + 1]

            # Check if the pixel in the direction has a higher \
            # intensity than the current pixel
            if (img[i, j] >= u) and (img[i, j] >= v):
                suppressed_image[i, j] = img[i, j]
            else:
                suppressed_image[i, j] = 0

    return suppressed_image


def double_threshold(img, th1=0.05, th2=0.1):
    weak_pixels = np.array(25)
    strong_pixels = np.array(255)

    high_threshold = img.max() * th2
    low_threshold = high_threshold * th1

    thresholded_image = np.zeros(img.shape)
    thresholded_image[np.where(img >= high_threshold)] = strong_pixels
    thresholded_image[np.where((img <= high_threshold) & (
            img >= low_threshold))] = weak_pixels
    thresholded_image[np.where(img < low_threshold)] = 0

    return thresholded_image, weak_pixels, strong_pixels


def hysteresis(thresholded_image, weak_pixels, strong_pixels):
    hysteresis_image = np.copy(thresholded_image)
    for i in range(1, hysteresis_image.shape[0] - 1):
        for j in range(1, hysteresis_image.shape[1] - 1):
            # Check 8 connected neighborhood pixels (around the pixel: \
            # top, right, bottom,
            # left, top-right, top-left, bottom-right, bottom-left)
            if hysteresis_image[i, j] == weak_pixels:
                if ((hysteresis_image[i + 1, j - 1] == strong_pixels) or
                        (hysteresis_image[i + 1, j] == strong_pixels) or
                        (hysteresis_image[
                             i + 1, j + 1] == strong_pixels) or
                        (hysteresis_image[i, j - 1] == strong_pixels) or
                        (hysteresis_image[i, j + 1] == strong_pixels) or
                        (hysteresis_image[
                             i - 1, j - 1] == strong_pixels) or
                        (hysteresis_image[i - 1, j] == strong_pixels) or
                        (hysteresis_image[
                             i - 1, j + 1] == strong_pixels)):
                    hysteresis_image[i, j] = 255
                else:
                    hysteresis_image[i, j] = 0
    return hysteresis_image


def canny(img):
    # EDGE FILTERING
    # 1. Blur image with Gaussian filter
    blurred_image = convolve(img, get_gaussian_kernel(5, sigma=1.4))
    # 2. Emphasising edges
    gradient, theta = sobel_filter(blurred_image)

    # EDGE LOCALISATION
    # 3. Edge thinning
    suppressed_image = non_maximum_suppression(gradient, theta)
    # 4. Thresholding
    thresholded_image, weak_pixels, strong_pixels = double_threshold(
        suppressed_image, th1=0.08, th2=0.2)
    # 5. Edge tracking
    canny_edges = hysteresis(thresholded_image, weak_pixels,
                             strong_pixels)

    return (
        blurred_image, gradient, theta, suppressed_image,
        thresholded_image,
        weak_pixels, strong_pixels, canny_edges)


entries = os.listdir('test_images/')
entries = [str(i) for i in entries]

for image in entries:
    image = io.imread('test_images/' + image, as_gray=True)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.show()

    blurred, grad, t, suppressed, thresholded, \
        weak, strong, edges = canny(image)

    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    axs[0, 0].imshow(grad, cmap='gray')
    axs[0, 0].set_title('Prewitt Filter', fontsize=16)

    axs[0, 1].imshow(suppressed, cmap='gray')
    axs[0, 1].set_title('Non-Maximum Suppression', fontsize=16)

    axs[1, 0].imshow(thresholded, cmap='gray')
    axs[1, 0].set_title('Thresholded Image', fontsize=16)

    axs[1, 1].imshow(edges, cmap='gray')
    axs[1, 1].set_title('Canny (After Hysteresis)', fontsize=16)
    plt.show()
