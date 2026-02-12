import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#task 1
def sample_image(image, factor):
    height, width = image.shape[:2]
    return cv2.resize(image, (width // factor, height // factor), interpolation=cv2.INTER_NEAREST)

def quantize_image(image, levels):
    return (np.floor(image / (256 // levels)) * (256 // levels)).astype(np.uint8)

def plot_images(original, sampled, quantized):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(sampled, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(quantized, cmap='gray')
    plt.axis('off')
    plt.show()

original_image = cv2.imread('../images/lena_gray_256.tif', cv2.IMREAD_GRAYSCALE)
if original_image is not None:
    sampled = sample_image(original_image, 14)
    quantized = quantize_image(original_image, 9)
    plot_results(original_image, sampled, quantized)

#task 2
img1 = Image.open('../images/lena_gray_256.tif')
img2 = Image.open('../images/cameraman.tif')
resize = (400, 400)
img1 = img1.resize(resize, Image.Resampling.LANCZOS)
img2 = img2.resize(resize, Image.Resampling.LANCZOS)
im1arr = np.asarray(img1)
im2arr = np.asarray(img2)

Image.fromarray(cv2.subtract(im1arr, im2arr)).show()
Image.fromarray(cv2.add(im1arr, 175)).show()
Image.fromarray(cv2.bitwise_and(im1arr, cv2.bitwise_not(im2arr))).show()
Image.fromarray(cv2.bitwise_xor(im1arr, im2arr)).show()
Image.fromarray(cv2.bitwise_and(im1arr, im2arr)).show()
