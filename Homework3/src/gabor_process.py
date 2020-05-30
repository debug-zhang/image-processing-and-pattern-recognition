import cv2
import numpy as np


def BGR_to_GRAY(img):
    gray = 0.229 * img[..., 2] + 0.587 * img[..., 1] + 0.114 * img[..., 0]
    return gray.astype(np.uint8)


def gabor_filtering(gray, K_size, Sigma, Gamma, Lambda, Psi, angle):
    height, width = gray.shape
    gray = np.pad(gray, (K_size // 2, K_size // 2), mode='edge')
    out = np.zeros((height, width), dtype=np.float32)

    d = K_size // 2
    filter = np.zeros((K_size, K_size), dtype=np.float32)
    for y in range(K_size):
        for x in range(K_size):
            A = angle / 180. * np.pi
            kernel_x = np.cos(A) * (x - d) + np.sin(A) * (y - d)
            kernel_y = -np.sin(A) * (x - d) + np.cos(A) * (y - d)
            filter[y, x] = np.exp(-(kernel_x ** 2 + (Gamma * kernel_y) ** 2) / (2 * Sigma ** 2)) * np.cos(
                2 * np.pi * kernel_x / Lambda + Psi)
    filter /= np.sum(np.abs(filter))

    for y in range(height):
        for x in range(width):
            out[y, x] = np.sum(gray[y: y + K_size, x: x + K_size] * filter)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def gabor_process(img):
    height, width, _ = img.shape
    gray = BGR_to_GRAY(img).astype(np.float32)
    A = [0, 45, 90, 135]
    out = np.zeros([height, width], dtype=np.float32)
    for a in A:
        out += gabor_filtering(gray, K_size=11, Sigma=1.6, Gamma=1.2, Lambda=3, Psi=0, angle=a)
    out = (out / out.max() * 255).astype(np.uint8)
    return out


if __name__ == '__main__':
    input_img = cv2.imread("../image/lena_std.tif").astype(np.float32)

    output_img = gabor_process(input_img)

    cv2.imwrite("../image/lena_gabor.jpg", output_img)
