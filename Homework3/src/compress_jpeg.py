import cv2
import numpy as np


def BGR_to_YCbCr(img):
    height, width, channel = img.shape
    YCbCr = np.zeros([height, width, channel], dtype=np.float32)
    YCbCr[..., 0] = 0.257 * img[..., 2] + 0.504 * img[..., 1] + 0.098 * img[..., 0] + 16
    YCbCr[..., 1] = -0.148 * img[..., 2] - 0.291 * img[..., 1] + 0.439 * img[..., 0] + 128.
    YCbCr[..., 2] = 0.439 * img[..., 2] - 0.368 * img[..., 1] - 0.071 * img[..., 0] + 128.
    return YCbCr


def YCbCr_to_BGR(img):
    height, width, channel = img.shape
    BGR = np.zeros([height, width, channel], dtype=np.float32)
    BGR[..., 2] = 1.164 * (img[..., 0] - 16) + 1.596 * (img[..., 2] - 128.)
    BGR[..., 1] = 1.164 * (img[..., 0] - 16) - 0.392 * (img[..., 1] - 128.) - 0.813 * (img[..., 2] - 128.)
    BGR[..., 0] = 1.164 * (img[..., 0] - 16) + 2.017 * (img[..., 1] - 128.)
    return np.clip(BGR, 0, 255).astype(np.uint8)


def cos_core(x, y, u, v, T):
    cu = 1. / np.sqrt(2) if u == 0 else 1.
    cv = 1. / np.sqrt(2) if v == 0 else 1.
    result = 2 / T * cu * cv * np.cos((2 * x + 1) * u * np.pi / (2 * T)) * np.cos((2 * y + 1) * v * np.pi / (2 * T))
    return result


def dct(img, T=8):
    height, width, channel = img.shape
    F = np.zeros((height, width, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, height, T):
            for xi in range(0, width, T):
                for v in range(T):
                    for u in range(T):
                        for y in range(T):
                            for x in range(T):
                                F[v + yi, u + xi, c] += img[y + yi, x + xi, c] * cos_core(x, y, u, v, T)
    return F


def idct(F, T=8, K=8):
    height, width, channel = F.shape
    out = np.zeros((height, width, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, height, T):
            for xi in range(0, width, T):
                for y in range(T):
                    for x in range(T):
                        for v in range(K):
                            for u in range(K):
                                out[y + yi, x + xi, c] += F[v + yi, u + xi, c] * cos_core(x, y, u, v, T)
    return np.round(np.clip(out, 0, 255)).astype(np.uint8)


def quantization(F, T=8):
    Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
                  (12, 12, 14, 19, 26, 58, 60, 55),
                  (14, 13, 16, 24, 40, 57, 69, 56),
                  (14, 17, 22, 29, 51, 87, 80, 62),
                  (18, 22, 37, 56, 68, 109, 103, 77),
                  (24, 35, 55, 64, 81, 104, 113, 92),
                  (49, 64, 78, 87, 103, 121, 120, 101),
                  (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

    height, width, channel = F.shape
    for c in range(channel):
        for yi in range(0, height, T):
            for xi in range(0, width, T):
                F[yi: yi + T, xi: xi + T, c] = np.round(F[yi: yi + T, xi: xi + T, c] / Q) * Q
    return F


def jpeg_compress(img):
    YCbCr = BGR_to_YCbCr(img)
    F = dct(YCbCr)
    F = quantization(F)
    YCbCr = idct(F)
    compress = YCbCr_to_BGR(YCbCr)
    return compress


if __name__ == '__main__':
    input_img = cv2.imread("../image/lena_std.tif", flags=cv2.IMREAD_COLOR).astype(np.float32)

    output_img = jpeg_compress(input_img)

    cv2.imwrite("../image/lena_compress.jpg", output_img)
