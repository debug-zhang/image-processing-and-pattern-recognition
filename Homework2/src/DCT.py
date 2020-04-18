from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


def Plt_Contrast(origin, cosine, iimage):
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(cosine, cmap='gray')
    plt.title('Cosine Image')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(iimage, cmap='gray')
    plt.title('Inverse Cosine Image')
    plt.axis('off')

    plt.show()


def Cos_Core(x, y, u, v, M, N):
    cu = 1. / np.sqrt(2) if u == 0 else 1.
    cv = 1. / np.sqrt(2) if v == 0 else 1.

    result = cu * cv * np.cos((2 * x + 1) * u * np.pi / (2 * M)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))
    return result


def DCT(image):
    M, N = image.shape
    F = np.zeros((M, N), dtype=np.complex)

    for u in range(M):
        for v in range(N):
            result = 0
            for x in range(M):
                for y in range(N):
                    result += image[x, y] * Cos_Core(x, y, u, v, M, N)
            F[u, v] = result * 2 / np.sqrt(M * N)
    return F


def IDCT(F):
    M, N = F.shape
    f = np.zeros((M, N), dtype=np.float32)

    for x in range(M):
        for y in range(N):
            result = 0
            for u in range(M):
                for v in range(N):
                    result += F[u, v] * Cos_Core(x, y, u, v, M, N)
            f[x, y] = np.abs(result * 2 / np.sqrt(M * N))
    return f


def DCT_IDCT(image):
    F = DCT(image)
    fourier = 20 * np.log(1 + np.abs(F))
    fourier = fourier.astype(np.uint8)

    f = IDCT(F)
    iimage = np.clip(f, 0, 255)
    iimage = iimage.astype(np.uint8)

    Plt_Contrast(image, fourier, iimage)


def DCT_IDCT_OpenCv(image):
    dct = cv2.dct(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

    cosine = np.log(np.abs(dct))

    iimage = cv2.idct(dct)

    Plt_Contrast(image, cosine, iimage)


# read image to numpy array
image_path = "../image/"
image_name = "lena.jpg"
image = cv2.imread(image_path + image_name, cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32)

DCT_IDCT_OpenCv(image)
DCT_IDCT(image)
