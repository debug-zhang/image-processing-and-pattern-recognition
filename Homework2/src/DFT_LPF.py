from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


def Plt_Contrast(origin, iimage_5, iimage_10, iimage_15, iimage_25, iimage_50):
    plt.subplot(231)
    plt.imshow(origin, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(iimage_5, cmap='gray')
    plt.title('D0 = 5')
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(iimage_10, cmap='gray')
    plt.title('D0 = 10')
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(iimage_15, cmap='gray')
    plt.title('D0 = 15')
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(iimage_25, cmap='gray')
    plt.title('D0 = 25')
    plt.axis('off')

    plt.subplot(236)
    plt.imshow(iimage_50, cmap='gray')
    plt.title('D0 = 50')
    plt.axis('off')

    plt.show()


def DFT(image):
    M, N = image.shape
    F = np.zeros((M, N), dtype=np.complex)
    X = np.tile(np.arange(N), (M, 1))
    Y = np.arange(M).repeat(N).reshape(M, -1)

    for u in range(M):
        for v in range(N):
            F[u, v] = np.sum(image * np.exp(-2j * np.pi * (u * X / M + v * Y / N)))

    return F


def IDFT(F):
    M, N = F.shape
    f = np.zeros((M, N), dtype=np.float32)
    U = np.tile(np.arange(N), (M, 1))
    V = np.arange(M).repeat(N).reshape(M, -1)

    for x in range(M):
        for y in range(N):
            f[x, y] = np.abs(np.sum(F * np.exp(2j * np.pi * (x * U / M + y * V / N))))
    f = f / (M * N)

    return f


def LPK_OpenCv(mask, r, c, D0, dftShift):
    mask[r - D0:r + D0, c - D0:c + D0] = 1
    fshift = dftShift * mask
    ishift = np.fft.ifftshift(fshift)
    iimage = cv2.idft(ishift)
    iimage = cv2.magnitude(iimage[:, :, 0], iimage[:, :, 1])
    return iimage


def LPF_OpenCv(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)

    R, C = image.shape
    mask = np.zeros((R, C, 2), np.uint8)
    r, c = int(R / 2), int(C / 2)

    Plt_Contrast(image, LPK_OpenCv(mask, r, c, 5, dftShift), LPK_OpenCv(mask, r, c, 10, dftShift),
                 LPK_OpenCv(mask, r, c, 15, dftShift), LPK_OpenCv(mask, r, c, 25, dftShift),
                 LPK_OpenCv(mask, r, c, 50, dftShift))


# read image to numpy array
image_path = "../image/"
image_name = "imori.jpg"
image = cv2.imread(image_path + image_name, cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32)

LPF_OpenCv(image)
