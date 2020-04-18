from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


def Plt_Contrast(origin, iimage_1, iimage_2, iimage_4, iimage_6, iimage_8):
    plt.subplot(231)
    plt.imshow(origin, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(iimage_1, cmap='gray')
    plt.title('D0 = 1')
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(iimage_2, cmap='gray')
    plt.title('D0 = 2')
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(iimage_4, cmap='gray')
    plt.title('D0 = 4')
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(iimage_6, cmap='gray')
    plt.title('D0 = 6')
    plt.axis('off')

    plt.subplot(236)
    plt.imshow(iimage_8, cmap='gray')
    plt.title('D0 = 8')
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


def HPK_OpenCv(r, c, D0, dftShift):
    dftShift[r - D0:r + D0, c - D0:c + D0] = 0
    ishift = np.fft.ifftshift(dftShift)
    iimage = cv2.idft(ishift)
    iimage = cv2.magnitude(iimage[:, :, 0], iimage[:, :, 1])
    return iimage


def HPF_OpenCv(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)

    R, C = image.shape
    r, c = int(R / 2), int(C / 2)

    Plt_Contrast(image, HPK_OpenCv(r, c, 1, dftShift), HPK_OpenCv(r, c, 2, dftShift),
                 HPK_OpenCv(r, c, 4, dftShift), HPK_OpenCv(r, c, 6, dftShift),
                 HPK_OpenCv(r, c, 8, dftShift))


# read image to numpy array
image_path = "../image/"
image_name = "lena.jpg"
image = cv2.imread(image_path + image_name, cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32)

HPF_OpenCv(image)
