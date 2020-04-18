from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


def Plt_Contrast(origin, fourier, iimage):
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(fourier, cmap='gray')
    plt.title('Fourier Image')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(iimage, cmap='gray')
    plt.title('Inverse Fourier Image')
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


def DFT_IDFT(image):
    F = DFT(image)
    fourier = 20 * np.log(1 + np.abs(F))
    fourier = fourier.astype(np.uint8)

    f = IDFT(F)
    iimage = np.clip(f, 0, 255)
    iimage = iimage.astype(np.uint8)

    Plt_Contrast(image, fourier, iimage)


def DFT_IDFT_Numpy(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fourier = 20 * np.log(np.abs(fshift))

    ishift = np.fft.ifftshift(fshift)
    iimage = np.fft.ifft2(ishift)
    iimage = np.abs(iimage)

    Plt_Contrast(image, fourier, iimage)


def DFT_IDFT_OpenCv(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)
    fourier = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))

    ishift = np.fft.ifftshift(dftShift)
    iimage = cv2.idft(ishift)
    iimage = cv2.magnitude(iimage[:, :, 0], iimage[:, :, 1])

    Plt_Contrast(image, fourier, iimage)


# read image to numpy array
image_path = "../image/"
image_name = "imori_128.jpg"
image = cv2.imread(image_path + image_name, cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32)

DFT_IDFT_Numpy(image)
DFT_IDFT_OpenCv(image)
DFT_IDFT(image)
