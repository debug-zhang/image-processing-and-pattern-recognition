from cv2 import cv2
import numpy as np


def BGR_to_GRAY(img):
    gray = 0.229 * img[..., 2] + 0.587 * img[..., 1] + 0.114 * img[..., 0]
    return gray.astype(np.uint8)


def dft(img):
    M, N = img.shape
    F = np.zeros((M, N), dtype=np.complex)
    X = np.tile(np.arange(N), (M, 1))
    Y = np.arange(M).repeat(N).reshape(M, -1)
    for u in range(M):
        for v in range(N):
            F[u, v] = np.sum(img * np.exp(-2j * np.pi * (u * X / M + v * Y / N)))
    return F


def idft(F):
    M, N = F.shape
    f = np.zeros((M, N), dtype=np.float32)
    U = np.tile(np.arange(N), (M, 1))
    V = np.arange(M).repeat(N).reshape(M, -1)

    for x in range(M):
        for y in range(N):
            f[x, y] = np.abs(np.sum(F * np.exp(2j * np.pi * (x * U / M + y * V / N))))
    f = f / (M * N)

    return f


def hpk(r, c, D0, dftShift):
    dftShift[r - D0:r + D0, c - D0:c + D0] = 0
    ishift = np.fft.ifftshift(dftShift)
    iimage = cv2.idft(ishift)
    iimage = cv2.magnitude(iimage[:, :, 0], iimage[:, :, 1])
    return iimage


def hpf(img):
    gray = BGR_to_GRAY(img)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)
    R, C = gray.shape
    r, c = int(R / 2), int(C / 2)
    out = hpk(r, c, 8, dftShift)
    out = (out / out.max() * 255).astype(np.uint8)
    return out


if __name__ == '__main__':
    input_img = cv2.imread("../image/lena_std.tif").astype(np.float32)

    output_img = hpf(input_img)

    cv2.imwrite("../image/lena_hpf.jpg", output_img)
