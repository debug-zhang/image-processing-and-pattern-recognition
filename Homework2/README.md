# 图像处理与模式识别

## 1 任务描述

- 实现一个数字图像的傅里叶变换和余弦变换:

## 2 实现环境

使用语言

- $Python 3.7.3$

$requirements.txt$：

```
numpy==1.16.1
opencv-python==4.1.1.26
```

## 3 目标图像

本次作业我选择了两张图像，一张是标准的 $Lena$ 人像，一张是蝾螈的动物图像：

 <center class="half">
 	<img src="../asserts/lena.jpg" width="300"/><img src="../asserts/imori.jpg" width="300"/>
</center> 



## 4 傅里叶变换

### 4.1 任务

对图像使用离散二维傅里叶变换，将灰度化的图像表示为频谱图，然后使用离散二维傅里叶逆变换将图形复原。

### 4.2 算法原理

傅里叶变换的物理意义是将图像的灰度分布变换为频率分布，而傅里叶的逆变换则是将图像的频率分布变换为灰度分布。

在处理实际问题时，信号往往都是离散的，例如图像就是一个二维离散信号，数字图像使用 $[0,255]$ 范围内的离散值表示，并且图像使用$M\times N$的二维矩阵表示，所以在这里使用离散二维傅里叶变换。因此，离散信号的傅里叶变换及其逆变换是非常具有现实意义的。

二维离散傅里叶变换使用下式计算，其中 $f$ 表示输入图像：
$$
F(u,v)=\sum\limits_{x=0}^{M-1}\ \sum\limits_{y=0}^{N-1}\ f(x,y)\ e^{-j\  2\  \pi\ (\frac{u\  x}{M}+\frac{v\  y}{N})}
$$
离散二维傅里叶逆变换按照下式定义：
$$
f(x,y)=\frac{1}{M\  N}\ \sum\limits_{u=0}^{M-1}\ \sum\limits_{v=0}^{N-1}\ F(u,v)\ e^{j\ 2\  \pi\ (\frac{u\  x}{M}+\frac{v\  y}{N})}
$$

### 4.3 代码实现

#### 4.3.1 Python 实现

手动实现过程使用 `opencv` 在读入图片时自动转为灰度图，然后按照公式完成离散二维傅里叶变换，将图片转换为频谱图，再完成逆变换。

由于函数是简单地使用 `for` 循环实现，计算量达到 $128^4$，这里我使用了矩阵计算，减少复杂度，但也达到了 $128^2$，非常耗费时间，参考意义不大。

```python
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

    for x in range (M):
        for y in range(N):
            f[x, y] = np.abs(np.sum(F * np.exp(2j * np.pi * (x * U / M + y * V / N))))
    f = f / (M * N)

    return f
```

#### 4.3.2 Numpy 实现

`numpy` 中的实现使用了快速傅里叶变换。

FFT 实际上一种分治算法，它将长度为 $N$ 的信号分解成两个长度为 $\frac{N}{2}$ 信号进行处理，这样分解一直到最后，每一次的分解都会减少计算的次数。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+X_k+%26%3D+%5Csum_%7Bn%3D0%7D%5E%7BN-1%7D+x_n+%5Ccdot+e%5E%7B-i~2%5Cpi~k~n~%2F~N%7D+%5C%5C++++++%26%3D+%5Csum_%7Bm%3D0%7D%5E%7BN%2F2+-+1%7D+x_%7B2m%7D+%5Ccdot+e%5E%7B-i~2%5Cpi~k~%282m%29~%2F~N%7D+%2B+%5Csum_%7Bm%3D0%7D%5E%7BN%2F2+-+1%7D+x_%7B2m+%2B+1%7D+%5Ccdot+e%5E%7B-i~2%5Cpi~k~%282m+%2B+1%29~%2F~N%7D+%5C%5C+%26%3D+%5Csum_%7Bm%3D0%7D%5E%7BN%2F2+-+1%7D+x_%7B2m%7D+%5Ccdot+e%5E%7B-i~2%5Cpi~k~m~%2F~%28N%2F2%29%7D+%2B+e%5E%7B-i~2%5Cpi~k~%2F~N%7D+%5Csum_%7Bm%3D0%7D%5E%7BN%2F2+-+1%7D+x_%7B2m+%2B+1%7D+%5Ccdot+e%5E%7B-i~2%5Cpi~k~m~%2F~%28N%2F2%29%7D+%5Cend%7Balign%7D+)

`numpy.fft.fft2(a, s=None, axes=(-2, -1), norm=None)`

`numpy.fft.ifft2(a, s=None, axes=(-2, -1), norm=None)`

```python
def DFT_IDFT_Numpy(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fourier = 20 * np.log(np.abs(fshift))

    ishift = np.fft.ifftshift(fshift)
    iimage = np.fft.ifft2(ishift)
    iimage = np.abs(iimage)
```

#### 4.3.3 OpenCv 实现

`opencv` 的实现和 `numpy` 基本一样，区别在于 `opencv` 的输出是双通道的，第一个通道是结果的实数部分，第二个通道是结果的虚数部分。

`cv2.dft(src, dst=None, flags=None, nonzeroRows=None)`

`idft(src, dst=None, flags=None, nonzeroRows=None)`

```python
def DFT_IDFT_OpenCv(image):
    dft = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)
    fourier = 20 * np.log(cv2.magnitude(dftShift[:,:,0], dftShift[:,:,1]))
    
    ishift = np.fft.ifftshift(dftShift)
    iimage = cv2.idft(ishift)
    iimage = cv2.magnitude(iimage[:,:,0], iimage[:,:,1])
```

### 4.4 实验总结

![](../asserts/DFT_lena.png)

![](../asserts/DFT_imori.png)

**傅里叶变换在图像处理中的应用基于两点：**

- 图片是波，这个波可以看成N个正弦，余弦波的叠加。
- 大自然中的各种信号的大部分信息都集中在低频，而且人眼对低频更敏感。

数字图像经过傅里叶变换后，图像的本质没有发生变换，只是经过了从欧几里得空间变到了信号空间，这样的好处就是我们可以看到图像的主要能量集中在什么地方，后续就可以有针对性的进行处理。

**亮线：**

- 在二维傅里叶变换中，空间域中横向的周期变化会反应在频谱图中的Y轴上，而空间域中纵向的周期变化会反应在频谱图中的X轴上。空间域中东南方向的周期变化会反应在频谱图中的东北方向，反之亦然。
- 在二维频谱图中的任意“一对亮点”，都在相应的空间域有一个与之相对应的二维正弦波。亮点在二维频谱中的位置决定了与之对应的正弦波的频率和方向。

**去噪：**

- 原理： 当图像出现的噪声是有规律的，相当于让某个频率波的幅度增大，把这个值减小，就是去掉这个频率的波，所以可以去噪，比如高斯噪声。
- 缺陷：当出现的噪声是没有规律的，随机出现的一些东西，FT 是没有作用的。

**图片压缩：**

- 根据傅立叶变换推导出的 DCT（下文包含） 有很重要作用。JPEG 格式的图片就是用 Huffman 编码方式压缩图片的 DCT 的系数。

## 5 低通滤波

### 5.1 任务

对图像使用离散二维傅里叶变换，并进行低通滤波，然后使用离散二维傅里叶逆变换将图形复原。

### 5.2 算法原理

通过离散傅立叶变换得到的频率在左上、右上、左下、右下等地方频率较低，在中心位置频率较高。

高频：

- 图像中灰度变化剧烈的点，一般是图像轮廓或者是噪声。

低频：

- 图像中平坦的，灰度变化不大的点，图像中的大部分区域。

低通滤波器：

- 只允许某一频率以下的信号无衰减地通过滤波器，其分界处的频率称为截止频率。

理想的低通滤波器模板为（高通滤波正好相反）：

![img](https://images2018.cnblogs.com/blog/1308890/201803/1308890-20180317213554048-2129235735.png)

其中，D0 表示通带半径，D(u，v) 是到频谱中心的距离（欧式距离），计算公式如下：

 ![img](https://images2018.cnblogs.com/blog/1308890/201803/1308890-20180317213607154-1019966770.png)

M 和 N 表示频谱图像的大小，（M/2，N/2）即为频谱中心.

### 5.3 代码实现

在 DFT 的基础上，增加 LPF 进行低通滤波，然后再进行 IDFT：

```python
R, C = image.shape
mask = np.zeros((R, C, 2), np.uint8)
r, c = int(R / 2), int(C / 2)
mask[r - D0:r + D0, c - D0:c + D0] = 1
fshift = dftShift * mask
```

### 5.4 实验总结

![](../asserts/DFT_LPF_lena.png)

![](../asserts/DFT_LPF_imori.png)

低通滤波使得图像的高频区域变成低频，即色彩变化剧烈的区域变得平滑，也就是出现模糊效果。由于理想低通滤波器的过度特性过于急峻，所以会产生了振铃现象。

高频的部分是图像的细节，如果低通滤波的截止频率越大，则保留的细节越多，相反截止频率越小，那么就会有更多的细节被过滤了，所以图像会更模糊，振铃现象更加明显。

## 6 高通滤波

### 6.1 任务

对图像使用离散二维傅里叶变换，并进行高通滤波，然后使用离散二维傅里叶逆变换将图形复原。

### 6.2 算法原理

通过离散傅立叶变换得到的频率在左上、右上、左下、右下等地方频率较低，在中心位置频率较高。

高频：

- 图像中灰度变化剧烈的点，一般是图像轮廓或者是噪声。

低频：

- 图像中平坦的，灰度变化不大的点，图像中的大部分区域。

高通滤波器：

- 只允许某一频率以上的信号无衰减地通过滤波器，去掉了信号中低于该频率的不必要的成分或者说去掉了低于该频率的干扰信号。

### 6.3 代码实现

在 DFT 的基础上，增加 HPF 进行高通滤波，然后再进行 IDFT：

```python
R, C = image.shape
r, c = int(R / 2), int(C / 2)
dftShift[r - D0:r + D0, c - D0:c + D0] = 0
```

### 6.4 实验总结

![](../asserts/DFT_HPF_lena.png)

![](../asserts/DFT_HPF_imori.png)

通过高通滤波，图像被过滤了低频，只保留那些变化最快速最剧烈的区域，也就是图像里面的物体边缘，所以常用于边缘识别。

从上图看出，在 D0 = 1 时就出现了明显的特征，这与低通滤波时完全不同，随着 D0 的增大，其边缘越来越明显。

## 7 离散余弦变换

### 7.1 任务

对图像使用离散余弦变换，将灰度化的图像表示为频谱图，然后使用离散余弦逆变换将图形复原。

### 7.2 算法原理

离散余弦变换用 DCT 表示，它是变换核为实数的余弦函数，因而 DCT 的计算速度比变换核为指数的 DFT 要快得多。

从傅里叶变换的性质可知，当一个函数为偶函数时，其傅里叶变换的虚部为零，因而不需要计算整个傅里叶变换，只需要计算余弦项变换，这就是余弦变换。因此余弦变换是傅里叶变换的特例，并且是简化傅里叶变换的一种重要方法。

DCT 变换的基本思想是将一个实函数对称延拓成一个实偶函数，实偶函数的傅里叶变换也必然是实偶函数。

离散余弦变换的变换式为：
$$
F(u,v)=\frac{2}{\sqrt{MN}}\  \sum\limits_{x=0}^{M-1}\ \sum\limits_{y=0}^{N-1}\ f(x,y)\  C(u)\  C(v)\ \cos(\frac{(2\  x+1)\  u\  \pi}{2\  M})\ \cos(\frac{(2\  y+1)\  v\  \pi}{2\  N})\\C(u)=\begin{cases}\frac{1}{\sqrt{2}}& u=0\\1&\text{其他}\end{cases}
$$
离散余弦逆变换使用下式定义：

$$
f(x,y)=\frac{2}{\sqrt{MN}}\  \sum\limits_{u=0}^{M-1}\ \sum\limits_{v=0}^{N-1}\ C(u)\  C(v)\ F(u,v)\ \cos(\frac{(2\  x+1)\  u\  \pi}{2\  M})\ \cos(\frac{(2\  y+1)\  v\  \pi}{2\  N})\\C(u)=\begin{cases}\frac{1}{\sqrt{2}}& u=0\\1&\text{其他}\end{cases}
$$

### 7.3 代码实现

#### 7.3.1 Python 实现

与 DFT 类似，手动实现过程使用 `opencv` 在读入图片时自动转为灰度图，然后按照公式完成离散二维余弦变换，将图片转换为频谱图，再完成逆变换。

```python
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

```

#### 7.3.2 OpenCv 实现

`dct(src, dst=None, flags=None)`

`idct(src, dst=None, flags=None)`

```python
def DCT_IDCT_OpenCv(image):
    dct = cv2.dct(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)

    cosine = np.log(np.abs(dct))

    iimage = cv2.idct(dct)
```

### 7.4 实验总结

![](../asserts/DCT_lena.png)

![](../asserts/DCT_imori.png)

离散余弦变换，尤其是二维离散余弦变换，经常在数字图像的处理中使用到，主要用于图像和视频的有损压缩。这是由于离散余弦变换具有很强的“能量集中”特性：

- 大多数的自然信号（包括声音和图像），其能量分布大多集中在离散余弦变换后的低频部分，且当信号具有接近马尔科夫过程的统计特性时，离散余弦变换的去相关性接近于 K-L变换，即具有最优的去相关性的性能。

- DCT 没有虚部，本质是傅里叶变换(无损) ---> 图像从空间域快速（没有虚部）的变换到了频率域。
- 从图中可以看到，左上角亮度高，即 Lena 的主要能量集中在左上角；而且，从其逆变换可看出，DCT 为无损变换。

与 DFT 对比：

- 对于比较平滑的图像/数据，DFT 变换数据集中在中间（低频信号区），DCT 变换数据集中在左上角，几乎无法看出 DCT 的优势在哪里。
- 但是，对于细节丰富的图像，DCT 变化后的数据很发散，DCT 变化后的数据仍然比较集中。如果同样从频率谱恢复原始图像，那么选用 DCT 更合理，因为 DCT 只需要存储更少的数据点。正是这个原因，使得 DCT 广泛地应用于图像压缩。

