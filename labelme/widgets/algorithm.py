import numpy as np
import cv2
from PIL import Image
from qtpy.QtGui import QPixmap, QImage

import numpy as np
import cv2

# ndarray转QPixmap（支持RGB/BGR）
def ndarray_to_pixmap(arr):
    if arr.ndim == 2:  # 灰度图处理
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 3:  # BGR转RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    height, width, _ = arr.shape
    bytes_per_line = 3 * width
    return QPixmap.fromImage(QImage(arr.data, width, height, bytes_per_line, QImage.Format_RGB888))

# QPixmap转ndarray（保留Alpha通道）
def pixmap_to_ndarray(pixmap):
    '''
    qimage = pixmap.toImage()
    ptr = qimage.constBits()
    arr = np.array(ptr).reshape(qimage.height(), qimage.width(), 4)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    '''
    img = Image.fromqpixmap(pixmap)  # 需Pillow库支持

    res = np.array(img)
    if res.shape[-1] == 3:
        return cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(res, cv2.COLOR_RGBA2GRAY)



def fft_smooth(img):
    fft = np.fft.fft2(img)
    fftshift = np.fft.fftshift(fft)
    width = fftshift.shape[1]
    height = fftshift.shape[0]
    dh = height*3//8
    dw = width*3//8
    fftshift[:dh] = 0
    fftshift[-dh:] = 0
    fftshift[:,:dw] = 0
    fftshift[:,-dw:] = 0

    fft = np.fft.ifftshift(fftshift)
    n_img = np.fft.ifft2(fft)
    n_img = np.abs(n_img)
    n_img = np.clip(n_img,0,255).astype(np.uint8)
    return n_img


def enhancement_gray_img(pixmap):
    img = pixmap_to_ndarray(pixmap)
    img = cv2.equalizeHist(img)
    img = fft_smooth(img)
    img = ndarray_to_pixmap(img)
    return img