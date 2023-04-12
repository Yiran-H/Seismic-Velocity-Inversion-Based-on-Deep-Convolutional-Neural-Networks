import cv2 as cv
import os
import re
import sys
import time
import random

# 读取图像，支持 bmp、jpg、png、tiff 等常用格式

height = 0
length = 0

key = 0

picPath = 'split1.png'
if not os.path.exists(picPath):
    print("picture not exists! exit!")
    sys.exit()
srcImage = cv.imread(picPath)
if srcImage is None:
    print("read picture failed! exit!")
    sys.exit()
size = srcImage.shape

height = size[0] #40000
length = size[1] #2400
print("srcImage: height(%u) length(%u)" % (height, length))

# 创建窗口并显示图像
mid = int(height / 50 * 8)

up = srcImage[0 : mid, 0:length]
cv.imwrite('split1_8_1.png', up, [cv.IMWRITE_PNG_COMPRESSION, 1])

up = srcImage[mid : mid * 2, 0:length]
cv.imwrite('split8_16_1.png', up, [cv.IMWRITE_PNG_COMPRESSION, 1])

up = srcImage[mid * 2 : mid * 3, 0:length]
cv.imwrite('split16_24_1.png', up, [cv.IMWRITE_PNG_COMPRESSION, 1])

up = srcImage[mid * 3 : mid * 4, 0:length]
cv.imwrite('split24_32_1.png', up, [cv.IMWRITE_PNG_COMPRESSION, 1])

up = srcImage[mid * 4 : mid * 5, 0:length]
cv.imwrite('split32_40_1.png', up, [cv.IMWRITE_PNG_COMPRESSION, 1])