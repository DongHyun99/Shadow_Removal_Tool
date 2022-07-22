import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


mask_path = os.path.join('data', 'test', 'mask.png')
nmask_path = 'nmask.png'
img_path= 'img_predicted.png'
seg_path = 'mask_seg.png'
nseg_path = 'nmask_seg.png'

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
nmask = cv2.imread(nmask_path, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(img_path)
test = cv2.imread(seg_path)
test2 = cv2.imread(nseg_path)

def mask_and():
    global img
    img = cv2.bitwise_and(img, img, mask=nmask)
    cv2.imwrite(nseg_path, img)

def normalize():
    channels = cv2.split(test2)
    colors = ('b', 'g', 'r')
    for (ch, color) in zip (channels, colors):
        hist = cv2.calcHist([ch], [0], None, [256], [1, 256])
        plt.plot(hist, color = color)
    plt.show()
    
# mask_and()    
normalize()
