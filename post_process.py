import cv2
from cv2 import dilate
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from skimage import exposure
import random

def hist(img, mask = None, ratio = [0,256]):

    hist = cv2.calcHist(img, [0], mask, [256], ratio)
    plt.plot(hist, color = 'b')
    
    hist = cv2.calcHist(img, [1], mask, [256], ratio)
    plt.plot(hist, color = 'g')
    
    hist = cv2.calcHist(img, [2], mask, [256], ratio)
    plt.plot(hist, color = 'r')
    
    plt.show()

img = cv2.imread(os.path.join('save', 'Color Constancy2', 'img_inapinted_original.png'))
mask = cv2.imread(os.path.join('save', 'Color Constancy2', 'img_inapinted_mask.png'), cv2.IMREAD_GRAYSCALE)

#img = cv2.imread(os.path.join('save', 'dron low resolution', 'img_original_inpainted.png'))
#mask = cv2.imread(os.path.join('save', 'dron low resolution', 'img_original_mask.png'), cv2.IMREAD_GRAYSCALE)


mask = np.expand_dims(mask, axis=2)

kernel = np.ones((3,3), np.uint8)
dilate_mask = cv2.dilate(mask, kernel,3 )
dilate_mask = np.expand_dims(dilate_mask, axis=2)
dilate_mask[dilate_mask>0] = 255

masked = img

#masked = exposure.adjust_sigmoid(masked)
#masked = exposure.adjust_log(masked, gain=0.8)
val = random.uniform(1,3)
img = exposure.adjust_sigmoid(img, cutoff=0.48, gain=val)
print(val)
masked = dilate_mask // 255 * masked
img = (255-dilate_mask) // 255 * img

masked = dilate_mask // 255 * masked + img
cv2.imwrite('masked4.png', masked)
#hist([img], mask = mask)
#hist([img], mask = (255-mask))