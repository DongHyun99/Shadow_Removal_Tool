import cv2
import numpy as np
import os

dat = os.path.join('data', 'drone_patch', '01_DSC02247.jpg')
print(os.path.basename(dat[:-4]))

'''
def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')

''''''
mask = cv2.imread('data/ISTD/mask/90-1.png', cv2.IMREAD_GRAYSCALE)
#mask = mask.astype('float32') / 255

kernel = np.ones((3,3), np.uint8)
mask = cv2.dilate(mask, kernel, int(3))

cv2.imwrite('name2.png', mask)

name = 'data/ISTD/mask/90-1.png'
print(name[:-4]+'_mask'+'.png')
'''