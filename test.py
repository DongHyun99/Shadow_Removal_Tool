import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

alpha = -0.5

mask_path = os.path.join('data', 'patch', 'mask', '01_DSC02912_2400_2400.png')
img_path = os.path.join('data', 'patch', 'shadow_free', '01_DSC02912_3000_1600.png')

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(img_path)
img_mask = cv2.bitwise_and(img, img, mask=mask)
img_none_mask = cv2.bitwise_and(img, img, mask=255-mask)

img_mask = np.clip((1+alpha) * img_mask - 128 * alpha, 0, 255).astype(np.uint8)
# img_mask = cv2.subtract(img_mask, (120, 120, 120, 0))

mask = np.expand_dims(mask,axis=-1)
img_masked = img_mask*(mask/255) + img_none_mask

cv2.imwrite(os.path.join('mask.png'), mask)
cv2.imwrite(os.path.join('img.png'), img_masked)
