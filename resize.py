import cv2
import os
import numpy as np

img = cv2.imread(os.path.join('data', 'test', '01_DSC02371.JPG'))
img = cv2.resize(img, (3000, 2000), cv2.INTER_LANCZOS4)

mask = cv2.imread(os.path.join('data', 'test', '01_DSC02371_mask.png'), cv2.IMREAD_GRAYSCALE)
mask = np.array(np.where(mask > 10, 255, 0), np.uint8)
mask = cv2.resize(mask, (3000, 2000), cv2.INTER_LANCZOS4)

cv2.imwrite(os.path.join('data', 'test','img.png'), img)
cv2.imwrite(os.path.join('data', 'test','mask.png'), mask)