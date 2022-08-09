import cv2
import os
import numpy as np

ratio = 0.5

img = cv2.imread(os.path.join('data', 'test', '01_DSC02735.JPG'))
img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LANCZOS4)

mask = cv2.imread(os.path.join('data', 'test', '01_DSC02735_mask.png'), cv2.IMREAD_GRAYSCALE)
mask = np.array(np.where(mask > 10, 255, 0), np.uint8)
mask = cv2.resize(mask, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LANCZOS4)

cv2.imwrite(os.path.join('data', 'test','mask.png'), mask)
cv2.imwrite(os.path.join('data', 'test','img.png'), img)