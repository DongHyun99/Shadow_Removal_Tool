import cv2
import numpy as np
import os
from post_util import hist, match_histograms
from skimage import exposure

img = cv2.imread(os.path.join('save', 'baegot2 Color Constancy', 'img_inpainted_original.png'))
mask = cv2.imread(os.path.join('save', 'baegot2 Color Constancy', 'img_inpainted_mask.png'), cv2.IMREAD_GRAYSCALE)

#img = cv2.imread(os.path.join('save', 'baegot','img_edgeInpainted.png'))
#mask = cv2.imread(os.path.join('save', 'baegot', 'img_mask.png'), cv2.IMREAD_GRAYSCALE)

if img is not None and mask is not None:
    print('img & mask True')
else:
    print('img or mask is None')

mask[mask>0] = 255
mask = np.expand_dims(mask, axis=2)

masked = mask // 255 * img
none_masked = (255-mask) // 255 * img

masked = match_histograms(masked, none_masked, mask)
masked = mask // 255 * masked + none_masked

cv2.imwrite('post-processing.png', masked)
masked = masked.astype(np.uint8)

#hist([masked], mask = mask)
#hist([masked], mask = (255-mask))