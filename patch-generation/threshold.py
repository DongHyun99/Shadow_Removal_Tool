import cv2
import glob
import numpy as np

dir = '/home/dh/Project/Shadow_Removal/'
merge_input = dir + 'data/patch_detection/merged_mask/'
threshold_output = dir + 'data/patch_detection/threshold_10/'

img_list = glob.glob(merge_input+'*.*')

for img_name in img_list:
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    img = np.array(np.where(img > 10, 255, 0), np.uint8)

    cv2.imwrite(threshold_output + img_name.split('/')[-1], img)