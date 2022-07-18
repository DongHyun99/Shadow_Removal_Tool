import cv2
import numpy as np
import glob

img_x_size = 6000
img_y_size = 4000
patch_x_size = 640
patch_y_size = 480
stride = 0
dir = '/home/dh/Project/Shadow_Removal/'
original_input = dir + 'data/original/input/'
patch_input = dir + 'data/patch_detection/img/'
merge_output = dir + 'data/patch_detection/merged_mask/'

img_list = glob.glob(original_input+'*.JPG')

for img_name in img_list:
    name = img_name.split('/')[-1][:-4]
    img = np.zeros((img_y_size, img_x_size), np.uint8)

    for i in range(0,img_x_size//patch_x_size,1):
        x = i*patch_x_size
        for k in range(0,img_y_size//patch_y_size,1):
            y = k*patch_y_size
            patch = cv2.imread(patch_input + '{}_{}_{}.png'.format(name, x, y), cv2.IMREAD_GRAYSCALE)
            patch = cv2.resize(patch, (patch_x_size, patch_y_size))
            img[y:y+patch_y_size, x:x+patch_x_size] = patch
        
        patch = cv2.imread(patch_input + '{}_{}_{}.png'.format(name, x, img_y_size-patch_y_size), cv2.IMREAD_GRAYSCALE)
        patch = cv2.resize(patch, (patch_x_size, patch_y_size))
        img[-patch_y_size:, x:x+patch_x_size] = patch

    for i in range(0,img_y_size//patch_y_size,1):
        y = i*patch_y_size
        patch = cv2.imread(patch_input + '{}_{}_{}.png'.format(name, img_x_size-patch_x_size, y), cv2.IMREAD_GRAYSCALE)
        patch = cv2.resize(patch, (patch_x_size, patch_y_size))
        img[y:y+patch_y_size, -patch_x_size:] = patch

    patch = cv2.imread(patch_input + '{}_{}_{}.png'.format(name, img_x_size-patch_x_size, img_y_size-patch_y_size), cv2.IMREAD_GRAYSCALE)
    patch = cv2.resize(patch, (patch_x_size, patch_y_size))
    img[-patch_y_size:, -patch_x_size:] = patch

    cv2.imwrite(merge_output+'{}.png'.format(name), img)
    print('save ' + merge_output + '{}.png'.format(name))