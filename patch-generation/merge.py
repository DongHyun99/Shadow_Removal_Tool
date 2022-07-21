import cv2
import numpy as np
import glob
import os

img_x_size = 3000
img_y_size = 2000
patch_x_size = 600
patch_y_size = 400
stride = 0
dir = 'save/'
original_input = 'data/patch_test/original/shadow/'
patch_input = dir + 'edgeInpainted/'
merge_output = dir + ''
form ='.png'

img_list = os.listdir(original_input)

for img_name in img_list:
    name = img_name[:-4]
    img = np.zeros((img_y_size, img_x_size), np.uint8)
    img = np.array([img, img, img]).transpose(1, 2, 0)

    for i in range(0,img_x_size//patch_x_size,1):
        x = i*patch_x_size
        for k in range(0,img_y_size//patch_y_size,1):
            y = k*patch_y_size
            patch = cv2.imread(patch_input + '{}_{}_{}{}'.format(name, x, y, form))
            # patch = cv2.resize(patch, (patch_x_size, patch_y_size))
            img[y:y+patch_y_size, x:x+patch_x_size] = patch
        
        patch = cv2.imread(patch_input + '{}_{}_{}{}'.format(name, x, img_y_size-patch_y_size, form))
        # patch = cv2.resize(patch, (patch_x_size, patch_y_size))
        img[-patch_y_size:, x:x+patch_x_size] = patch

    for i in range(0,img_y_size//patch_y_size,1):
        y = i*patch_y_size
        patch = cv2.imread(patch_input + '{}_{}_{}{}'.format(name, img_x_size-patch_x_size, y, form))
        # patch = cv2.resize(patch, (patch_x_size, patch_y_size))
        img[y:y+patch_y_size, -patch_x_size:] = patch

    patch = cv2.imread(patch_input + '{}_{}_{}{}'.format(name, img_x_size-patch_x_size, img_y_size-patch_y_size, form))
    # patch = cv2.resize(patch, (patch_x_size, patch_y_size))
    img[-patch_y_size:, -patch_x_size:] = patch

    cv2.imwrite(merge_output+'{}{}'.format(name, form), img)
    print('save ' + merge_output + '{}{}'.format(name, form))