# -*- coding: utf-8 -*-

import cv2
import glob
import numpy as np
import os


def merge(img_x_size, img_y_size, img_path, save_path, patch_x_size=640, patch_y_size=480, stride=0):
    """crop을 통해 나눠진 patch를 다시 붙이는 메소드.

    Args:
        img_x_size (int): 원본 이미지의 x축 크기.
        img_y_size (int): 원본 이미지의 y축 크기.
        img_path (str): 이미지의 디렉토리 위치.
        save_path (str): 저장 디렉토리 위치.
        patch_x_size (int, optional):치 이미지의 x축 크기. Defaults to 640.
        patch_y_size (int, optional): 패치 이미지의 y축 크기. Defaults to 480.
        stride (int, optional): patch끼리 겹치는 stride의 크기. Defaults to 0.
        
    Returns:
       img(numpy.array): 합쳐진 이미지
    """

    original_input = img_path
    
    patch_input = os.path.join(save_path, 'patch')
    merge_output = save_path

    img_list = glob.glob(original_input+'/*.png')

    for img_name in img_list:
        name = img_name.split('/')[-1][:-4]
        img = np.zeros((img_y_size, img_x_size), np.uint8)

        for i in range(0,img_x_size//patch_x_size,1):
            x = i*patch_x_size
            for k in range(0,img_y_size//patch_y_size,1):
                y = k*patch_y_size
                patch = cv2.imread(patch_input + '/{}_{}_{}.png'.format(name, x, y), cv2.IMREAD_GRAYSCALE)
                patch = cv2.resize(patch, (patch_x_size, patch_y_size))
                img[y:y+patch_y_size, x:x+patch_x_size] = patch

            patch = cv2.imread(patch_input + '/{}_{}_{}.png'.format(name, x, img_y_size-patch_y_size), cv2.IMREAD_GRAYSCALE)
            patch = cv2.resize(patch, (patch_x_size, patch_y_size))
            img[-patch_y_size:, x:x+patch_x_size] = patch

        for i in range(0,img_y_size//patch_y_size,1):
            y = i*patch_y_size
            patch = cv2.imread(patch_input + '/{}_{}_{}.png'.format(name, img_x_size-patch_x_size, y), cv2.IMREAD_GRAYSCALE)
            patch = cv2.resize(patch, (patch_x_size, patch_y_size))
            img[y:y+patch_y_size, -patch_x_size:] = patch

        patch = cv2.imread(patch_input + '/{}_{}_{}.png'.format(name, img_x_size-patch_x_size, img_y_size-patch_y_size), cv2.IMREAD_GRAYSCALE)
        patch = cv2.resize(patch, (patch_x_size, patch_y_size))
        img[-patch_y_size:, -patch_x_size:] = patch

        cv2.imwrite(merge_output+'/{}.png'.format(name), img)
        print('merging ' + merge_output + '/{}.png'.format(name))
        
        return img