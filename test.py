import cv2
import numpy as np
import os

'''
img_path = os.path.join('result', '01_DSC02276_predicted.png')
mask_path = os.path.join('result', '01_DSC02276_mask.png')

src = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = np.where(mask == 255, 1 , 0)

mask = np.array([mask, mask, mask]).transpose(1, 2, 0)


lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)

# l, a, b 채널 분리
l, a, b = cv2.split(lab)

# CLAHE 객체 생성
clahe = cv2.createCLAHE(clipLimit=1.0,tileGridSize=(4, 4))
# CLAHE 객체에 l 채널 입력하여 CLAHE가 적용된 l 채널 생성 
l = clahe.apply(l)

# l, a, b 채널 병합
lab = cv2.merge((l, a, b))
# lab 색공간 이미지를 bgr 색공간 이미지로 변환
cont_dst = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

img = mask * cont_dst + (1 - mask) * src

cv2.imwrite('test.png', img)
cv2.imwrite('cont.png', cont_dst)
'''

img = cv2.imread(os.path.join('data', 'dron_patch', '01_DSC02088_mask.png'), cv2.IMREAD_GRAYSCALE)
img = np.array(np.where(img > 10, 255, 0), np.uint8)
cv2.imwrite('img.png', img)