import cv2
import base64
import requests
import json
import numpy as np
import glob
import os

''''''
ip = 'http://129.254.81.18:50000/'

def shadowRemovalUpload():
        # 이미지 base64 인코딩 
        with open("shadow.zip", "rb") as f:
            bytes = f.read()
            shadowImage = base64.b64encode(bytes)
        with open("mask.zip", "rb") as f:
            bytes = f.read()
            maskImage = base64.b64encode(bytes)
        
        GTImage = 'None'
        dilateValue = 'None' # 9
        
        upload = {'mode': 'Batch', 'removal': True, 'detection': False, 'perf': False,
                  'removalModel': 'Inpainting', 'detectionModel': 'None', 
                  'shadowImage': shadowImage, 'maskImage': maskImage, 'GTImage': GTImage, 'dilateValue': dilateValue}
        
        # predict requests
        requests.post(ip + 'predict', data=upload)
        return
    
    
    
shadowRemovalUpload()


'''
# edge mask 제작 코드

def extractMaskEdge(mask):    # mask [:,:, 1]
        #mask_double =np.asarry(mask, dtype=np.double)
        gx, gy = np.gradient(mask)
        
        temp_edge = gy * gy + gx * gx
        
        dilateValue = 9        
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(temp_edge, kernel, dilateValue)
        print(f"Edge Dilate: {dilateValue}")
 
        temp_edge[temp_edge != 0.0] = 255.0
        temp_edge = np.asarray(temp_edge, dtype=np.uint8)
        
        return temp_edge

masks = glob.glob('data/ISTD/mask/*.png')

for mask in masks:
    img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    img = extractMaskEdge(img)
    cv2.imwrite(os.path.join('data', 'ISTD', 'edgeMask', mask.split('\\')[-1]), img)
    print(os.path.join('data', 'ISTD', 'edgeMask', mask.split('\\')[-1]))
'''