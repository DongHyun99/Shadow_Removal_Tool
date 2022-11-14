# -*- coding: utf-8 -*-

import argparse
import base64
from distutils import util
import glob
import json
import os
import shutil
from urllib import response
import zipfile

import cv2
from flask import Flask, request, jsonify
import numpy as np

from calc_rmse_eval import main
from detection import MTMT, MTMT_patch, SynShadow
from mypredict import predict, predictWithDir
from utils import pad_img_to_modulo

# ip:port 번호
rest_port = 50000

parser = argparse.ArgumentParser()
parser.add_argument('--use', type=bool, default=False)
opt = parser.parse_args()
app = Flask(__name__)

@app.route('/modelList', methods=['GET'])


def model_list():
    """detection model과 removal model의 종류를 디렉토리에서 읽어와 json 형태로 변환해 Rest-Api
    형태로 response함
    
    Args:

    Returns:
        js(string): detection model과 removal model의 list를 받아와 json 형태의 string으로 반환
    """
    
    # client가 새로 접속하면 (Tool을 시작하면) 이전에 사용했던 데이터 디렉토리 삭제
    ipData = os.path.join('data', request.remote_addr)
    if os.path.isdir(ipData):
        shutil.rmtree(ipData)
    
    # detection_model과 removal_model의 list를 json 형태로 response
    detection_model = list(filter(os.path.isdir,glob.glob('model/detection_models/*')))
    detection_model = list(map(lambda m : m.split('/')[-1], detection_model))
    removal_model = list(filter(os.path.isdir,glob.glob('model/removal_models/*')))
    removal_model = list(map(lambda m : m.split('/')[-1], removal_model))
    
    response = {'detection_model': detection_model, 'removal_model': removal_model}
    js = json.dumps(response, indent=4)
    return jsonify(js)

@app.route('/predict', methods=['POST'])


def prediction():
    """batch mode, interactive mode에 대한 shadow detection, removal, evaluation을 진행한 뒤 값을
    Rest-Api 형태로 response 함. mode, detection 여부, removal 여부 등을 request를 통해 받아옴

    Returns:
        js(string): image는 base64로 인코딩하고, evaluation 한 값을 포함한 json 형태의 string을 반환
    """
    
    # 각 ip별 디렉토리 할당
    shadow_dir = os.path.join('data', request.remote_addr, 'shadow')
    mask_dir = os.path.join('data', request.remote_addr, 'mask')
    inpainted_dir = os.path.join('data', request.remote_addr, 'result', 'inpainted')
    predicted_dir = os.path.join('data', request.remote_addr, 'result', 'predicted')
    GT_dir = os.path.join('data', request.remote_addr, 'GT')
    
    # 디렉토리 생성
    os.makedirs(shadow_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(inpainted_dir, exist_ok=True)
    os.makedirs(predicted_dir, exist_ok=True)
    os.makedirs(GT_dir, exist_ok=True)
    
    # detection & removal model list
    removal_model_dir = 'model/removal_models/'
    detection_model_dir = 'model/detection_models/'
    
    # detection이나 removal에 대한 request를 받아옴
    mode = request.form['mode']
    removal = util.strtobool(request.form['removal'])
    detection = util.strtobool(request.form['detection'])
    perf = util.strtobool(request.form['perf'])
    removalModel = request.form['removalModel']
    detectionModel = request.form['detectionModel']
    shadowImage = request.form['shadowImage']
    maskImage = request.form['maskImage']
    GTImage = request.form['GTImage']
    dilateValue = request.form['dilateValue']
    
    # result 값을 None으로 초기화
    predictImage, inpaintingImage, perfValue = None, None, None
    
    # Batch Mode
    if mode == 'Batch':
        
        print('Batch Mode Start')
        
        # base64로 인코딩 된 shadow image를 디코딩한 뒤 shadow_dir 디렉토리에 압축을 품
        shadowImageZip = base64.b64decode(shadowImage)
        
        with open(os.path.join('data',request.remote_addr,'shadow.zip'), 'wb') as f:
            f.write(shadowImageZip)
            
        shadowImageZip = zipfile.ZipFile(os.path.join('data',request.remote_addr,'shadow.zip'))
        for file in shadowImageZip.namelist():
            shadowImageZip.extract(file, shadow_dir)
            
        # shadow detection 시 
        if detection:
                
            shadowImages = os.listdir(shadow_dir)
            
            # detection model로 MTMT를 선택시 MTMT 메소드를 통해 그림자 검출을 진행
            if detectionModel == 'MTMT':
                print('MTMT')
                maskImages = MTMT(detection_model_dir, detectionModel, shadow_dir, mask_dir)
                for maskIm, shadow in zip(maskImages, shadowImages):
                    maskIm = np.array(np.where(maskIm > 10, 255, 0), np.uint8)
                    cv2.imwrite(os.path.join(mask_dir, shadow.replace('/shadow/', '/mask/')), maskIm)
        
        # detection을 하지 않는 경우 주어진 shadow mask를 사용함
        else:
            
            maskImageZip = base64.b64decode(maskImage)

            with open(os.path.join('data',request.remote_addr,'mask.zip'), 'wb') as f:
                f.write(maskImageZip)
            
            maskImageZip = zipfile.ZipFile(os.path.join('data',request.remote_addr,'mask.zip'))
            for file in maskImageZip.namelist():
                maskImageZip.extract(file, mask_dir)
                
            maskImages = glob.glob(mask_dir+'/*.png')
            
            # dilate 값이 주어진 경우 mask를 읽어와 dilation 후 다시 저장
            if dilateValue != 'None':
                print('dilate')
                for mask in maskImages:
                    maskim = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
                
                    # dilate 값이 있는 경우 mask를 수정해줌
                    kernel = np.ones((3,3), np.uint8)
                    maskim = cv2.dilate(maskim, kernel, iterations=int(dilateValue))
                    cv2.imwrite(mask, maskim)
        
        # Shadow Removal 시        
        path = os.path.join(removal_model_dir, removalModel)
        add_dir = os.path.join('data', request.remote_addr)
        predictWithDir(add_dir, None, path, False, '.png', opt)  
        dir = os.getcwd()
        
        # mask 압축 및 decoding
        maskList = os.listdir(mask_dir)
        os.chdir(mask_dir)
        maskZip = zipfile.ZipFile(os.path.join(dir, 'data', request.remote_addr, 'mask.zip'), 'w')
        for mask in maskList:
            maskZip.write(mask)
        maskZip.close()    
        os.chdir(dir)
        with open(os.path.join('data', request.remote_addr, 'mask.zip'), "rb") as f:
            bytes = f.read()
            maskZip = base64.b64encode(bytes).decode('UTF-8')
            
        # predict 압축 및 decoding
        predictList = os.listdir(predicted_dir)
        os.chdir(predicted_dir)
        predictZip = zipfile.ZipFile(os.path.join(dir, 'data', request.remote_addr, 'predicted.zip'), 'w')
        for pred in predictList:
            predictZip.write(pred)
        predictZip.close()    
        os.chdir(dir)
        with open(os.path.join('data', request.remote_addr, 'predicted.zip'), "rb") as f:
            bytes = f.read()
            predictZip = base64.b64encode(bytes).decode('UTF-8')
            
        # inpaint 압축 및 decoding
        inpaintList = os.listdir(inpainted_dir)
        os.chdir(inpainted_dir)
        inpaintZip = zipfile.ZipFile(os.path.join(dir, 'data', request.remote_addr, 'inpainted.zip'), 'w')
        for inp in inpaintList:
            inpaintZip.write(inp)
        inpaintZip.close()    
        os.chdir(dir)
        with open(os.path.join('data', request.remote_addr, 'inpainted.zip'), "rb") as f:
            bytes = f.read()
            inpaintZip = base64.b64encode(bytes).decode('UTF-8')
        
        # perf가 True일 경우 GT shadow-free image를 읽어와 RMSE performance를 계산하여 같이 반환
        if perf:
            print('RMSE calculate')
            GTZip = base64.b64decode(GTImage)
        
            with open(os.path.join('data',request.remote_addr,'GT.zip'), 'wb') as f:
                f.write(GTZip)

            GTZip = zipfile.ZipFile(os.path.join('data',request.remote_addr,'GT.zip'))
            for file in GTZip.namelist():
                GTZip.extract(file, GT_dir)

            #RMSE 계산
            inpainted_perfValue = main(mask_dir, GT_dir, inpainted_dir, True)
            predicted_perfValue = main(mask_dir, GT_dir, predicted_dir, True)
            
            perfValue = {'inpainted': inpainted_perfValue, 'predicted': predicted_perfValue}
        
        # response 정의 및 반환
        response = {'predictImage': predictZip, 'inpaintingImage': inpaintZip, 'maskImage': maskZip, 'perf': perfValue}
        js = json.dumps(response, indent=4)
        return jsonify(js)
    
    # Interactive Mode
    elif mode == 'Interactive':
        
        # detection method
        if detection:
            shadowImage = cv2.imdecode(np.frombuffer(base64.b64decode(shadowImage), dtype=np.uint8), flags=1)
            cv2.imwrite(os.path.join(shadow_dir, 'sample.png'), shadowImage)
            
            # MTMT / MTMT_patch / SynShadow 중 선택하여 mask 검출
            if detectionModel == 'MTMT':
                print('MTMT')
                maskImages = MTMT(detection_model_dir, detectionModel, shadow_dir, mask_dir)
                for maskImage in maskImages:
                    maskImage = np.array(np.where(maskImage > 10, 255, 0), np.uint8)
                    cv2.imwrite(os.path.join(mask_dir, 'sample.png'), maskImage)
            
            if detectionModel == 'MTMT_patch':
                print('MTMT_patch')
                maskImage = MTMT_patch(detection_model_dir, 'MTMT', shadow_dir, mask_dir, shadowImage.shape[1], shadowImage.shape[0])
                maskImage = np.array(np.where(maskImage > 10, 255, 0), np.uint8)
                cv2.imwrite(os.path.join(mask_dir, 'sample.png'), maskImage)
                
            if detectionModel == 'SynShadow':
                print('SynShadow')
                maskImage = SynShadow(detection_model_dir, shadow_dir, mask_dir)[0]
                maskImage = np.array(np.where(maskImage > 10, 255, 0), np.uint8)
                maskImage = cv2.resize(maskImage, (shadowImage.shape[1], shadowImage.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(os.path.join(mask_dir, 'sample.png'), maskImage)
        
        # detection이 아닐경우 mask를 받아옴
        else:
            maskImage = cv2.imdecode(np.frombuffer(base64.b64decode(maskImage), dtype=np.uint8), flags=1)
            cv2.imwrite(os.path.join(mask_dir, 'sample.png'), maskImage)
            mask = maskImage.transpose([2,0,1])[0]
            
            # dilate 값이 주어진 경우 mask를 읽어와 dilation 후 다시 저장
            if dilateValue != 'None':
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=int(dilateValue))
            
            mask = np.expand_dims(mask, 0)
            mask = mask.astype('float32') / 255
            mask = pad_img_to_modulo(mask, 8)

        # removal method
        if removal:
            shadowImage = cv2.imdecode(np.frombuffer(base64.b64decode(shadowImage), dtype=np.uint8), flags=1)
            cv2.imwrite(os.path.join(shadow_dir, 'sample.png'), shadowImage)
            
            if 'hsv' in removalModel:
                shadowImage = cv2.cvtColor(shadowImage, cv2.COLOR_BGR2HSV)
            else: shadowImage = cv2.cvtColor(shadowImage, cv2.COLOR_RGB2BGR)
            
            img = shadowImage.transpose([2,0,1])
            img = img.astype('float32') / 255
            img = pad_img_to_modulo(img, 8)
            
            data = dict()
            data['image'] = img
            if 'Ipainting' in removalModel:
                data['image'] = (1-mask) / 255 * img
            data['mask'] = mask
            data['shape'] = np.array(img.shape[1:])
    
            path = os.path.join(removal_model_dir, removalModel)
            img_inpainted, img_predicted, model = predict(data, path)
            
            if 'hsv' in removalModel:
                img_inpainted = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2HSV)
                img_predicted = cv2.cvtColor(img_predicted, cv2.COLOR_BGR2HSV)
                
            else:
                img_inpainted = cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR)
                img_predicted = cv2.cvtColor(img_predicted, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(os.path.join(inpainted_dir, 'sample.png'), img_inpainted)
            cv2.imwrite(os.path.join(predicted_dir, 'sample.png'), img_predicted)
            
            inpaintingImage = base64.b64encode(cv2.imencode('.png', img_inpainted)[1]).decode()
            predictImage = base64.b64encode(cv2.imencode('.png', img_predicted)[1]).decode()
            
            # perf가 True일 경우 RMSE performance를 계산하여 같이 반환
            if perf:
                print('RMSE calculate')
                GTImage = cv2.imdecode(np.frombuffer(base64.b64decode(GTImage), dtype=np.uint8), flags=1)
                cv2.imwrite(os.path.join(GT_dir, 'sample.png'), GTImage)

                inpainted_perfValue = main(mask_dir, GT_dir, inpainted_dir, True)
                predicted_perfValue = main(mask_dir, GT_dir, predicted_dir, True)
                
                perfValue = {'inpainted': inpainted_perfValue, 'predicted': predicted_perfValue}
            
        mask = base64.b64encode(cv2.imencode('.png', maskImage)[1]).decode()    
        response = {'predictImage': predictImage, 'inpaintingImage': inpaintingImage, 'maskImage': mask, 'perf': perfValue}
        js = json.dumps(response, indent=4)
        return jsonify(js)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=rest_port)