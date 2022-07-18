import zipfile
from urllib import response
from flask import Flask, request, jsonify
import os
import json
import glob
import cv2
import base64
import numpy as np
from distutils import util
import glob
import argparse

from detection import MTMT, MTMT_patch, SynShadow
from utils import pad_img_to_modulo
from mypredict import predict, predictWithDir
from calc_rmse_eval import main

rest_port = 50000

parser = argparse.ArgumentParser()
parser.add_argument('--use', type=bool, default=False)
opt = parser.parse_args()

app = Flask(__name__)

@app.route('/modelList', methods=['GET'])
def model_list():
    detection_model = list(filter(os.path.isdir,glob.glob('model/detection_models/*')))
    detection_model = list(map(lambda m : m.split('/')[-1], detection_model))
    
    removal_model = list(filter(os.path.isdir,glob.glob('model/removal_models/*')))
    removal_model = list(map(lambda m : m.split('/')[-1], removal_model))
    
    response = {'detection_model': detection_model, 'removal_model': removal_model}
    js = json.dumps(response, indent=4)
    return jsonify(js)

@app.route('/predict', methods=['POST'])
def prediction():
    
    shadow_dir = os.path.join('data', request.remote_addr, 'shadow')
    mask_dir = os.path.join('data', request.remote_addr, 'mask')
    inpainted_dir = os.path.join('data', request.remote_addr, 'result', 'inpainted')
    predicted_dir = os.path.join('data', request.remote_addr, 'result', 'predicted')
    GT_dir = os.path.join('data', request.remote_addr, 'GT')
    data_dir = os.path.join('data', request.remote_addr, 'data')
    
    os.makedirs(shadow_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(inpainted_dir, exist_ok=True)
    os.makedirs(predicted_dir, exist_ok=True)
    os.makedirs(GT_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    removal_model_dir = 'model/removal_models/'
    detection_model_dir = 'model/detection_models/'
    
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
    
    predictImage, inpaintingImage, perfValue = None, None, None
    
    if mode == 'Batch':
        if detection:
            pass
        
        else:
            maskImageZip = base64.b64decode(maskImage)
            
            with open(os.path.join('data',request.remote_addr,'mask.zip'), 'wb') as f:
                f.write(maskImageZip)
            
            maskImageZip = zipfile.ZipFile(os.path.join('data',request.remote_addr,'mask.zip'))
            
            for file in maskImageZip.namelist():
                maskImageZip.extract(file, mask_dir)
                
            maskImage = glob.glob(mask_dir+'/*.png')
            
            for mask in maskImage:
                maskim = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
                
                # dilate 값이 있는 경우 mask를 수정해줌
                if dilateValue != 'None':
                    kernel = np.ones((3,3), np.uint8)
                    maskim = cv2.dilate(maskim, kernel, int(dilateValue))
                cv2.imwrite(os.path.join(data_dir, mask[:-4].split('/')[-1]+'_mask'+'.png'), maskim)
                
        if removal:
            
            shadowImageZip = base64.b64decode(shadowImage)
            
            with open(os.path.join('data',request.remote_addr,'shadow.zip'), 'wb') as f:
                f.write(shadowImageZip)
            
            shadowImageZip = zipfile.ZipFile(os.path.join('data',request.remote_addr,'shadow.zip'))
            
            for file in shadowImageZip.namelist():
                shadowImageZip.extract(file, shadow_dir)
                
            shadowImage = glob.glob(shadow_dir+ '/*.png')
            
            for shadow in shadowImage:
                shadowim = cv2.imread(shadow)
                #shadowim = cv2.cvtColor(shadowim, cv2.COLOR_RGB2BGR)
                print(os.path.join(data_dir, shadow.split('/')[-1]))
                cv2.imwrite(os.path.join(data_dir, shadow.split('/')[-1]), shadowim)
                
            path = os.path.join(removal_model_dir, removalModel)
            predictWithDir(data_dir, None, path, False, '.png', opt)
            
        return 'complete'
    
    elif mode == 'Interactive':
        if detection:
            shadowImage = cv2.imdecode(np.frombuffer(base64.b64decode(shadowImage), dtype=np.uint8), flags=1)
            cv2.imwrite(os.path.join(shadow_dir, 'sample.png'), shadowImage)
            
            if detectionModel == 'MTMT':
                print('MTMT')
                maskImage = MTMT(detection_model_dir, detectionModel, shadow_dir, mask_dir)
                maskImage = np.array(np.where(maskImage > 10, 255, 0), np.uint8)
                cv2.imwrite(os.path.join(mask_dir, 'sample.png'), maskImage)
            
            if detectionModel == 'MTMT_patch':
                print('MTMT_patch')
                
            if detectionModel == 'SynShadow':
                print('SynShadow')
                maskImage = SynShadow(detection_model_dir, shadow_dir, mask_dir)[0]
                maskImage = np.array(np.where(maskImage > 10, 255, 0), np.uint8)
                maskImage = cv2.resize(maskImage, (shadowImage.shape[1], shadowImage.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(os.path.join(mask_dir, 'sample.png'), maskImage)
        
        else:
            maskImage = cv2.imdecode(np.frombuffer(base64.b64decode(maskImage), dtype=np.uint8), flags=1)
            cv2.imwrite(os.path.join(mask_dir, 'sample.png'), maskImage)
            mask = maskImage.transpose([2,0,1])[0]
            
            if dilateValue != 'None':
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.dilate(mask, kernel, int(dilateValue))
            
            mask = np.expand_dims(mask, 0)
            mask = mask.astype('float32') / 255
            mask = pad_img_to_modulo(mask, 8)
    
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