from base64 import b64decode
import re
from urllib import response
from flask import Flask, request, jsonify
import os
import json
import glob
import cv2
import base64
import numpy as np
from distutils import util

from detection import MTMT
from utils import pad_img_to_modulo
from mypredict import predict

rest_port = 50000
image_path = '/data5/dh/UI'

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
    
    shadow_dir = 'data/shadow'
    mask_dir = 'data/mask'
    shadow_free_dir = 'data/result/'
    removal_model_dir = 'model/removal_models/'
    detection_model_dir = 'model/detection_models/'
    
    mode = request.form['mode']
    removal = util.strtobool(request.form['removal'])
    detection = util.strtobool(request.form['detection'])
    perfValue = util.strtobool(request.form['perf'])
    removalModel = request.form['removalModel']
    detectionModel = request.form['detectionModel']
    shadowImage = request.form['shadowImage']
    maskImage = request.form['maskImage']
    GTImage = request.form['GTImage']
    dilateValue = request.form['dilateValue']
    
    predictImage, inpaintingImage, perf = None, None, None
    
    
    if mode == 'Batch':
        pass
    
    elif mode == 'Interactive':
        if detection:
            if detectionModel == 'MTMT':
                print('MTMT')
                shadowImage = cv2.imdecode(np.frombuffer(base64.b64decode(shadowImage), dtype=np.uint8), flags=1)
                cv2.imwrite(os.path.join(shadow_dir, 'shadow.png'), shadowImage)
                maskImage = MTMT(detection_model_dir, detectionModel, shadow_dir, mask_dir)
                maskImage = np.array(np.where(maskImage > 10, 255, 0), np.uint8)
                cv2.imwrite(os.path.join(mask_dir, 'mask.png'), maskImage)
        
        else:
            maskImage = cv2.imdecode(np.frombuffer(base64.b64decode(maskImage), dtype=np.uint8), flags=1)
            cv2.imwrite(os.path.join(mask_dir, 'mask.png'), maskImage)
            mask = maskImage.transpose([2,0,1])[0]
            
            if dilateValue != 'None':
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.dilate(mask, kernel, int(dilateValue))
            
            mask = np.expand_dims(mask, 0)
            mask = mask.astype('float32') / 255
            mask = pad_img_to_modulo(mask, 8)
    
        if removal:
            shadowImage = cv2.imdecode(np.frombuffer(base64.b64decode(shadowImage), dtype=np.uint8), flags=1)
            if 'hsv' in removalModel:
                shadowImage = cv2.cvtColor(shadowImage, cv2.COLOR_BGR2HSV)
            else: shadowImage = cv2.cvtColor(shadowImage, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(os.path.join(shadow_dir, 'shadow.png'), shadowImage)
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
            
            cv2.imwrite(os.path.join(shadow_free_dir, 'result_inpainted.png'), img_inpainted)
            cv2.imwrite(os.path.join(shadow_free_dir, 'result_predict.png'), img_predicted)
            inpaintingImage = base64.b64encode(cv2.imencode('.png', img_inpainted)[1]).decode()
            predictImage = base64.b64encode(cv2.imencode('.png', img_predicted)[1]).decode()
            
        mask = base64.b64encode(cv2.imencode('.png', maskImage)[1]).decode()    
        response = {'predictImage': predictImage, 'inpaintingImage': inpaintingImage, 'maskImage': mask, 'perf': perf}
        js = json.dumps(response, indent=4)
        return jsonify(js)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=rest_port)