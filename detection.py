# -*- coding: utf-8 -*-

import cv2
import glob
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from utils import crf_refine

from model.detection_models.MTMT.MTMT import build_model
from model.detection_models.MTMT_patch.crop import crop
from model.detection_models.MTMT_patch.merge import merge
from model.detection_models.SynShadow.data.demo_dataset import DemoDataset
from model.detection_models.SynShadow.options.test_options import TestOptions
from model.detection_models.SynShadow.models.detect_model import DetectModel


def MTMT(model_path, model_name, img_path ='data/shadow/', save_path='data/mask/'):
    """MTMT를 통해 mask를 검출하여 반환함

    Args:
        model_path (str): MTMT model 위치
        model_name (str): MTMT model 위치
        img_path (str, optional): shadow image directory path. Defaults to 'data/shadow/'.
        save_path (str, optional): 검출된 mask의 save path. Defaults to 'data/mask/'.

    Returns:
        predict(list): 검출된 마스크를 list 형태로 만들어 return
    """
    
    snapshot_path = os.path.join(model_path, model_name, 'iter_10000.pth')
    image_path = img_path
    num_classes = 1
    scale = 416
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(image_path)) if f.endswith('.png')]
    data_path = [(os.path.join(image_path, img_name + '.png'),
             os.path.join(image_path, img_name + '.png'))
            for img_name in img_list]
    
    net = build_model('resnext101').cuda()
    net.load_state_dict(torch.load(snapshot_path))
    net.eval()
    predict = test_all_case(net, data_path, num_classes=num_classes,
                               save_result=True, test_save_path=save_path, trans_scale=scale)
    
    return predict


def test_all_case(net, image_list, num_classes=1, trans_scale=416):
    """MTMT의 dataset과 transforms를 정의하여 mask를 검출하는 method

    Args:
        net (torch.nn.module): resnext101 기반 MTMT model
        image_list (list): shadow image, gt image 쌍의 tupple들로 이루어진 list
        num_classes (int, optional): resnext의 output class 개수. Defaults to 1.
        trans_scale (int, optional): resize할 크기 정의. Defaults to 416.

    Returns:
        predictions(list): 검출된 마스크를 list 형태로 만들어 return
    """
    normal = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # SBU
    # normal = transforms.Normalize([0.517, 0.514, 0.492], [0.186, 0.173, 0.181]) # ISTD
    # normal = transforms.Normalize([0.723, 0.616, 0.569], [0.169, 0.177, 0.197]) # ISIC2017
    
    predictions = []
    
    img_transform = transforms.Compose([
        transforms.Resize((trans_scale, trans_scale)),
        transforms.ToTensor(),
        normal,
    ])
    to_pil = transforms.ToPILImage()
    for (img_path, target_path) in tqdm(image_list):
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img_var = img_transform(img).unsqueeze(0).cuda()
        # res = net(img_var).softmax(dim=1) # shape(1, 2, 416, 416)
        # prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()[1, :, :])))
        # up_edge, up_shadow, up_subitizing, up_shadow_final = net(img_var) #subitizing
        up_shadow_final = net(img_var)  # subiziting
        # up_sal_final = net(img_var)
        # up_shadow_final, up_subitizing = net(img_var)  # subiziting
        res = torch.sigmoid(up_shadow_final[-1][-1])
        ''' # original size crf
        prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
        # prediction = np.uint8(prediction>=100)*255
        prediction = crf_refine(np.array(img.convert('RGB')), prediction)
        '''
        # resized image crf
        prediction = np.array(to_pil(res.data.squeeze(0).cpu()))
        # prediction = np.uint8(prediction>=90)*255 # trick just for SBU
        prediction = crf_refine(np.array(img.convert('RGB').resize((trans_scale, trans_scale))), prediction)
        prediction = np.array(transforms.Resize((h, w))(Image.fromarray(prediction.astype('uint8')).convert('L')))
        predictions.append(prediction)
        
        # if save_result:
            # Image.fromarray(prediction).save(os.path.join(test_save_path, img_name[:-4]+'.png'), "PNG")

    return predictions

def MTMT_patch(model_path, model_name, img_path, save_path, 
               img_x_size, img_y_size, patch_x_size=640, patch_y_size=480, stride=0):
    """MTMT를 patch 단위로 쪼개서 각각에 대한 그림자를 검출한 뒤 합치는 메소드

    Args:
        model_path (str): MTMT model 위치.
        model_name (str): MTMT model 위치.
        img_path (str, optional): shadow image directory path. 
        save_path (str, optional): 검출된 mask의 save path .
        img_x_size (int): 입력 이미지의 x 크기.
        img_y_size (int): 입력 이미지의 y 크기.
        patch_x_size (int, optional): patch의 x 크기. Defaults to 640.
        patch_y_size (int, optional): patch의 y 크기. Defaults to 480.
        stride (int, optional): patch의 stride. Defaults to 0.

    Returns:
        maskImages(np.array): 각 patch에 대한 merge를 수행한 뒤 나온 그림자 검출 이미지를 반환.
    """
    
    shadowPatchPath = os.path.join(img_path, 'patch')
    maskPatchPath = os.path.join(save_path, 'patch')
    
    os.makedirs(shadowPatchPath, exist_ok=True)
    os.makedirs(maskPatchPath, exist_ok=True)
    
    crop(img_x_size, img_y_size, img_path, patch_x_size, patch_y_size, stride)
    
    snapshot_path = os.path.join(model_path, model_name, 'iter_10000.pth')
    image_path = img_path
    num_classes = 1
    scale = 416
    img_list = glob.glob(shadowPatchPath+'/*.png')
    data_path = [(img_name, img_name) for img_name in img_list]
    net = build_model('resnext101').cuda()
    net.load_state_dict(torch.load(snapshot_path))
    net.eval()
    maskImages = test_all_case(net, data_path, num_classes=num_classes,
                               save_result=True, test_save_path=save_path, trans_scale=scale)
    
    for mask, imname in zip(maskImages, img_list):
        imname = imname.replace(shadowPatchPath+'/', '')
        print(os.path.join(maskPatchPath, imname))
        mask = np.array(np.where(mask > 10, 255, 0), np.uint8)
        cv2.imwrite(os.path.join(maskPatchPath, imname), mask)
    
    maskImages = merge(img_x_size, img_y_size, img_path, save_path, patch_x_size, patch_y_size, stride)
    
    return maskImages


def SynShadow(model_path, img_path, save_path):
    """Synshadow로 augmentation한 DSDNet++ detection model을 사용해 그림자를 검출.

    Args:
        model_path (str): 모델의 위치.
        img_path (str): shadow 이미지의 위치.
        save_path (str): save 디렉토리의 위치.

    Returns:
        predict(list): 결과 그림자 검출 이미지를 list 형태로 만들어 return.
    """
    
    predict = []
    
    opt = TestOptions().parse()
    opt.checkpoints_dir = model_path + 'SynShadow/checkpoints' # 'model/detection_models/SynShadow/checkpoints'
    opt.dataset_root = img_path
    opt.results_dir = save_path
    
    dataset = DemoDataset()
    dataset.initialize(opt)
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads), drop_last=False)
    
    model = DetectModel(opt)
    model.eval()

    for i, data_i in enumerate(dataloader):
        generated = model(data_i, mode='inference')
        img_path = data_i['path'][0]
        attn_outputs = generated['attn']
        if isinstance(attn_outputs, list):
            attn_outputs = attn_outputs[-1]
        attn_outputs = torch.clamp(attn_outputs, min=0.0, max=1.0)
        
        output = attn_outputs.detach().cpu().numpy()
        output = np.transpose(output[0], (1, 2, 0))
        output = np.round(output * 255 / (output.max()-output.min()), -1)
        predict.append(output)
        
    return predict