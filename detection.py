import os
import torch
import numpy as np
# from medpy import metric
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import os

from model.detection_models.MTMT.MTMT import build_model
from utils import crf_refine

def MTMT(model_path, model_name, img_path ='data/shadow/', save_path='data/mask/'):
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


def test_all_case(net, image_list, num_classes=1, save_result=True, test_save_path=None, trans_scale=416, GT_access=True):
    normal = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # SBU
    # normal = transforms.Normalize([0.517, 0.514, 0.492], [0.186, 0.173, 0.181]) # ISTD
    # normal = transforms.Normalize([0.723, 0.616, 0.569], [0.169, 0.177, 0.197]) # ISIC2017
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
        
        # if save_result:
            # Image.fromarray(prediction).save(os.path.join(test_save_path, img_name[:-4]+'.png'), "PNG")

    return prediction

from model.detection_models.MTMT_patch.crop import crop
from model.detection_models.MTMT_patch.merge import merge

def MTMT_patch(model_path, model_name, img_path, save_path, 
               img_x_size, img_y_size, patch_x_size=640, patch_y_size=480, stride=0):
    
    crop(img_x_size, img_y_size, img_path, patch_x_size=640, patch_y_size=480, stride=0)
    maskImages = MTMT(model_path, model_name, img_path, save_path)
    merge(img_x_size, img_y_size, img_path, save_path, patch_x_size, patch_y_size, stride)

from model.detection_models.SynShadow.options.test_options import TestOptions
from model.detection_models.SynShadow.models.detect_model import DetectModel
from model.detection_models.SynShadow.data.demo_dataset import DemoDataset

def SynShadow(model_path, img_path, save_path):
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