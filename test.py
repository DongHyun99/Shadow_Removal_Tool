from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from saicinpainting.training.data.masks import get_mask_generator

from omegaconf import OmegaConf
import yaml
import glob
import os
import cv2
import numpy as np

train_config_path = os.path.join('model', 'removal_models', 'Inpainting', 'config.yaml')
with open(train_config_path, 'r') as f:
    train_config = OmegaConf.create(yaml.safe_load(f))
    
indir = os.path.join('data', '129.254.82.87', 'shadow')
  
tf = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(brightness=1.0, contrast=1.0)])
mask_generator = get_mask_generator(kind="mixed", kwargs=train_config.data.train.mask_gen_kwargs)

class sampleDataset(Dataset):
    def __init__(self, indir, mask_generator, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '*.png'), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0
        
    def __len__(self):
        return len(self.in_files)
    
    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = np.transpose(img, (2, 0, 1))
        mask = np.uint8(self.mask_generator(img, iter_i=self.iter_i)*255)
        img = np.transpose(img, (1, 2, 0))
        masked_img = cv2.bitwise_and(img, img, mask=mask[0])
        #print('before transform', img)
        masked_img = self.transform(masked_img)
        mask = np.transpose(mask, (1, 2, 0))
        img = masked_img
        #print('after transform', img)
        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        self.iter_i += 1
        return dict(image=img,
                    mask=mask)
    
dataset = sampleDataset(indir, mask_generator, tf)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

batch = next(iter(dataloader))
img = batch['image'][0]
img = np.transpose(img, (1, 2, 0))
img = img.numpy()

cv2.imwrite('sample.png', img)