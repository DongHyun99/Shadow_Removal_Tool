from cv2 import transform
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A

from saicinpainting.training.data.masks import get_mask_generator

from omegaconf import OmegaConf
import yaml
import glob
import os
import cv2
import numpy as np

white = np.ones((2000, 3000)) * 255

cv2.imwrite('mask.png', white)