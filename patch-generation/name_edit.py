import glob
import os

dir = '/home/dh/Project/Shadow_Removal/'
input_dir_image = dir + 'data/original/mask/'

img_list = glob.glob(input_dir_image+'*.png')

for img in img_list:
    new_name = img.split('/')[-1]
    new_name = input_dir_image + '_'.join(new_name.split('_')[:2]) + '.png'
    os.rename(img, new_name)
    print('save ' + new_name)