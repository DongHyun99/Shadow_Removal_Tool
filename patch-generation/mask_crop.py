import glob
import cv2
import os

img_x_size = 6000
img_y_size = 4000
patch_x_size = 600
patch_y_size = 400
stride_x = patch_x_size
stride_y = patch_y_size
# stride를 없애고 싶으면, patch 사이즈와 같이 해주면 됨

input_dir_image = os.path.join('data', 'dron')

img_dir = os.path.join(input_dir_image, 'shadow')
mask_dir = os.path.join(input_dir_image, 'mask')
output_dir = os.path.join('data', 'patch')
shadow_ratio = 0.1

img_list = os.listdir(os.path.join(img_dir))
mask_list = os.listdir(os.path.join(mask_dir))
img_list = [img for img in img_list if img.endswith('.JPG')]
mask_list = [mask for mask in mask_list if mask.endswith('.png')]

os.makedirs(os.path.join(output_dir, 'shadow'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'shadow_free'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'free_mask'), exist_ok=True)

for imname, maskname in zip(img_list, mask_list):
    img = cv2.imread(os.path.join(img_dir, imname))
    mask = cv2.imread(os.path.join(mask_dir, maskname), cv2.IMREAD_GRAYSCALE)
    for i in range(0,(img_x_size-patch_x_size)//stride_x+1,1):
        for k in range(0,(img_y_size-patch_y_size)//stride_y+1,1):
            x = i*stride_x
            y = k*stride_y
            patch = img[y:y+patch_y_size, x:x+patch_x_size]
            mask_patch = mask[y:y+patch_y_size, x:x+patch_x_size]
            count_contour = cv2.countNonZero(mask_patch)
            if count_contour == 0:
                print('save '+os.path.join(output_dir, 'shadow_free', '{}_{}_{}.png'.format(imname[:-4],x, y)))
                cv2.imwrite(os.path.join(output_dir, 'shadow_free', '{}_{}_{}.png'.format(imname[:-4],x, y)), patch)
                cv2.imwrite(os.path.join(output_dir, 'free_mask', '{}_{}_{}.png'.format(imname[:-4],x, y)), mask_patch)
            elif count_contour/(patch_x_size*patch_y_size) < shadow_ratio:
                print('save '+os.path.join(output_dir, 'shadow_free', '{}_{}_{}.png'.format(imname[:-4],x, y)))
                cv2.imwrite(os.path.join(output_dir, 'shadow_free', '{}_{}_{}.png'.format(imname[:-4],x, y)), patch)
                cv2.imwrite(os.path.join(output_dir, 'free_mask', '{}_{}_{}.png'.format(imname[:-4],x, y)), mask_patch)
            else:
                print('save '+os.path.join(output_dir, 'shadow', '{}_{}_{}.png'.format(imname[:-4],x, y)))
                cv2.imwrite(os.path.join(output_dir, 'shadow', '{}_{}_{}.png'.format(imname[:-4],x, y)), patch)
                cv2.imwrite(os.path.join(output_dir, 'mask', '{}_{}_{}.png'.format(imname[:-4],x, y)), mask_patch)