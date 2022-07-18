import glob
import cv2

img_x_size = 6000
img_y_size = 4000
patch_x_size = 1024
patch_y_size = 1024
stride = 512
input_dir_image = 'data/patch/'
output_dir = 'data/labeled/'
shadow_ratio = 0.1

img_list = glob.glob(input_dir_image+'image/'+'*.jpg')
mask_list = glob.glob(input_dir_image+'mask/'+'*.png')
for imname, maskname in zip(img_list, mask_list):
    img = cv2.imread(imname)
    mask = cv2.imread(maskname, cv2.IMREAD_GRAYSCALE)
    for i in range(0,(img_x_size-patch_x_size)//stride+1,1):
        for k in range(0,(img_y_size-patch_y_size)//stride+1,1):
            x = i*stride
            y = k*stride
            patch = img[y:y+patch_y_size, x:x+patch_x_size]
            mask_patch = mask[y:y+patch_y_size, x:x+patch_x_size]
            count_contour = cv2.countNonZero(mask_patch)
            if count_contour == 0:
                print('save '+output_dir+'shadow_free/'+'{}_{}_{}.jpg'.format(imname[29:-4],x, y))
                cv2.imwrite(output_dir+'shadow_free/'+'{}_{}_{}.jpg'.format(imname[29:-4],x, y), patch)
                cv2.imwrite(output_dir+'free_mask/'+'{}_{}_{}.jpg'.format(imname[29:-4],x, y), mask_patch)
            elif count_contour/(patch_x_size*patch_y_size) < shadow_ratio:
                print('save '+output_dir+'shadow_free/'+'{}_{}_{}.jpg'.format(imname[29:-4],x, y))
                cv2.imwrite(output_dir+'shadow_free/'+'{}_{}_{}.jpg'.format(imname[29:-4],x, y), patch)
                cv2.imwrite(output_dir+'free_mask/'+'{}_{}_{}.jpg'.format(imname[29:-4],x, y), mask_patch)
            else:
                print('save '+output_dir+'shadow/''{}_{}_{}.jpg'.format(imname[29:-4],x, y))
                cv2.imwrite(output_dir+'shadow/'+'{}_{}_{}.jpg'.format(imname[29:-4],x, y), patch)
                cv2.imwrite(output_dir+'mask/'+'{}_{}_{}.jpg'.format(imname[29:-4],x, y), mask_patch)