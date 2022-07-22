import glob
import cv2
import os

img_x_size = 3000
img_y_size = 2000
patch_x_size = 1000
patch_y_size = 1000
stride = 0
dir = 'data/patch_test/'
input_dir_image = dir + 'original/mask/'
output_dir = dir + 'patch/mask/'
form = '.png'

img_list = os.listdir(input_dir_image)

if stride!=0:
    for imname in img_list:
        name = imname
        img = cv2.imread(os.path.join(input_dir_image, imname))
        
        for i in range(0,(img_x_size-patch_x_size)//stride+1,1):
            for k in range(0,(img_y_size-patch_y_size)//stride+1,1):
                x = i*stride
                y = k*stride
                patch = img[y:y+patch_y_size, x:x+patch_x_size]
                print('save '+output_dir+'{}_{}_{}{}'.format(name[:-4],x, y, form))
                cv2.imwrite(output_dir+'{}_{}_{}{}'.format(name[:-4],x, y, form), patch)

    # Use if you want to include a truncated image
    for imname in img_list:
        name = imname
        img = cv2.imread(os.path.join(input_dir_image, imname))

        for i in range(0,(img_x_size-patch_x_size)//stride+1,1):
            x = i*stride
            patch = img[-patch_y_size:,x:x+patch_x_size]
            print('save '+output_dir+'{}_{}_{}{}'.format(name[:-4],x, img_y_size-patch_y_size, form))
            cv2.imwrite(output_dir+'{}_{}_{}{}'.format(name[:-4],x, img_y_size-patch_y_size, form), patch)

        for i in range(0,(img_y_size-patch_y_size)//stride+1,1):
            y = i*stride
            patch = img[y:y+patch_y_size, -patch_x_size:]
            print('save '+output_dir+'{}_{}_{}{}'.format(name[:-4], img_x_size-patch_x_size, y, form))
            cv2.imwrite(output_dir+'{}_{}_{}{}'.format(name[:-4], img_x_size-patch_x_size, y, form), patch)

        patch = img[-patch_y_size:, -patch_x_size:]
        print('save '+output_dir+'{}_{}_{}{}'.format(name[:-4], img_x_size-patch_x_size, img_y_size-patch_y_size, form))
        cv2.imwrite(output_dir+'{}_{}_{}{}'.format(name[:-4], img_x_size-patch_x_size, img_y_size-patch_y_size, form), patch)

else:
    for imname in img_list:
        name = imname
        img = cv2.imread(os.path.join(input_dir_image, imname))
        for i in range(0,img_x_size//patch_x_size,1):
            for k in range(0,img_y_size//patch_y_size,1):
                x = i*patch_x_size
                y = k*patch_y_size
                patch = img[y:y+patch_y_size, x:x+patch_x_size]
                print('save '+output_dir+'{}_{}_{}{}'.format(name[:-4],x, y, form))
                cv2.imwrite(output_dir+'{}_{}_{}{}'.format(name[:-4],x, y, form), patch)

    # Use if you want to include a truncated image
    for imname in img_list:
        name = imname
        img = cv2.imread(os.path.join(input_dir_image, imname))

        for i in range(0,img_x_size//patch_x_size,1):
            x = i*patch_x_size
            patch = img[-patch_y_size:,x:x+patch_x_size]
            print('save '+output_dir+'{}_{}_{}{}'.format(name[:-4],x, img_y_size-patch_y_size, form))
            cv2.imwrite(output_dir+'{}_{}_{}{}'.format(name[:-4],x, img_y_size-patch_y_size, form), patch)

        for i in range(0,img_y_size//patch_y_size,1):
            y = i*patch_y_size
            patch = img[y:y+patch_y_size, -patch_x_size:]
            print('save '+output_dir+'{}_{}_{}{}'.format(name[:-4], img_x_size-patch_x_size, y, form))
            cv2.imwrite(output_dir+'{}_{}_{}{}'.format(name[:-4], img_x_size-patch_x_size, y, form), patch)

        patch = img[-patch_y_size:, -patch_x_size:]
        print('save '+output_dir+'{}_{}_{}{}'.format(name[:-4], img_x_size-patch_x_size, img_y_size-patch_y_size, form))
        cv2.imwrite(output_dir+'{}_{}_{}{}'.format(name[:-4], img_x_size-patch_x_size, img_y_size-patch_y_size, form), patch)

