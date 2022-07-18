import glob
import cv2
import os

def crop(img_x_size, img_y_size, img_path, patch_x_size=640, patch_y_size=480, stride=0) :

    input_dir_image = img_path
    output_dir = os.path.join(img_path, 'patch')
    img_list = glob.glob(input_dir_image+'/*.png')
    
    if stride!=0:
        for imname in img_list:
            name = imname.split('/')[-1]
            img = cv2.imread(imname)
            for i in range(0,(img_x_size-patch_x_size)//stride+1,1):
                for k in range(0,(img_y_size-patch_y_size)//stride+1,1):
                    x = i*stride
                    y = k*stride
                    patch = img[y:y+patch_y_size, x:x+patch_x_size]
                    print('save '+output_dir+'/{}_{}_{}.png'.format(name[:-4],x, y))
                    cv2.imwrite(output_dir+'/{}_{}_{}.png'.format(name[:-4],x, y), patch)
    
        # Use if you want to include a truncated image
        for imname in img_list:
            name = imname.split('/')[-1]
            img = cv2.imread(imname)
    
            for i in range(0,(img_x_size-patch_x_size)//stride+1,1):
                x = i*stride
                patch = img[-patch_y_size:,x:x+patch_x_size]
                print('save '+output_dir+'/{}_{}_{}.png'.format(name[:-4],x, img_y_size-patch_y_size))
                cv2.imwrite(output_dir+'/{}_{}_{}.png'.format(name[:-4],x, img_y_size-patch_y_size), patch)
    
            for i in range(0,(img_y_size-patch_y_size)//stride+1,1):
                y = i*stride
                patch = img[y:y+patch_y_size, -patch_x_size:]
                print('save '+output_dir+'/{}_{}_{}.png'.format(name[:-4], img_x_size-patch_x_size, y))
                cv2.imwrite(output_dir+'/{}_{}_{}.png'.format(name[:-4], img_x_size-patch_x_size, y), patch)
    
            patch = img[-patch_y_size:, -patch_x_size:]
            print('save '+output_dir+'/{}_{}_{}.png'.format(name[:-4], img_x_size-patch_x_size, img_y_size-patch_y_size))
            cv2.imwrite(output_dir+'/{}_{}_{}.png'.format(name[:-4], img_x_size-patch_x_size, img_y_size-patch_y_size), patch)
    
    else:
        for imname in img_list:
            name = imname.split('/')[-1]
            img = cv2.imread(imname)
            for i in range(0,img_x_size//patch_x_size,1):
                for k in range(0,img_y_size//patch_y_size,1):
                    x = i*patch_x_size
                    y = k*patch_y_size
                    patch = img[y:y+patch_y_size, x:x+patch_x_size]
                    print('save '+output_dir+'/{}_{}_{}.png'.format(name[:-4],x, y))
                    cv2.imwrite(output_dir+'/{}_{}_{}.png'.format(name[:-4],x, y), patch)
    
        # Use if you want to include a truncated image
        for imname in img_list:
            name = imname.split('/')[-1]
            img = cv2.imread(imname)
    
            for i in range(0,img_x_size//patch_x_size,1):
                x = i*patch_x_size
                patch = img[-patch_y_size:,x:x+patch_x_size]
                print('save '+output_dir+'/{}_{}_{}.png'.format(name[:-4],x, img_y_size-patch_y_size))
                cv2.imwrite(output_dir+'/{}_{}_{}.png'.format(name[:-4],x, img_y_size-patch_y_size), patch)
    
            for i in range(0,img_y_size//patch_y_size,1):
                y = i*patch_y_size
                patch = img[y:y+patch_y_size, -patch_x_size:]
                print('save '+output_dir+'/{}_{}_{}.png'.format(name[:-4], img_x_size-patch_x_size, y))
                cv2.imwrite(output_dir+'/{}_{}_{}.png'.format(name[:-4], img_x_size-patch_x_size, y), patch)
    
            patch = img[-patch_y_size:, -patch_x_size:]
            print('save '+output_dir+'/{}_{}_{}.png'.format(name[:-4], img_x_size-patch_x_size, img_y_size-patch_y_size))
            cv2.imwrite(output_dir+'/{}_{}_{}.png'.format(name[:-4], img_x_size-patch_x_size, img_y_size-patch_y_size), patch)

