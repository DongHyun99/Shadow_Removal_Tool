import os
import shutil

output_dir = 'patch-generation/output/'

if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)

os.mkdir(output_dir)
os.mkdir(output_dir+'mask')
os.mkdir(output_dir+'free_mask')
os.mkdir(output_dir+'shadow')
os.mkdir(output_dir+'shadow_free')