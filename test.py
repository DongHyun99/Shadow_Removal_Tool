from calc_rmse_eval import main
import os

indir = os.path.join('/data5/dh/ISTD_dataset/test/', 'test_B')
gtdir = os.path.join('/data5/dh/ISTD_dataset/test/', 'test_C_fixed_official')
outdir = os.path.join('result', 'sg-shadowNet-official', 'aistd')

value = main(indir, gtdir, outdir, True)