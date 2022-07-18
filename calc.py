from calc_rmse_eval import main
import os

indir = os.path.join('data', 'mask')
gtdir = os.path.join('data', 'GT')
# outdir = os.path.join('data', 'shadow_removal_only', 'predicted')
# outdir = os.path.join('data', 'shadow_removal_only', 'inpainted')
# outdir = os.path.join('data', 'edge_detection', 'predicted')
# outdir = os.path.join('data', 'edge_detection', 'inpainted')
# outdir = os.path.join('data', 'edge_detection_and_inpainting', 'predicted')
# outdir = os.path.join('data', 'edge_detection_and_inpainting', 'inpainted')
# outdir = os.path.join('data', 'edge_detection_dilate', 'predicted')
# outdir = os.path.join('data', 'edge_detection_dilate', 'inpainted')
# outdir = os.path.join('data', 'edge_detection_and_inpainting_dilate', 'predicted')
outdir = os.path.join('data', 'edge_detection_and_inpainting_dilate', 'inpainted')

main(indir, gtdir, outdir, True)