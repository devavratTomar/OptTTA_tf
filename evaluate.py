import os
 

import numpy as np

from medpy.metric.binary import dc
import argparse
from networks.unet import UNet
from utils import *

from scipy.ndimage import zoom


def compute_vol_dice(pred_path, input_path, site):
    seg_names = sorted([f for f in os.listdir(input_path) if f.endswith('.npy') and 'mask' in f and site in f])
    pred_names = sorted([f for f in os.listdir(pred_path) if f.endswith('.npy') and site in f])

    patient_names = np.unique(['-'.join(f.split('-')[:2]) for f in seg_names])
    print(patient_names)

    dice_scores = {1: [], 2: []}

    for p in patient_names:
        slice_names = natural_sort([f for f in seg_names if p in f])
        
        seg_slices = np.stack([np.load(os.path.join(input_path, f)) for f in slice_names])
        seg_slices = zoom(seg_slices, (1, 256/seg_slices.shape[1], 256/seg_slices.shape[2]), order=0)
        pred_slices = np.load(os.path.join(pred_path, p))

        for c in [1, 2]:
            dice_scores[c].append(dc(pred_slices==c, seg_slices == c))

    for c in dice_scores:
        dice_scores[c] = [np.mean(dice_scores[c]), np.std(dice_scores[c])]

    print(dice_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--rootdir", type=str)
    parser.add_argument("--prediction_path", type=str)
    parser.add_argument("--target_site", type=str)

    args = parser.parse_args()

    compute_vol_dice(args.prediction_path, args.rootdir, args.target_site)
