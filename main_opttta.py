import os

from trainers_policy.opttta import OptTTA
import argparse

def ensure_dirs(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if not os.path.exists(os.path.join(checkpoints_dir, 'visuals')):
        os.makedirs(os.path.join(checkpoints_dir, 'visuals'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'predictions')):
        os.makedirs(os.path.join(checkpoints_dir, 'predictions'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'uncertainties')):
        os.makedirs(os.path.join(checkpoints_dir, 'uncertainties'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source Free Adaptation to test time Image.')
    parser.add_argument('--checkpoints_opttta', type=str)
    parser.add_argument('--source_segmentor_path', type=str)
    parser.add_argument('--lr', default=0.01, type=float)

    ## Sizes
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--ncolor_channels', default=1, type=int)
    parser.add_argument('--n_classes', default=3, type=int)

    ## Datasets
    parser.add_argument('--rootdir', type=str)
    parser.add_argument('--target_sites')
   
    ## Networks
    parser.add_argument("--n_steps", default=1000, type=int)
    parser.add_argument("--alpha_1", default=0.1, type=float)
    parser.add_argument("--alpha_2", default=0.1, type=float)
    parser.add_argument("--n_augs", default=5, type=int)
    parser.add_argument("--k", default=128, type=int)

    opt = parser.parse_args()

    ensure_dirs(opt.checkpoints_opttta)
    trainer = OptTTA(opt)
    trainer.launch()