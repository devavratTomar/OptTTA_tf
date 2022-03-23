from ast import parse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from trainers.source_trainer import SourceTrainer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str)
    parser.add_argument('--checkpoints_dir', type=str)
    parser.add_argument("--sites", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_classes", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--n_steps", type=int, default=251000)

    parser.add_argument("--save_freq", default=50000, type=int)
    parser.add_argument("--evaluation_freq", default=10000, type=int)
    parser.add_argument("--print_freq", default=480, type=int)
    parser.add_argument("--display_freq", default=10000, type=int)

    args = parser.parse_args()
    args.sites = args.sites.split(",")

    trainer = SourceTrainer(args)
    trainer.run()