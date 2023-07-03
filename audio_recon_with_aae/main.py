import os
import argparse

from Nsynth_trainer import Nsynth_train
from Nsynth_reconstruct import Nsynth_recon

from FSD_trainer import FSD_train
from FSD_reconstruct import FSD_recon

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Adversarial AutoEncoder Models for Nsynth Dataset & FSD Dataset')

    parser.add_argument('--Nsynth', action="store_true", 
                        help='train Nsynth dataset')  

    parser.add_argument('--FSD', action="store_true",
                        help='train FSD dataset')

    parser.add_argument('--train', action="store_true",
                        help='train new model')

    parser.add_argument('--recon', action="store_true",
                        help='reconstruct the audio and spectrogram with trained model')
    args = parser.parse_args()

    if args.Nsynth:
        if args.train:
            Nsynth_train()
        if args.recon:
            Nsynth_recon()

    if args.FSD:
        if args.train:  
            FSD_train()
        if args.recon: 
            FSD_recon()
    
