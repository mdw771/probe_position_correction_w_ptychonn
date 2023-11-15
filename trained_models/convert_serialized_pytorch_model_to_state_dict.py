"""
Converts a serialized PyTorch model to a state-dict model, the latter of which offers much better flexibility
when integrated into a larger codebase.

Since the serialized model is a pickle object that saves the exact namespace where the model is defined, the model
definition code must be reproduced in this file. Make sure the definition code is up-to-date.
"""
import os
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from numpy.fft import fftn, fftshift


class recon_model(nn.Module):
    def __init__(self):
        super(recon_model, self).__init__()

        nconv = 32
        self.encoder = nn.Sequential(
            # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            nn.Conv2d(in_channels=1, out_channels=nconv, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv, nconv, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(nconv, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(nconv * 2, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.decoder1 = nn.Sequential(

            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv * 4, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv * 2, 1, 3, stride=1, padding=(1, 1)),
            nn.Sigmoid()  # Amplitude model
        )

        self.decoder2 = nn.Sequential(

            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 4, nconv * 4, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv * 4, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(nconv * 2, nconv * 2, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv * 2, 1, 3, stride=1, padding=(1, 1)),
            nn.Tanh()  # Phase model
        )

    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        # Restore -pi to pi range
        ph = ph * np.pi  # Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp, ph


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', default='ptychoNN_CNN_encoder_decoder_pytorch.pth')
    args = parser.parse_args()

    model_path = args.model_path
    model = torch.load(model_path)
    summary(model, (1, 256, 256), device='cuda')
    output_fname = os.path.splitext(model_path)[0] + '_statedict.pth'
    torch.save(model.state_dict(), output_fname)
