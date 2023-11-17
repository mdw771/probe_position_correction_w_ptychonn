import os

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

from pppc.util import class_timeit


class PtychoNNModel(nn.Module):
    def __init__(self):
        super(PtychoNNModel, self).__init__()

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
