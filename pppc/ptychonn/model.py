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
    def __init__(self, n_levels=3, nconv=32):
        """
        Vanilla PtychoNN model with adjustable number of levels.

        :param n_levels: int. Rule of thumb: use 3 levels for 64x64 inputs, 4 for 128x128.
        :param nconv: int.
        """
        super(PtychoNNModel, self).__init__()
        self.n_levels = n_levels
        self.nconv = nconv

        down_blocks = []
        for level in range(self.n_levels):
            down_blocks += self.get_down_block(level)
        up_blocks_1 = []
        up_blocks_2 = []
        for level in range(self.n_levels - 1, -1, -1):
            up_blocks_1 += self.get_up_block(level)
            up_blocks_2 += self.get_up_block(level)
        self.encoder = nn.Sequential(
            # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            *down_blocks
        )

        self.decoder1 = nn.Sequential(
            *up_blocks_1,
            nn.Conv2d(self.nconv * 2, 1, 3, stride=1, padding=(1, 1)),
            nn.Sigmoid()  # Amplitude model
        )

        self.decoder2 = nn.Sequential(
            *up_blocks_2,
            nn.Conv2d(self.nconv * 2, 1, 3, stride=1, padding=(1, 1)),
            nn.Tanh()  # Phase model
        )

    def get_down_block(self, level):
        """
        Get a list of layers in a downsampling block.

        :param level: int. 0-based level index.
        :return: list[nn.Module].
        """
        num_in_channels = int(self.nconv * 2 ** (level - 1))
        num_out_channels = int(self.nconv * 2 ** level)
        if level == 0:
            num_in_channels = 1
        blocks = [
            nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels,
                      kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(num_out_channels, num_out_channels, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        ]
        return blocks

    def get_up_block(self, level):
        """
        Get a list of layers in a upsampling block.

        :param level: int. 0-based level index.
        :return: list[nn.Module].
        """
        if level == self.n_levels - 1:
            num_in_channels = self.nconv * 2 ** level
            num_out_channels = self.nconv * 2 ** level
        elif level == 0:
            num_in_channels = self.nconv * 2 ** (level + 1)
            num_out_channels = self.nconv * 2 ** (level + 1)
        else:
            num_in_channels = self.nconv * 2 ** (level + 1)
            num_out_channels = self.nconv * 2 ** level
        num_in_channels = int(num_in_channels)
        num_out_channels = int(num_out_channels)
        blocks = [
            nn.Conv2d(num_in_channels, num_out_channels, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(num_out_channels, num_out_channels, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        ]
        return blocks

    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        # Restore -pi to pi range
        ph = ph * np.pi  # Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp, ph


class PtychoNNPhaseOnlyModel(nn.Module):
    def __init__(self, nconv: int = 16):
        """
        Phase-only model of PtychoNN. Model graph replicated From Anahka's ONNX file.

        :param nconv: int.
        """
        super(PtychoNNPhaseOnlyModel, self).__init__()
        self.nconv = nconv
        self.encoder = nn.Sequential(
            # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            *self.down_block(1, self.nconv),
            *self.down_block(self.nconv, self.nconv * 2),
            *self.down_block(self.nconv * 2, self.nconv * 4),
            *self.down_block(self.nconv * 4, self.nconv * 8),
            # *self.down_block(self.nconv * 8, self.nconv * 16),
            # *self.down_block(self.nconv * 16, self.nconv * 32),
            # *self.down_block(self.nconv * 32, self.nconv * 32)
        )

        # phase model
        self.decoder2 = nn.Sequential(
            # *self.up_block(self.nconv * 32, self.nconv * 32),
            # *self.up_block(self.nconv * 32, self.nconv * 16),
            # *self.up_block(self.nconv * 16, self.nconv * 8),
            *self.up_block(self.nconv * 8, self.nconv * 8),
            *self.up_block(self.nconv * 8, self.nconv * 4),
            *self.up_block(self.nconv * 4, self.nconv * 2),
            *self.up_block(self.nconv * 2, self.nconv * 1),
            nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1, 1)),
            nn.Tanh()
        )

    def down_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        ]
        return block

    def up_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        ]
        return block

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x1 = self.encoder(x)
            ph = self.decoder2(x1)
            # Restore -pi to pi range
            ph = ph * np.pi  # Using tanh activation (-1 to 1) for phase so multiply by pi
        # Return ones for magnitude.
        return None, ph


class PtychoNNTransposedConvModel(nn.Module):
    def __init__(self, n_levels=3, nconv=32):
        """
        Vanilla PtychoNN model with adjustable number of levels.

        :param n_levels: int. Rule of thumb: use 3 levels for 64x64 inputs, 4 for 128x128.
        :param nconv: int.
        """
        super(PtychoNNTransposedConvModel, self).__init__()
        self.n_levels = n_levels
        self.nconv = nconv

        down_blocks = []
        for level in range(self.n_levels):
            down_blocks += self.get_down_block(level)
        up_blocks_1 = []
        up_blocks_2 = []
        for level in range(self.n_levels - 1, -1, -1):
            up_blocks_1 += self.get_up_block(level)
            up_blocks_2 += self.get_up_block(level)
        self.encoder = nn.Sequential(
            # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            *down_blocks
        )

        self.decoder1 = nn.Sequential(
            *up_blocks_1,
            nn.Conv2d(self.nconv * 2, 1, 3, stride=1, padding=(1, 1)),
            nn.Sigmoid()  # Amplitude model
        )

        self.decoder2 = nn.Sequential(
            *up_blocks_2,
            nn.Conv2d(self.nconv * 2, 1, 3, stride=1, padding=(1, 1)),
            nn.Tanh()  # Phase model
        )

    def get_down_block(self, level):
        """
        Get a list of layers in a downsampling block.

        :param level: int. 0-based level index.
        :return: list[nn.Module].
        """
        num_in_channels = int(self.nconv * 2 ** (level - 1))
        num_out_channels = int(self.nconv * 2 ** level)
        if level == 0:
            num_in_channels = 1
        blocks = [
            nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels,
                      kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(num_out_channels, num_out_channels, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        ]
        return blocks

    def get_up_block(self, level):
        """
        Get a list of layers in a upsampling block.

        :param level: int. 0-based level index.
        :return: list[nn.Module].
        """
        if level == self.n_levels - 1:
            num_in_channels = self.nconv * 2 ** level
            num_out_channels = self.nconv * 2 ** level
        elif level == 0:
            num_in_channels = self.nconv * 2 ** (level + 1)
            num_out_channels = self.nconv * 2 ** (level + 1)
        else:
            num_in_channels = self.nconv * 2 ** (level + 1)
            num_out_channels = self.nconv * 2 ** level
        num_in_channels = int(num_in_channels)
        num_out_channels = int(num_out_channels)
        blocks = [
            nn.Conv2d(num_in_channels, num_out_channels, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(num_out_channels, num_out_channels, 3, stride=2, padding=(1, 1),
                               output_padding=(1, 1)),
            nn.ReLU(),
        ]
        return blocks

    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        # Restore -pi to pi range
        ph = ph * np.pi  # Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp, ph
