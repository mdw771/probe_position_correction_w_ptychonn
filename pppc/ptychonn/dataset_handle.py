import h5py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import resize

from pppc.helper import transform_data_for_ptychonn


class HDF5Dataset(Dataset):
    def __init__(self, filename, verbose=False):
        self.filename = filename
        self.f = h5py.File(self.filename, 'r')
        self.verbose = verbose
        self.check_dataset()

    def __len__(self):
        return self.f['data/real'].shape[0]

    def __getitem__(self, idx):
        if self.verbose:
            print('Retrieving index: {}'.format(idx))
        real = self.f['data/real'][idx, :, :]
        real_phase = np.angle(real)
        real_mag = np.abs(real)
        dp = self.f['data/reciprocal'][idx, :, :]
        dp, real_mag, real_phase = self.process_data(dp, real_mag, real_phase)
        return dp, real_mag, real_phase

    def __getitems__(self, idx_list):
        idx_list = tuple(np.sort(idx_list))
        if self.verbose:
            print('Retrieving indices: {}'.format(idx_list))
        real = self.f['data/real'][idx_list, :, :]
        real_phase = np.angle(real)
        real_mag = np.abs(real)
        dp = self.f['data/reciprocal'][idx_list, :, :]
        dp, real_mag, real_phase = self.process_data(dp, real_mag, real_phase)
        return dp, real_mag, real_phase

    def process_data(self, dp, real_mag, real_phase):
        if not np.array_equal(dp.shape[-2:], real_phase.shape[-2:]):
            dp = self.resize_dp(dp, target_shape=real_phase.shape[-2:])
        # Zero small elements
        dp = np.where(dp < 3, 0, dp)
        # Reshape to (N, C, H, W) and cast to tensor.
        if len(dp.shape) == 3:
            dp = torch.tensor(dp[:, np.newaxis, :, :].astype('float32'))
            real_mag = torch.tensor(real_mag[:, np.newaxis, :, :].astype('float32'))
            real_phase = torch.tensor(real_phase[:, np.newaxis, :, :].astype('float32'))
        else:
            dp = torch.tensor(dp[np.newaxis, np.newaxis, :, :].astype('float32'))
            real_mag = torch.tensor(real_mag[np.newaxis, np.newaxis, :, :].astype('float32'))
            real_phase = torch.tensor(real_phase[np.newaxis, np.newaxis, :, :].astype('float32'))
        return dp, real_mag, real_phase

    def resize_dp(self, dp, target_shape):
        return transform_data_for_ptychonn(dp, target_shape)

    def check_dataset(self):
        required_keys = ['data/real', 'data/reciprocal']
        for key in required_keys:
            if not self.f[key]:
                raise ValueError('HDF5 file does not have all the datasets required. Missing dataset: {}'.format(key))
