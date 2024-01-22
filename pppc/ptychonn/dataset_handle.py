import h5py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import resize

from pppc.helper import transform_data_for_ptychonn


class HDF5Dataset(Dataset):
    def __init__(self, filename, transform_func='default', transform_func_kwargs=None,
                 label_transform_func=None,
                 standardized=False, verbose=False):
        self.filename = filename
        self.f = h5py.File(self.filename, 'r')
        self.verbose = verbose
        self.standardized = standardized
        self.read_complex = True
        self.check_dataset()
        if transform_func == 'default':
            self.transform_func = self.resize_dp
        else:
            self.transform_func = transform_func
        self.label_transform_func = label_transform_func
        if transform_func_kwargs is None:
            transform_func_kwargs = {}
        self.transform_func_kwargs = transform_func_kwargs

    def __len__(self):
        if self.read_complex:
            return self.f['data/real'].shape[0]
        else:
            return self.f['data/real_phase'].shape[0]

    def __getitem__(self, idx):
        if self.verbose:
            print('Retrieving index: {}'.format(idx))
        if self.read_complex:
            real = self.f['data/real'][idx, :, :]
            real_phase = np.angle(real)
            real_mag = np.abs(real)
        else:
            real_phase = self.f['data/real_phase'][idx, :, :]
            real_mag = self.f['data/real_magnitude'][idx, :, :]
        dp = self.f['data/reciprocal'][idx, :, :]
        dp, real_mag, real_phase = self.process_data(dp, real_mag, real_phase)
        return dp, real_mag, real_phase

    def __getitems__(self, idx_list):
        idx_list = tuple(np.sort(idx_list))
        if self.verbose:
            print('Retrieving indices: {}'.format(idx_list))
        if self.read_complex:
            real = self.f['data/real'][idx_list, :, :]
            real_phase = np.angle(real)
            real_mag = np.abs(real)
        else:
            real_phase = self.f['data/real_phase'][idx_list, :, :]
            real_mag = self.f['data/real_magnitude'][idx_list, :, :]
        dp = self.f['data/reciprocal'][idx_list, :, :]
        dp, real_mag, real_phase = self.process_data(dp, real_mag, real_phase)
        return dp, real_mag, real_phase

    def process_data(self, dp, real_mag, real_phase):
        target_shape = real_phase.shape[-2:]
        if self.transform_func == self.resize_dp:
            self.transform_func_kwargs = {'target_shape': target_shape}
        if self.transform_func is not None:
            dp = self.transform_func(dp, **self.transform_func_kwargs)
        if self.label_transform_func is not None:
            real_mag, real_phase = self.label_transform_func(real_mag, real_phase)
        # Zero small elements
        if not self.standardized:
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

    def resize_dp(self, dp, target_shape=(128, 128)):
        if not np.array_equal(dp.shape[-2:], target_shape):
            return transform_data_for_ptychonn(dp, target_shape, overflow_correction=True)

    def has_key(self, f, key):
        try:
            f[key]
            return True
        except:
            return False

    def check_dataset(self):
        required_keys = [('data/real', 'data/real_phase', 'data/real_magnitude'), 'data/reciprocal']
        for key in required_keys:
            good = False
            if isinstance(key, tuple):
                for subkey in key:
                    if self.has_key(self.f, subkey):
                        good = True
                        if subkey == 'data/real_phase':
                            self.read_complex = False
                            print('{}: Reading real-numbered phase and magnitude.'.format(self.check_dataset.__name__))
                        break
            else:
                if self.has_key(self.f, key):
                    good = True
            if not good:
                raise ValueError('HDF5 file does not have all the datasets required. Missing dataset: {}'.format(key))
