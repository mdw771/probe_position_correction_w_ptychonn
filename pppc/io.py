import os

import numpy as np
import pandas as pd
from skimage.transform import resize

from pppc.util import class_timeit
from pppc.helper import transform_data_for_ptychonn

class DataFileHandle:

    def __init__(self, file_path):
        self.file_path = file_path
        self.f = None
        self.array = None
        self.num_dps = None
        self.shape = None

    def get_dp_by_raw_index(self, ind):
        """
        Takes in an index and return the corresponding DP. The index is assumed to be row-major for a 4D data array
        with a 2D scan grid.

        :param ind: int.
        :return: np.ndarray.
        """
        if self.array.ndim == 3:
            return self.array[ind]
        else:
            unraveled_inds = np.unravel_index(ind, self.array.shape[:2])
            return self.array[unraveled_inds[0], unraveled_inds[1], :, :]

    def get_dp_by_consecutive_index(self, ind):
        """
        Takes in an index and return the corresponding DP. The index is assumed to be consecutive, such that
        the DP with `ind` and `ind - 1` are guaranteed to be spatially connected. This means that for a 2D scan grid,
        `ind` indexes the DPs in a snake pattern.

        :param ind: int. A consecutive index.
        :return: np.ndarray.
        """
        pass

    def get_actual_indices_for_consecutive_index(self, ind, ravel=False):
        """
        Assuming a 2D scan grid (such that the DP data are stored in a 4D array), where DPs are arranged in
        row-major order, this function takes in a consecutive index `ind`, where it is assumed that the object
        indexed by `ind` and `ind - 1` must be spatially connected. In other words, `ind` follows a snake pattern.
        The function returns the raw indices of the DP indexed by `ind` in the actual data array.

        :param ind: int. The consecutive index of a DP.
        :param ravel: bool. Whether to return a raveled scalar index or unraveled indices (y, x) if the scan grid is
                            2D.
        :return: int, int, or int if `ravel == True`.
        """
        if len(self.shape) == 3:
            return ind
        if ind // self.shape[1] % 2 == 0:
            # On even rows
            raw_inds = (ind // self.shape[1], ind % self.shape[1])
        else:
            raw_inds = (ind // self.shape[1], self.shape[1] - ind % self.shape[1] - 1)

        if ravel:
            return raw_inds[0] * self.shape[1] + raw_inds[1]
        else:
            return raw_inds

    def slice_array(self, slicers):
        """
        Subslice the data array.

        :param slicers: list[slice].
        """
        self.array = self.array[slicers]
        if self.array.ndim == 4:
            self.num_dps = self.array.shape[0] * self.array.shape[1]
        else:
            self.num_dps = self.array.shape[0]
        self.shape = self.array.shape

    def transform_data(self, target_shape=(128, 128), discard_len=None):
        batch_size = 32
        new_arr = np.zeros([self.num_dps, *target_shape])
        i_start = 0
        i_end = min(i_start + batch_size, self.num_dps)
        while i_start < self.num_dps:
            data_transformed = transform_data_for_ptychonn(self.array[i_start:i_end], target_shape,
                                                           discard_len=discard_len)
            # Zero small elements
            data_transformed = np.where(data_transformed < 3, 0, data_transformed)
            new_arr[i_start:i_end] = data_transformed
            i_start = i_end
            i_end = min(i_start + batch_size, self.num_dps)
        self.array = new_arr


class VirtualDataFileHandle(DataFileHandle):

    def __init__(self, file_path, dp_shape, num_dps):
        super().__init__(file_path)
        self.dp_shape = dp_shape
        self.num_dps = num_dps
        self.shape = [self.num_dps, *self.dp_shape]

    def get_dp_by_consecutive_index(self, ind):
        return np.zeros(self.dp_shape)

    def get_dp_by_raw_index(self, ind):
        return np.zeros(self.dp_shape)


class NPZFileHandle(DataFileHandle):

    def __init__(self, file_path):
        super().__init__(file_path)
        self.f = np.load(self.file_path)
        self.array = self.f['reciprocal']
        self.num_dps = self.array.shape[0]
        self.shape = self.array.shape

    def get_dp_by_consecutive_index(self, ind):
        if hasattr(ind, '__len__'):
            return self.array[*ind, :, :]
        else:
            return self.array[ind, :, :]


def create_data_file_handle(file_path) -> DataFileHandle:
    """
    Create a file handle (file pointer) to diffraction data file, so that data can be loaded on-demands.

    :param file_path: str.
    :return: DataFileHandle.
    """
    fmt = os.path.splitext(file_path)[-1]
    if fmt == '.npz':
        return NPZFileHandle(file_path)


def load_probe_positions_from_file(file_path, first_is_x=True):
    """
    Load probe positions from file.

    :param file_path: str.
    :param first_is_x: bool. If True, the first value of each coordinates is assumed to be x, and the output array
                             is flipped to return coordinates in (y, x).
    :return: np.ndarray. The returned shape is [n, 2], with each subarray storing a position in (y, x).
    """
    fmt = os.path.splitext(file_path)[-1]
    if fmt == '.csv':
        probe_pos = pd.read_csv(file_path, header=None).to_numpy()
    else:
        raise ValueError('Unsupported format "{}".'.format(fmt))
    if first_is_x:
        probe_pos = probe_pos[:, ::-1]
    return probe_pos
