import os

import numpy as np
import pandas as pd

from pppc.util import class_timeit

class DataFileHandle:

    def __init__(self, file_path):
        self.file_path = file_path
        self.f = None
        self.array = None
        self.num_dps = None
        self.shape = None

    def get_dp_by_index(self, ind):
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


class NPZFileHandle(DataFileHandle):

    def __init__(self, file_path):
        super().__init__(file_path)
        self.f = np.load(self.file_path)
        self.array = self.f['reciprocal']
        self.num_dps = self.array.shape[0]
        self.shape = self.array.shape

    def get_dp_by_index(self, ind):
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


def load_probe_positions_from_file(file_path):
    """
    Load probe positions from file.

    :param file_path: str.
    :return: np.ndarray. The returned shape is [n, 2], with each subarray storing a position in (y, x).
    """
    fmt = os.path.splitext(file_path)[-1]
    if fmt == '.csv':
        probe_pos = pd.read_csv(file_path, header=None)
        probe_pos = probe_pos.to_numpy()[:, ::-1]
    else:
        raise ValueError('Unsupported format "{}".'.format(fmt))
    return probe_pos
