import sys
sys.path.insert(0, '/data/programs/probe_position_correction_w_ptychonn/pppc')

import numpy as np

import pppc
from pppc.configs import InferenceConfig
from pppc.core import PtychoNNProbePositionCorrector
from pppc.io import DataFileHandle
from pppc.util import class_timeit

class RegularGridFileHandle(DataFileHandle):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.f = np.load(file_path)
        self.array = self.f['arr_0']
        self.shape = self.array.shape
        self.num_dps = self.shape[0] * self.shape[1]

    def get_dp_by_index(self, ind):
        y, x = self.get_actual_indices_for_consecutive_index(ind)
        return self.array[y, x, :, :]


if __name__ == '__main__':
    config_dict = InferenceConfig()
    config_dict['model_path'] = '../../trained_models/ptychoNN_CNN_encoder_decoder_pytorch_statedict.pth'
    config_dict['dp_data_file_handle'] = RegularGridFileHandle('data/20191008_39_diff_reduced.npz')
    config_dict['dp_data_file_handle'].slice_array((slice(0, 20), slice(0, 20)))
    # config_dict['probe_position_data_path'] = 'data/data_w_pos_errors/pos221.csv'
    # config_dict['probe_position_data_unit'] = 'm'
    # config_dict['pixel_size_nm'] = 7.92485795
    # config_dict['central_crop'] = (48, 48)
    config_dict['debug'] = False

    corrector = PtychoNNProbePositionCorrector(config_dict)
    corrector.build()
    corrector.run()
