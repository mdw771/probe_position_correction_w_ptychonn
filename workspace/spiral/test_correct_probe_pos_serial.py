import sys
sys.path.insert(0, '/data/programs/probe_position_correction_w_ptychonn/pppc')

import numpy as np

import pppc
from pppc.configs import InferenceConfig
from pppc.core import PtychoNNProbePositionCorrector


if __name__ == '__main__':
    config_dict = InferenceConfig()
    config_dict['model_path'] = '../trained_models/ptychoNN_CNN_encoder_decoder_pytorch_statedict.pth'
    config_dict['dp_data_path'] = 'data/data_w_pos_errors/scan221.npz'
    config_dict['probe_position_data_path'] = 'data/data_w_pos_errors/pos221.csv'
    config_dict['probe_position_data_unit'] = 'm'
    config_dict['pixel_size_nm'] = 7.92485795

    corrector = PtychoNNProbePositionCorrector(config_dict)
    corrector.build()
    corrector.run()

