import os

import numpy as np
import tifffile

import pppc
from pppc.configs import InferenceConfigDict
from pppc.core import PtychoNNProbePositionCorrector, ProbePositionCorrectorChain
from pppc.ptychonn.model import PtychoNNModel
from pppc.io import DataFileHandle, NPZFileHandle, VirtualDataFileHandle
from pppc.position_list import ProbePositionList
from pppc.reconstructor import VirtualReconstructor
from pppc.util import class_timeit


def test_multiiter_pos_calculation():
    scan_idx = 235
    recons = tifffile.imread(os.path.join('data', 'pred_test{}'.format(scan_idx), 'pred_phase.tiff'))

    config_dict = InferenceConfigDict()
    config_dict['dp_data_file_handle'] = VirtualDataFileHandle('', dp_shape=recons.shape[1:], num_dps=recons.shape[0])
    config_dict['ptycho_reconstructor'] = VirtualReconstructor(InferenceConfigDict())
    config_dict['ptycho_reconstructor'].set_object_image_array(recons)
    config_dict['random_seed'] = 196
    config_dict['debug'] = False
    config_dict['probe_position_list'] = None
    config_dict['central_crop'] = None
    config_dict.load_from_json(os.path.join('data', 'config_{}.json'.format(scan_idx)))

    corrector_chain = ProbePositionCorrectorChain(config_dict)
    corrector_chain.verbose = False
    corrector_chain.build()
    corrector_chain.run()

    calc_pos_list = corrector_chain.corrector_list[-1].new_probe_positions.array

    gold_pos_list = np.genfromtxt(os.path.join('data_gold',
                                               'calc_pos_235_collective_niters_2_beta_0p5_nn_12_sw_1e-2_1e-3.csv'),
                                  delimiter=',')
    gold_pos_list = gold_pos_list / 8e-9
    calc_pos_list -= np.mean(calc_pos_list, axis=0)
    gold_pos_list -= np.mean(gold_pos_list, axis=0)
    assert np.allclose(calc_pos_list, gold_pos_list, atol=1e-1)


if __name__ == '__main__':
    test_multiiter_pos_calculation()
