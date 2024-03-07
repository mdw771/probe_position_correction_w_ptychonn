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

    config_dict = InferenceConfigDict(
        reconstruction_image_path=os.path.join('data', 'pred_test{}'.format(scan_idx), 'pred_phase.tiff'),
        random_seed=196,
        debug=False,
        probe_position_list=None,
        central_crop=None
    )
    config_dict.load_from_json(os.path.join('data', 'config_{}.json'.format(scan_idx)))
    print(config_dict)

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
    print(gold_pos_list, calc_pos_list)
    assert np.allclose(calc_pos_list, gold_pos_list, atol=1e-1)


if __name__ == '__main__':
    test_multiiter_pos_calculation()
