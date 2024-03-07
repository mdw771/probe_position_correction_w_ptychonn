import os
import argparse
import shutil

import numpy as np
import tifffile

import pppc
from pppc.configs import InferenceConfigDict
from pppc.ptychonn.model import PtychoNNModel, PtychoNNTransposedConvModel, PtychoNNPhaseOnlyModel
from pppc.reconstructor import DatasetInferencer, TileStitcher
from pppc.position_list import ProbePositionList
from pppc.io import NPZFileHandle


def clean_data(arr):
    mask = arr < 0
    vals = arr[mask]
    vals = 32768 + (vals - -32768)
    arr[mask] = vals
    return arr


def subtract_mean_and_standardize_data(data):
    data = data - np.mean(data, axis=0)
    data = (data - np.mean(data)) / np.std(data)
    return data


def test_ptychonn_predict(generate_gold=False):
    gold_dir = 'data_gold/test_ptychonn_predict'

    config_dict = InferenceConfigDict()
    config_dict['model_path'] = 'data/model_ptychonn_bn_meanSubStdData.pth'
    config_dict['model'] = (PtychoNNPhaseOnlyModel, {'use_batchnorm': True})
    config_dict['batch_size'] = 32
    config_dict['dp_data_file_handle'] = NPZFileHandle('data/test235_truncated.npz')
    # Correct int16 overflow values
    config_dict['dp_data_file_handle'].array = clean_data(config_dict['dp_data_file_handle'].array)
    config_dict['dp_data_file_handle'].transform_data((128, 128), discard_len=(64, 64))
    config_dict['dp_data_file_handle'].array = subtract_mean_and_standardize_data(
        config_dict['dp_data_file_handle'].array)

    config_dict['cpu_only'] = True
    config_dict['prediction_output_path'] = 'temp' if not generate_gold else gold_dir

    inferencer = DatasetInferencer(config_dict)
    inferencer.build()
    inferencer.run()
    inferencer.convert_output_files_into_single_tiff('pred_phase', delete_individual_files_after_complete=True)

    if not generate_gold:
        data = tifffile.imread('temp/pred_phase.tiff')
        gold_data = tifffile.imread(os.path.join(gold_dir, 'pred_phase.tiff'))
        shutil.rmtree('temp')
        assert np.allclose(data, gold_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    test_ptychonn_predict(generate_gold=args.generate_gold)
