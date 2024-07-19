import os
import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

import pppc
from pppc.configs import InferenceConfigDict
from pppc.ptychonn.model import PtychoNNModel, PtychoNNTransposedConvModel, PtychoNNPhaseOnlyModel
from pppc.reconstructor import DatasetInferencer, TileStitcher
from pppc.position_list import ProbePositionList
from pppc.io import NPZFileHandle

os.chdir('/data/programs/probe_position_correction_w_ptychonn/workspace/large')

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

def plot_distribution(x):
    x = x.reshape(-1)
    #x = x[x > 0]
    plt.hist(x, bins=32)
    plt.semilogy()
    plt.show()

def run_prediction(scan_idx, model_path, output_path=None):
    try:
        dataset_handle = NPZFileHandle('data/test{}.npz'.format(scan_idx))
    except:
        print('data/test{}.npz not found.'.format(scan_idx))
        dataset_handle = NPZFileHandle('data/scan{}.npz'.format(scan_idx))
    dataset_handle.array = clean_data(dataset_handle.array)
    dataset_handle.transform_data((128, 128), discard_len=(64, 64))
    dataset_handle.array = subtract_mean_and_standardize_data(dataset_handle.array)

    if output_path is None:
        output_path = 'outputs/pred_{}_{}'.format(
                            os.path.splitext(os.path.basename(dataset_handle.f.fid.name))[0],
                            os.path.basename(os.path.dirname(model_path)))

    config_dict = InferenceConfigDict(
        model_path=model_path,
        model=(PtychoNNPhaseOnlyModel, {'use_batchnorm': True}),
        batch_size=32,
        dp_data_file_handle=dataset_handle,
        cpu_only=False,
        prediction_output_path=output_path,
    )

    inferencer = DatasetInferencer(config_dict)
    inferencer.build()
    inferencer.run()
    inferencer.convert_output_files_into_single_tiff('pred_phase')
    #inferencer.convert_output_files_into_single_tiff('pred_amp')


if __name__ == '__main__':
    # scan_indices = [233, 234, 235, 236, 239, 240, 241, 242, 244, 245, 246, 247, 250, 251, 252, 253]
    scan_indices = [235]
    decimate_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    # decimate_ratios = [0.1]

    # for scan_idx in scan_indices:
    #     for decimate_ratio in decimate_ratios:
    #         model_path = '../../trained_models/model_phaseOnly_BN_36SpiralDatasets_meanSubStdData_dataDecimation_{}/best_model.pth'.format(decimate_ratio)
    #         run_prediction(scan_idx, model_path)
    for scan_idx in scan_indices:
        model_path = '../../trained_models/model_phaseOnly_BN_36SpiralDatasets_meanSubStdData_cleaned_valRatio_10/best_model.pth'
        run_prediction(scan_idx, model_path)

