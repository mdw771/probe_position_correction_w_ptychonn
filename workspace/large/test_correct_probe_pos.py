import sys
import os
import collections
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tifffile

import pppc
from pppc.configs import InferenceConfigDict
from pppc.core import PtychoNNProbePositionCorrector, ProbePositionCorrectorChain
from pppc.ptychonn.model import PtychoNNModel
from pppc.io import DataFileHandle, NPZFileHandle, VirtualDataFileHandle
from pppc.position_list import ProbePositionList
from pppc.reconstructor import VirtualReconstructor
from pppc.util import class_timeit

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
except:
    rank = 0
    n_ranks = 1


def run_pos_corr(scan_idx, image_path, save_path='outputs', use_baseline=False, config_file_postfix_to_use_if_any=None):
    save_figs = True
    output_dir = os.path.join(save_path, 'test{}'.format(scan_idx))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # psize_nm = np.load('data/scan221_raw.npz')['pixelsize'] * 1e9
    psize_nm = 8
    probe_pos_list = ProbePositionList(file_path='data/pos{}.csv'.format(scan_idx),
                                       unit='m', psize_nm=psize_nm, convert_to_pixel=True, first_is_x=False)
    fig, ax, scat = probe_pos_list.plot(show=False, return_obj=True)
    if save_figs:
        fig.savefig(os.path.join(output_dir, 'path_plot_true.pdf'.format(scan_idx)))
    else:
        plt.show()

    scaling_dict = collections.defaultdict(lambda: 1.0,
                                           {236: 0.5, 239: 0.5, 240: 0.25, 241: 0.25, 242: 0.25, 250: 0.5, 251: 0.5,
                                            252: 0.25, 253: 0.25})
    s = scaling_dict[scan_idx]
    probe_pos_list_baseline = np.genfromtxt('data/pos221.csv', delimiter=',').astype('float32') / (psize_nm * 1e-9) * s  # Baseline

    try:
        #recons = tifffile.imread('outputs/pred_test{}_model_36SpiralDatasets_model_PtychoNNModel_nLevels_4_batchSizePerProcess_32_learningRatePerProcess_0.0001/pred_phase.tiff'.format(scan_idx))
        recons = tifffile.imread(image_path)
    except:
        print('Reading images from scan### folder.')
        recons = tifffile.imread('outputs/pred_scan{}_model_36SpiralDatasets_model_PtychoNNModel_nLevels_4_batchSizePerProcess_32_learningRatePerProcess_0.0001/pred_phase.tiff'.format(scan_idx))
    #recons = tifffile.imread('data/scan{}_phase.tiff'.format(scan_idx))
    config_dict = InferenceConfigDict()
    # config_dict['model_path'] = '../../trained_models/model_36SpiralDatasets_model_PtychoNNModel_nLevels_4_batchSizePerProcess_32_learningRatePerProcess_0.0001/best_model.pth'
    config_dict['model_path'] = '../../trained_models/model_36SpiralDatasets_cleaned/best_model.pth'
    config_dict['model'] = (PtychoNNModel, {'n_levels': 4})
    # config_dict['dp_data_file_handle'] = NPZFileHandle('data/test{}.npz'.format(scan_idx))
    config_dict['dp_data_file_handle'] = VirtualDataFileHandle('', dp_shape=recons.shape[1:], num_dps=recons.shape[0])
    # config_dict['dp_data_file_handle'].transform_data((128, 128), discard_len=(64, 64))
    config_dict['ptycho_reconstructor'] = VirtualReconstructor(InferenceConfigDict())
    config_dict['ptycho_reconstructor'].set_object_image_array(recons)
    config_dict['random_seed'] = 196 
    config_dict['debug'] = False
    config_dict['central_crop'] = None

    if use_baseline:
        config_dict['probe_position_list'] = ProbePositionList(position_list=probe_pos_list_baseline)
        config_dict['baseline_position_list'] = None
    else:
        config_dict['probe_position_list'] = None if scan_idx != 247 else ProbePositionList(position_list=probe_pos_list_baseline)
        config_dict['baseline_position_list'] = ProbePositionList(position_list=probe_pos_list_baseline)

    # Load config
    config_fname = os.path.join('config_jsons', 'config_{}.json'.format(scan_idx))
    if config_file_postfix_to_use_if_any is not None:
        temp_fname = os.path.join('config_jsons', 'config_{}_{}.json'.format(scan_idx, config_file_postfix_to_use_if_any))
        if os.path.exists(temp_fname):
            config_fname = temp_fname
    config_dict.load_from_json(config_fname)

    # If using baseline as initial guess, skip serial mode iteration.
    if use_baseline:
        print('Skipping serial mode iteration because use_baseline is True.')
        time.sleep(3)
        if 'method_multiiter' in config_dict.keys() and config_dict['method_multiiter'][0] == 'serial':
            for key in config_dict.keys():
                if 'multiiter' in key:
                    config_dict[key] = config_dict[key][1:]

    print(config_dict)

    corrector_chain = ProbePositionCorrectorChain(config_dict)
    corrector_chain.build()
    corrector_chain.verbose = False

    corrector_chain.run_correction_iteration(0)
    corrector_s = corrector_chain.corrector_list[-1]
    fig, ax, scat = corrector_s.new_probe_positions.plot(return_obj=True, show=False)
    if save_figs:
        fig.savefig(os.path.join(output_dir, 'path_plot_iter_0.pdf'.format(scan_idx)))
    else:
        plt.show()

    # corrector_chain.config_dict['sift_outlier_removal_method'] = 'trial_error'
    corrector_chain.run_correction_iteration(1)
    corrector_c1 = corrector_chain.corrector_list[1]
    fig, ax, scat = corrector_c1.new_probe_positions.plot(return_obj=True, show=False)
    if save_figs:
        fig.savefig(os.path.join(output_dir, 'path_plot_collective_iter_1_nn_12_sw_1e-2.pdf'.format(scan_idx)), format='pdf')
    else:
        plt.show()

    if corrector_chain.n_iters > 2:
        corrector_chain.run_correction_iteration(2)
        corrector_c2 = corrector_chain.corrector_list[2]
    else:
        corrector_c2 = corrector_c1
    fig, ax, scat = corrector_c2.new_probe_positions.plot(return_obj=True, show=False)
    if save_figs:
        fig.savefig(os.path.join(output_dir, 'path_plot_collective_iter_2_nn_12_sw_1e-3_1e-2.pdf'.format(scan_idx)), format='pdf')
    else:
        plt.show()

    with open(os.path.join(output_dir, 'redone_with_baseline.txt'.format(scan_idx)), 'w') as f:
        f.write(str(int(corrector_chain.redone_with_baseline)))

    probe_pos_list_calc = corrector_c2.new_probe_positions.array
    probe_pos_list_true = np.genfromtxt('data/pos{}.csv'.format(scan_idx), delimiter=',').astype('float32') / (psize_nm * 1e-9)
    probe_pos_list_calc -= np.mean(probe_pos_list_calc, axis=0)
    probe_pos_list_true -= np.mean(probe_pos_list_true, axis=0)
    # probe_pos_list_raw *= 1.05
    plt.figure()
    plt.scatter(probe_pos_list_calc[:, 1], probe_pos_list_calc[:, 0], s=1)
    plt.scatter(probe_pos_list_true[:, 1], probe_pos_list_true[:, 0], s=1)
    plt.plot(probe_pos_list_calc[:, 1], probe_pos_list_calc[:, 0], linewidth=0.5, label='Calculated')
    plt.plot(probe_pos_list_true[:, 1], probe_pos_list_true[:, 0], linewidth=0.5, label='True')
    plt.legend()
    if save_figs:
        plt.savefig(os.path.join(output_dir, 'comparison_path_plot_collective_iter_2_nn_12_sw_1e-3_1e-2.pdf'.format(scan_idx)))
    else:
        plt.show()

    if save_figs:
        corrector_c2.new_probe_positions.to_csv(os.path.join(output_dir, 'calc_pos_{}_collective_niters_2_beta_0p5_nn_12_sw_1e-2_1e-3.csv'.format(scan_idx)), psize_nm=psize_nm)

    plt.close()


if __name__ == '__main__':
    matplotlib.rcParams['pdf.fonttype'] = 'truetype'
    fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
    plt.rc('font', **fontProperties)
    plt.viridis()

    mode = 'decimation'

    # scan_indices = [233, 234, 235, 236, 239, 240, 241, 242, 244, 245, 246, 247, 250, 251, 252, 253]
    scan_indices = [235]
    #scan_indices = [234, 239, 240]
    # decimate_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    decimate_ratios = [0.9]

    if mode == 'decimation':
        for decimate_ratio in decimate_ratios:
            for scan_idx in scan_indices[rank::n_ranks]:
                print('==========================')
                print('Now running {}'.format(scan_idx))
                print('==========================')
                # run_pos_corr(scan_idx,
                #              'outputs/pred_test{}_model_36SpiralDatasets_cleaned_valRatio_10/pred_phase.tiff'.format(scan_idx),
                #              save_path='outputs'
                #              )
                run_pos_corr(scan_idx,
                             'outputs/pred_test{}_model_phaseOnly_BN_36SpiralDatasets_meanSubStdData_dataDecimation_{}/pred_phase.tiff'.format(scan_idx, decimate_ratio),
                             save_path='outputs/dataDecimation_{}'.format(decimate_ratio),
                             use_baseline=True,
                             config_file_postfix_to_use_if_any='decimation'
                             )
    else:
        for scan_idx in scan_indices[rank::n_ranks]:
            print('==========================')
            print('Now running {}'.format(scan_idx))
            print('==========================')
            run_pos_corr(scan_idx,
                         'outputs/pred_test{}_model_phaseOnly_BN_36SpiralDatasets_meanSubStdData_cleaned_valRatio_10/pred_phase.tiff'.format(scan_idx),
                         save_path='outputs'
                         )

