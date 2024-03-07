import sys
import os
import collections

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tifffile
import h5py

import pppc
from pppc.configs import InferenceConfigDict
from pppc.core import PtychoNNProbePositionCorrector, ProbePositionCorrectorChain
from pppc.ptychonn.model import PtychoNNModel, PtychoNNPhaseOnlyModel
from pppc.io import DataFileHandle, NPZFileHandle, VirtualDataFileHandle
from pppc.position_list import ProbePositionList
from pppc.reconstructor import VirtualReconstructor
from pppc.util import class_timeit
from pppc.message_logger import logger


def prediction_transform(recons):
    recons = recons.astype('float32')
    recons = recons * 7.74
    return recons


matplotlib.rc('font',family='Times New Roman')
matplotlib.rcParams['font.size'] = 14
plt.viridis()

###################################
scan_idx = 0
save_figs = True
# output_dir = os.path.join('outputs', 'test{}_reduced_0_22'.format(scan_idx), 'nn_8_tol_8e-3_variableTol_newErrorMap')
output_dir = os.path.join('outputs', 'test{}_unscaled_reduced_4_22'.format(scan_idx), 'nn_8_tol_3e-4_4e-4_variableTol_newErrorMap_unscaled')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

slicer = slice(None)

true_pos_fname = 'data/test/pos_true_m_reduced_4_22.csv'
baseline_pos_fname = 'data/test/pos_baseline_m_reduced_4_22.csv'
psize_nm = 16
###################################

probe_pos_true = np.genfromtxt(true_pos_fname, delimiter=',')[slicer]
probe_pos_true = probe_pos_true / (psize_nm * 1e-9)
probe_pos_list_true = ProbePositionList(position_list=probe_pos_true)
fig, ax, scat = probe_pos_list_true.plot(show=False, return_obj=True)
if save_figs:
    fig.savefig(os.path.join(output_dir, 'path_plot_true.pdf'))
    plt.close(fig)
else:
    plt.show()
    
probe_pos_list_baseline = np.genfromtxt(baseline_pos_fname, delimiter=',').astype('float32')[slicer] / (psize_nm * 1e-9)

recons = tifffile.imread('outputs/pred_data_test_labelPsize_16_every2Rows_reduced_4_22_model_2IDDSiemensStarDataset_labelPsize_16_std_meanSub_data_std_labels_every2Rows_bn_phaseOnlyModel_lr_1e-3_scheduledLR/pred_phase.tiff')
recons = recons[slicer]

reconstructor = VirtualReconstructor(InferenceConfigDict())
reconstructor.set_object_image_array(recons)

config_dict = InferenceConfigDict(
    model_path='../../trained_models/model_2IDDSiemensStarDataset_labelPsize_16_std_meanSub_data_std_labels_every2Rows_bn_phaseOnlyModel_lr_1e-3_scheduledLR/best_model.pth',
    model=(PtychoNNPhaseOnlyModel, {'use_batchnorm': True}),
    dp_data_file_handle=VirtualDataFileHandle('', dp_shape=recons.shape[1:], num_dps=recons.shape[0]),
    ptycho_reconstructor=reconstructor,
    random_seed=196,
    debug=False,
    probe_position_list=ProbePositionList(position_list=probe_pos_list_baseline),
    baseline_position_list=ProbePositionList(position_list=probe_pos_list_baseline),
)


##############################
config_dict.load_from_json(os.path.join('config_jsons', 'config_{}_init_unscaled.json'.format(scan_idx)))
##############################

corrector_chain = ProbePositionCorrectorChain(config_dict)
corrector_chain.build()
corrector_chain.verbose = False

corrector_chain.run_correction_iteration(0)

corrector_s = corrector_chain.corrector_list[-1]
fig, ax, scat = corrector_s.new_probe_positions.plot(return_obj=True, show=False)
if save_figs:
    fig.savefig(os.path.join(output_dir, 'path_plot_iter_0.pdf'))
    plt.close(fig)
else:
    plt.show()

corrector_s.new_probe_positions.to_csv(os.path.join(output_dir, 'pos_iter_0.csv'), unit='m', psize_nm=psize_nm)
    
probe_pos_true = np.genfromtxt(true_pos_fname, delimiter=',')[slicer]
probe_pos_true = probe_pos_true / (psize_nm * 1e-9)
probe_pos_list_calc = corrector_s.new_probe_positions.array

probe_pos_list_calc = probe_pos_list_calc - np.mean(probe_pos_list_calc, axis=0)
probe_pos_true = probe_pos_true - np.mean(probe_pos_true, axis=0)
probe_pos_baseline = probe_pos_list_baseline - np.mean(probe_pos_list_baseline, axis=0)
#probe_pos_list_calc = probe_pos_list_calc * 1.02
plt.close()
fig, ax = plt.subplots(1, 1)
ax.scatter(probe_pos_list_calc[:, 1], probe_pos_list_calc[:, 0], s=1)
ax.scatter(probe_pos_true[:, 1], probe_pos_true[:, 0], s=1)
ax.scatter(probe_pos_baseline[:, 1], probe_pos_baseline[:, 0], s=0.5, color='gray', alpha=0.2)
ax.plot(probe_pos_list_calc[:, 1], probe_pos_list_calc[:, 0], linewidth=0.5, label='Calculated')
ax.plot(probe_pos_true[:, 1], probe_pos_true[:, 0], linewidth=0.5, label='True')
ax.plot(probe_pos_baseline[:, 1], probe_pos_baseline[:, 0], linewidth=0.5, label='Baseline', color='gray', alpha=0.2)
plt.legend(fontsize=5)
if save_figs:
    fig.savefig(os.path.join(output_dir, 'comparison_path_plot_iter_0.pdf'))
    plt.close(fig)
else:
    plt.show()

sys.exit()

corrector_chain.run_correction_iteration(1)

corrector_c1 = corrector_chain.corrector_list[-1]
fig, ax, scat = corrector_c1.new_probe_positions.plot(return_obj=True, show=False)
if save_figs:
    fig.savefig(os.path.join(output_dir, 'path_plot_iter_1.pdf'), format='pdf')
    plt.close(fig)
else:
    plt.show()

corrector_c1.new_probe_positions.to_csv(os.path.join(output_dir, 'pos_iter_1.csv'), unit='m', psize_nm=psize_nm)

probe_pos_true = np.genfromtxt(true_pos_fname, delimiter=',')[slicer]
probe_pos_true = probe_pos_true / (psize_nm * 1e-9)
probe_pos_list_calc = corrector_c1.new_probe_positions.array

probe_pos_list_calc = probe_pos_list_calc - np.mean(probe_pos_list_calc, axis=0)
probe_pos_true = probe_pos_true - np.mean(probe_pos_true, axis=0)
#probe_pos_list_calc = probe_pos_list_calc * 1.02
fig, ax = plt.subplots(1, 1)
ax.scatter(probe_pos_list_calc[:, 1], probe_pos_list_calc[:, 0], s=1)
ax.scatter(probe_pos_true[:, 1], probe_pos_true[:, 0], s=1)
ax.plot(probe_pos_list_calc[:, 1], probe_pos_list_calc[:, 0], linewidth=0.5, label='Calculated')
ax.plot(probe_pos_true[:, 1], probe_pos_true[:, 0], linewidth=0.5, label='True')
plt.legend()
if save_figs:
    fig.savefig(os.path.join(output_dir, 'comparison_path_plot_iter_1.pdf'))
    plt.close(fig)
else:
    plt.show()


