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


matplotlib.rc('font',family='Times New Roman')
matplotlib.rcParams['font.size'] = 14
plt.viridis()

scan_idx = 235
n_images = 9
n_cols = 3
psize_nm = 8

image_path = 'outputs/pred_test{}_model_phaseOnly_BN_36SpiralDatasets_meanSubStdData_cleaned_valRatio_10/pred_phase.tiff'.format(scan_idx)

probe_pos_list = ProbePositionList(file_path='data/pos{}.csv'.format(scan_idx),
                                   unit='m', psize_nm=psize_nm, convert_to_pixel=True, first_is_x=False)

scaling_dict = collections.defaultdict(lambda: 1.0,
                                       {236: 0.5, 239: 0.5, 240: 0.25, 241: 0.25, 242: 0.25, 250: 0.5, 251: 0.5,
                                        252: 0.25, 253: 0.25})
s = scaling_dict[scan_idx]

probe_pos_list_true = np.genfromtxt('data/pos{}.csv'.format(scan_idx), delimiter=',').astype('float32') / (psize_nm * 1e-9)
probe_pos_list_true = probe_pos_list_true[:n_images]
probe_pos_list_baseline = np.genfromtxt('data/pos221.csv', delimiter=',').astype('float32') / (psize_nm * 1e-9) * s  # Baseline
probe_pos_list_baseline = probe_pos_list_baseline[:n_images]

try:
    recons = tifffile.imread(image_path)
except:
    print('Reading images from scan### folder.')
    recons = tifffile.imread('outputs/pred_scan{}_model_36SpiralDatasets_model_PtychoNNModel_nLevels_4_batchSizePerProcess_32_learningRatePerProcess_0.0001/pred_phase.tiff'.format(scan_idx))
recons = recons[:n_images]

reconstructor = VirtualReconstructor(InferenceConfigDict())
reconstructor.set_object_image_array(recons)

config_dict = InferenceConfigDict(
    model_path='../../trained_models/model_36SpiralDatasets_cleaned/best_model.pth',
    model=(PtychoNNModel, {'n_levels': 4}),
    ptycho_reconstructor=reconstructor,
    dp_data_file_handle=VirtualDataFileHandle('', dp_shape=recons.shape[1:], num_dps=recons.shape[0]),
    random_seed=196,
    debug=False,
    central_crop=None,
)

config_dict.probe_position_list = ProbePositionList(position_list=probe_pos_list_baseline)

# Load config
config_fname = os.path.join('config_jsons', 'config_{}.json'.format(scan_idx))
config_dict.load_from_json(config_fname)

config_dict.num_neighbors_collective = 8
config_dict.method = 'collective'

print(config_dict)

corrector = PtychoNNProbePositionCorrector(config_dict)
corrector.build()
corrector.run()

def convert_ab_to_offset_matrix(a, b):
    m = np.full([n_images, n_images, 2], np.nan)
    for p in range(a.shape[0]):
        i = np.where(a[p] > 0)[0][0]
        j = np.where(a[p] < 0)[0][0]
        m[i, j] = -b[p]
        if j != i:
            m[j, i] = b[p]
    m[np.arange(n_images), np.arange(n_images)] = 0
    return m


offsets = convert_ab_to_offset_matrix(corrector.a_mat, corrector.b_vec)

offsets_true = probe_pos_list_true - probe_pos_list_true.reshape(-1, 1, 2)

fig, ax = plt.subplots(2, 2, figsize=(5.7, 6))
ax[0, 0].imshow(np.linalg.norm(offsets_true, axis=2), vmin=0, vmax=50)
ax[0, 0].set_title('Magnitude, true')
ax[0, 0].set_xticks([0, 2, 4, 6, 8])
im = ax[0, 1].imshow(np.linalg.norm(offsets, axis=2), vmin=0, vmax=50)
ax[0, 1].set_title('Magnitude, registered')
ax[0, 1].set_xticks([0, 2, 4, 6, 8])
cb_ax = fig.add_axes([0.98, 0.55, 0.04, 0.415])
plt.colorbar(im, cax=cb_ax)
ax[1, 0].imshow(np.rad2deg(np.arctan2(offsets_true[:, :, 0], offsets_true[:, :, 1])), vmin=-180, vmax=180)
ax[1, 0].set_title('Direction, true')
ax[1, 0].set_xticks([0, 2, 4, 6, 8])
im = ax[1, 1].imshow(np.rad2deg(np.arctan2(offsets[:, :, 0], offsets[:, :, 1])), vmin=-180, vmax=180)
ax[1, 1].set_title('Direction, registered')
ax[1, 1].set_xticks([0, 2, 4, 6, 8])
cb_ax = fig.add_axes([0.98, 0.05, 0.04, 0.415])
plt.colorbar(im, cax=cb_ax, ticks=[-180, -90, 0, 90, 180])

plt.tight_layout()

# fig, ax = plt.subplots(2, 2)
# ax[0, 0].imshow(offsets_true[:, :, 0])
# ax[0, 0].set_title('y, true')
# ax[0, 1].imshow(offsets[:, :, 0])
# ax[0, 1].set_title('y, registered')
# ax[1, 0].imshow(offsets_true[:, :, 1])
# ax[1, 0].set_title('x, true')
# ax[1, 1].imshow(offsets[:, :, 1])
# ax[1, 1].set_title('x, registered')

plt.savefig(os.path.join('factory', 'demo_registration.pdf'), bbox_inches='tight')

