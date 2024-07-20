import os
import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

import tifffile

import pppc
from pppc.configs import InferenceConfigDict
from pppc.ptychonn.model import PtychoNNModel, PtychoNNTransposedConvModel
from pppc.reconstructor import DatasetInferencer, TileStitcher
from pppc.position_list import ProbePositionList
from pppc.io import NPZFileHandle

from test_predict import *

os.chdir('/data/programs/probe_position_correction_w_ptychonn/workspace/large')

matplotlib.rc('font',family='Times New Roman')
matplotlib.rcParams['font.size'] = 14
plt.viridis()


scan_idx = 233
n_images = 9
n_cols = 3


recons = tifffile.imread('outputs/pred_test235_model_phaseOnly_BN_36SpiralDatasets_meanSubStdData_cleaned_valRatio_10/pred_phase.tiff')[:n_images]

fig, ax = plt.subplots(int(np.ceil(n_images / n_cols)), n_cols, squeeze=False, figsize=(5, 5))
for i in range(len(recons)):
    ax[i // n_cols, i % n_cols].imshow(recons[i], cmap='viridis')
    ax[i // n_cols, i % n_cols].set_xticks([])
    ax[i // n_cols, i % n_cols].set_yticks([])

fontprops = fm.FontProperties(size=14)
scalebar = AnchoredSizeBar(ax[0, 0].transData,
                    100, '100 pixels', 'lower left',
                    pad=0.3,
                    color='black',
                    frameon=False,
                    size_vertical=1,
                    fontproperties=fontprops)

ax[0, 0].add_artist(scalebar)
plt.tight_layout()
plt.savefig('factory/demo_recon_images.pdf')
