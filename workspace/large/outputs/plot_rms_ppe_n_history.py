import os
import glob
import sys
sys.path.insert(0, '/data/programs/probe_position_correction_w_ptychonn/tools/utils')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from analysis_util import *

matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
plt.viridis()


dir_list = glob.glob('test*')
for d in dir_list:
    print(d)
    scan_idx = get_scan_index_from_dir_name(d)
    pos_true = get_true_position(scan_idx)
    pos_predicted_history = np.load(os.path.join(d, 'pos_history_calc_posCorr_1_clip_2.npy'))
    pos_baseline_history = np.load(os.path.join(d, 'pos_history_baseline_posCorr_1_clip_2.npy'))
    rms_ppe_n_history_predicted = []
    rms_ppe_n_history_baseline = []
    for i in range(pos_predicted_history.shape[0]):
        rms_ppe_n_history_predicted.append(calculate_rms_ppe_n(pos_predicted_history[i], pos_true, 3))
        rms_ppe_n_history_baseline.append(calculate_rms_ppe_n(pos_baseline_history[i], pos_true, 3))

    fig, ax = plt.subplots(1, 1)
    plt.semilogy(rms_ppe_n_history_baseline, label='Nominal', linestyle='--')
    plt.semilogy(rms_ppe_n_history_predicted, label='Predicted')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('RMS-PPE (pixel)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(frameon=False, fontsize=20)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(d, 'pos_error_comparison_baseline_calc.pdf'), transparent=True) 

