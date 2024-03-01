import glob
import os
import sys
import re
sys.path.insert(0, '/data/programs/probe_position_correction_w_ptychonn/tools/utils')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analysis_util import *


matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
plt.viridis()


dir_list_0 = glob.glob('test*')
dir_list = []
for d in dir_list_0:
    if os.path.isdir(d):
        dir_list.append(d)


for dset_folder in dir_list:
    scan_idx = int(re.findall('\d+', dset_folder)[-1])
    pos_baseline = get_baseline_position(scan_idx)
    pos_predicted = get_actual_position(dset_folder)
    pos_refined = get_actual_position(dset_folder, name_pattern='refined_calc_pos.csv')
    pos_refined_baseline = get_actual_position(dset_folder, name_pattern='refined_baseline_pos.csv')
    pos_true = get_true_position(scan_idx)

    error_baseline = calculate_absolute_pos_error(pos_baseline, pos_true, return_stats=False)
    error_baseline_refined = calculate_absolute_pos_error(pos_refined_baseline, pos_true, return_stats=False)
    error_predicted = calculate_absolute_pos_error(pos_predicted, pos_true, return_stats=False)
    error_predicted_refined = calculate_absolute_pos_error(pos_refined, pos_true, return_stats=False)

    print(scan_idx)
    print('{: <20} {: <20}'.format('baseline_refined', 'predicted_refined'))
    print('{: <20} {: <20}'.format(error_baseline_refined, error_predicted_refined))

    # fig, ax = plt.subplots(1, 1)
    # plt.semilogy(losses_baseline, label='Nominal', linestyle='--')
    # plt.semilogy(losses_calculated, label='Predicted')
    # plt.xlabel('Epoch', fontsize=20)
    # plt.ylabel('RMS-PPE (pixel)', fontsize=20)
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # plt.legend(frameon=False, fontsize=20)
    # plt.tight_layout()
    # plt.savefig(os.path.join(dset_folder, 'pos_error_comparison_baseline_calc.pdf'), transparent=True)
