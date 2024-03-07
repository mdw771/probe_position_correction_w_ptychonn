import glob
import os.path
import re
import collections
import sys
sys.path.insert(0, '/data/programs/probe_position_correction_w_ptychonn/tools/utils')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.ndimage as ndi
import tifffile
from analysis_util import *


matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)


metric_func = calculate_rms_ppe_n  
metric_func_kwargs = {'n': 3}

data_dir = '../outputs'

# Get baseline
dset_list = glob.glob(os.path.join(data_dir, 'test*'))
baseline_refined_ppe_list = []
baseline_ppe_list = []
baseline_accum_refined_ppe_list = []
baseline_accum_ppe_list = []
baseline_indept_refined_ppe_list = []
baseline_indept_ppe_list = []
baseline_refined_rms_list = []
baseline_rms_list = []
baseline_accum_refined_rms_list = []
baseline_accum_rms_list = []
baseline_indept_refined_rms_list = []
baseline_indept_rms_list = []
for dset_folder in dset_list:
    try:
        scan_idx = int(re.findall('\d+', dset_folder)[-1])
        pos_baseline = get_baseline_position(scan_idx)
        pos_baseline_refined = get_actual_position(dset_folder, name_pattern='refined_baseline*')
        pos_true = get_true_position(scan_idx)
        baseline_ppe = metric_func(pos_baseline, pos_true, **metric_func_kwargs)
        baseline_refined_ppe = metric_func(pos_baseline_refined, pos_true, **metric_func_kwargs)
        baseline_ppe_list.append(baseline_ppe)
        baseline_refined_ppe_list.append(baseline_refined_ppe)
        if scan_idx >= 244:
            baseline_indept_refined_ppe_list.append(baseline_refined_ppe)
            baseline_indept_ppe_list.append(baseline_ppe)
        else:
            baseline_accum_refined_ppe_list.append(baseline_refined_ppe)
            baseline_accum_ppe_list.append(baseline_ppe)
    except IndexError:
        pass
baseline_refined_ppe = np.mean(baseline_refined_ppe_list)
baseline_ppe = np.mean(baseline_ppe_list)
baseline_accum_refined_ppe = np.mean(baseline_accum_refined_ppe_list)
baseline_accum_ppe = np.mean(baseline_accum_ppe_list)
baseline_indept_refined_ppe = np.mean(baseline_indept_refined_ppe_list)
baseline_indept_ppe = np.mean(baseline_indept_ppe_list)
print(baseline_refined_ppe)


folder_list = glob.glob(os.path.join(data_dir, 'dataDecimation_*'))
folder_list.sort()

decimation_ratios = []
total_ppe_list = []
total_accum_ppe_list = []
total_indept_ppe_list = []
for folder in folder_list:
    decimation_ratio = float(re.findall('\d+\.\d+', folder)[-1])
    dset_list = glob.glob(os.path.join(folder, 'test*'))
    ppe_list = []
    accum_ppe_list = []
    indept_ppe_list = []
    try:
        for dset_folder in dset_list:
            print(dset_folder)
            scan_idx = int(re.findall('\d+', dset_folder)[-1])
            pos_actual = get_actual_position(dset_folder)
            pos_true = get_true_position(scan_idx)
            ppe = metric_func(pos_actual, pos_true, **metric_func_kwargs)
            ppe_list.append(ppe)
            if scan_idx >= 244:
                indept_ppe_list.append(ppe)
            else:
                accum_ppe_list.append(ppe)
        ppe = np.mean(ppe_list)
        accum_ppe = np.mean(accum_ppe_list)
        indept_ppe = np.mean(indept_ppe_list)
        total_ppe_list.append(ppe)
        total_accum_ppe_list.append(accum_ppe)
        total_indept_ppe_list.append(indept_ppe)
        decimation_ratios.append(decimation_ratio)
    except:
        print('Cannot get positions for {}.'.format(scan_idx))

decimation_ratios = decimation_ratios[2:]
total_ppe_list = total_ppe_list[2:]
total_accum_ppe_list = total_accum_ppe_list[2:]
total_indept_ppe_list = total_indept_ppe_list[2:]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(decimation_ratios, total_ppe_list, label='All')
ax[1].plot(decimation_ratios, total_accum_ppe_list, label='Accumulating')
ax[2].plot(decimation_ratios, total_indept_ppe_list, label='Independent')
ax[0].hlines(baseline_refined_ppe, 0, 1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], linestyles='dashdot')
ax[0].hlines(baseline_ppe, 0, 1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], linestyles='--')
ax[1].hlines(baseline_accum_refined_ppe, 0, 1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], linestyles='dashdot')
ax[1].hlines(baseline_accum_ppe, 0, 1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], linestyles='--')
ax[2].hlines(baseline_indept_refined_ppe, 0, 1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], linestyles='dashdot')
ax[2].hlines(baseline_indept_ppe, 0, 1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], linestyles='--')
ax[0].set_title('All', fontsize=20)
ax[1].set_title('Accumulating errors', fontsize=20)
ax[2].set_title('Independent errors', fontsize=20)
for a in ax:
    a.grid(True)
baseline_hline_pos = [[baseline_refined_ppe, baseline_ppe], [baseline_accum_refined_ppe, baseline_accum_ppe], [baseline_indept_refined_ppe, baseline_indept_ppe]]
baseline_hline_labels = [['Nominal positions after refinement', 'Nominal positions before refinement'],
                         ['Nominal positions after refinement (accumulating)', 'Nominal positions before refinement (accumulating)'],
                         ['Nominal positions after refinement (independent)', 'Nominal positions before refinement (independent)']]
for i, a in enumerate(ax):
#    #a.legend()
#    text_margin = (a.get_ylim()[1] - a.get_ylim()[0]) * 0.03
#    a.text(0.25, baseline_hline_pos[i][0] - text_margin, s=baseline_hline_labels[i][0], color='gray', fontsize='small')
#    a.text(0.25, baseline_hline_pos[i][1] - text_margin, s=baseline_hline_labels[i][1], color='gray', fontsize='small')
    a.set_xlim(0, 0.92)
    a.set_xticks(np.arange(0.0, 0.91, 0.1))
    a.set_xlabel('Decimation ratio', fontsize=15)
    a.set_ylabel('Averaged RMS-PPE (pixel)', fontsize=15)
    a.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
#plt.show()
plt.savefig('ablation_test.pdf')
