import glob
import os.path
import re
import collections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.ndimage as ndi
import tifffile


matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)


def get_true_position(scan_idx, psize_nm=8):
    probe_pos_list_true = np.genfromtxt('../data/pos{}.csv'.format(scan_idx), delimiter=',').astype('float32')
    probe_pos_list_true /= (psize_nm * 1e-9)
    return probe_pos_list_true


def get_baseline_position(scan_idx, psize_nm=8):
    scaling_dict = collections.defaultdict(lambda: 1.0,
                                           {236: 0.5, 239: 0.5, 240: 0.25, 241: 0.25, 242: 0.25, 250: 0.5, 251: 0.5,
                                            252: 0.25, 253: 0.25})
    s = scaling_dict[scan_idx]
    probe_pos_list_baseline = np.genfromtxt('../data/pos221.csv', delimiter=',').astype('float32') / (psize_nm * 1e-9) * s  # Baseline
    return probe_pos_list_baseline


def get_actual_position(folder, name_pattern=None, psize_nm=8):
    if name_pattern is None:
        name_pattern = 'calc_pos_*_collective_niters_2_beta_0p5_nn_12_sw_1e-2_1e-3.csv'
    fname = glob.glob(os.path.join(folder, name_pattern))[0]
    probe_pos_list = np.genfromtxt(fname, delimiter=',').astype('float32')
    probe_pos_list /= (psize_nm * 1e-9)
    return probe_pos_list


def calculate_position_grad_error(actual_pos, true_pos):
    actual_grad = actual_pos[1:] - actual_pos[:-1]
    true_grad = true_pos[1:] - true_pos[:-1]
    return np.sqrt(np.mean(np.sum((actual_grad - true_grad) ** 2, axis=1)))


data_dir = '../outputs'

# Get baseline
dset_list = glob.glob(os.path.join(data_dir, 'test*'))
baseline_refined_ppe_list = []
baseline_ppe_list = []
baseline_accum_refined_ppe_list = []
baseline_accum_ppe_list = []
baseline_indept_refined_ppe_list = []
baseline_indept_ppe_list = []
for dset_folder in dset_list:
    try:
        scan_idx = int(re.findall('\d+', dset_folder)[-1])
        pos_baseline = get_baseline_position(scan_idx)
        pos_baseline_refined = get_actual_position(dset_folder, name_pattern='refined_baseline*')
        pos_true = get_true_position(scan_idx)
        baseline_ppe = calculate_position_grad_error(pos_baseline, pos_true)
        baseline_refined_ppe = calculate_position_grad_error(pos_baseline_refined, pos_true)
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
            ppe = calculate_position_grad_error(pos_actual, pos_true)
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

decimation_ratios = decimation_ratios[3:]
total_ppe_list = total_ppe_list[3:]
total_accum_ppe_list = total_accum_ppe_list[3:]
total_indept_ppe_list = total_indept_ppe_list[3:]

plt.plot(decimation_ratios, total_ppe_list, label='All')
plt.plot(decimation_ratios, total_accum_ppe_list, label='Accumulating')
plt.plot(decimation_ratios, total_indept_ppe_list, label='Independent')
plt.hlines(baseline_refined_ppe, 0, 1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], linestyles='dashdot')
plt.hlines(baseline_ppe, 0, 1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], linestyles='--')
plt.hlines(baseline_accum_refined_ppe, 0, 1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], linestyles='dashdot')
plt.hlines(baseline_accum_ppe, 0, 1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], linestyles='--')
plt.hlines(baseline_indept_refined_ppe, 0, 1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], linestyles='dashdot')
plt.hlines(baseline_indept_ppe, 0, 1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], linestyles='--')
plt.text(0.35, baseline_refined_ppe - 0.3, s='Nominal positions after refinement', color='gray', fontsize='small')
plt.text(0.35, baseline_ppe - 0.3, s='Nominal positions before refinement', color='gray', fontsize='small')
plt.text(0.35, baseline_accum_refined_ppe - 0.3, s='Nominal positions after refinement (accumulating)', color='gray', fontsize='small')
plt.text(0.35, baseline_accum_ppe - 0.3, s='Nominal positions before refinement (accumulating)', color='gray', fontsize='small')
plt.text(0.35, baseline_indept_refined_ppe - 0.3, s='Nominal positions after refinement (independent)', color='gray', fontsize='small')
plt.text(0.35, baseline_indept_ppe - 0.3, s='Nominal positions before refinement (independent)', color='gray', fontsize='small')
plt.legend()
plt.xlim(0.08, 0.92)
plt.xticks(np.arange(0.1, 0.91, 0.1))
plt.xlabel('Decimation ratio')
plt.ylabel('Averaged RMS of pairwise position error (pixel)')
plt.savefig('ablation_test.pdf')
