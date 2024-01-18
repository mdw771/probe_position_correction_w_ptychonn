import glob
import os.path
import re
import collections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.ndimage as ndi
import tifffile

matplotlib.rc('font',family='Times New Roman')
matplotlib.rcParams['font.size'] = 14
plt.viridis()


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
ppe_list = []
for dset_folder in dset_list:
    try:
        scan_idx = int(re.findall('\d+', dset_folder)[-1])
        pos_actual = get_actual_position(dset_folder, name_pattern='refined_baseline*')
        pos_true = get_true_position(scan_idx)
        ppe = calculate_position_grad_error(pos_actual, pos_true)
        ppe_list.append(ppe)
    except IndexError:
        pass
baseline_ppe = np.mean(ppe_list)
print(baseline_ppe)


folder_list = glob.glob(os.path.join(data_dir, 'dataDecimation_*'))
folder_list.sort()

decimation_ratios = []
total_ppe_list = []
for folder in folder_list:
    decimation_ratio = float(re.findall('\d+\.\d+', folder)[-1])
    dset_list = glob.glob(os.path.join(folder, 'test*'))
    ppe_list = []
    for dset_folder in dset_list:
        print(dset_folder)
        scan_idx = int(re.findall('\d+', dset_folder)[-1])
        pos_actual = get_actual_position(dset_folder)
        pos_true = get_true_position(scan_idx)
        ppe = calculate_position_grad_error(pos_actual, pos_true)
        ppe_list.append(ppe)
    ppe = np.mean(ppe_list)
    total_ppe_list.append(ppe)
    decimation_ratios.append(decimation_ratio)

plt.plot(decimation_ratios, total_ppe_list)
plt.hlines(baseline_ppe, 0, 1, color='gray', linestyles='--', label='Nominal positions after refinement')
plt.text(0.35, baseline_ppe - 0.2, s='Nominal positions after optimization', color='gray', fontsize='small')
plt.xlim(0.08, 0.92)
plt.xticks(np.arange(0.1, 0.91, 0.1))
plt.xlabel('Decimation ratio')
plt.ylabel('Averaged RMS of pairwise position error (pixel)')
plt.savefig('ablation_test.pdf')
