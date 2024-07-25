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


def calculate_rms_ppe(actual_pos, true_pos):
    actual_grad = actual_pos[1:] - actual_pos[:-1]
    true_grad = true_pos[1:] - true_pos[:-1]
    return np.sqrt(np.mean(np.sum((actual_grad - true_grad) ** 2, axis=1)))


def calculate_absolute_pos_error(actual_pos, true_pos, return_stats=True):
    actual_pos = actual_pos - np.mean(actual_pos, axis=0)
    true_pos = true_pos - np.mean(true_pos, axis=0)
    squared_errors = np.sum((actual_pos - true_pos) ** 2, axis=1)
    error_dists = np.sqrt(squared_errors)
    rms_error = np.sqrt(np.mean(squared_errors))
    if return_stats:
        s = {'mean_error_dist': np.mean(error_dists),
             'max_error_dist': np.max(error_dists)
            }
        return rms_error, s
    else:
        return rms_error


def subtract_mean(pos):
    pos = pos - np.mean(pos, axis=0)
    return pos
    

def plot_path(ax, pos, name=None, linestyle=None, marker='.', color=None, alpha=1):
    ax.plot(pos[:, 1], pos[:, 0], linewidth=1.5 if name == 'Predicted' else 1, label=name, linestyle=linestyle, c=color, alpha=alpha)
    ax.scatter(pos[:, 1], pos[:, 0], s=2, marker=marker, c=color, label=name, alpha=alpha)


data_dir = '.'

# Get baseline
dset_list = glob.glob(os.path.join(data_dir, 'test*'))

for dset_folder in dset_list:
    print(dset_folder)
    scan_idx = int(re.findall('\d+', dset_folder)[-1])
    pos_baseline = get_baseline_position(scan_idx)
    pos_predicted = get_actual_position(dset_folder)
    pos_refined = get_actual_position(dset_folder, name_pattern='refined_calc_pos.csv')
    pos_refined_baseline = get_actual_position(dset_folder, name_pattern='refined_baseline_pos.csv')
    pos_true = get_true_position(scan_idx)

    pos_baseline = subtract_mean(pos_baseline)
    pos_predicted = subtract_mean(pos_predicted)
    pos_true = subtract_mean(pos_true)
    pos_refined = subtract_mean(pos_refined)

    fig, ax = plt.subplots(1, 1)
    plot_path(ax, pos_baseline, 'Nominal', linestyle=(0, (5, 5)), color='gray', alpha=0.3)
    plot_path(ax, pos_predicted, 'Predicted', marker='.')
    plot_path(ax, pos_true, 'True', linestyle='dashed', marker='v')
    plot_path(ax, pos_refined, 'Refined', linestyle='dotted', marker='s', color='#3cf536')
    ax.tick_params(axis='both', which='major', labelsize=15)
    xtick_st = int(np.ceil(ax.get_xlim()[0] / 100) * 100)
    ytick_st = int(np.ceil(ax.get_ylim()[0] / 100) * 100)
    step = int(np.ceil(((ax.get_xlim()[1] - ax.get_xlim()[0]) / 4) / 100) * 100)
    ax.set_xticks(list(range(xtick_st, int(ax.get_xlim()[1]), step)))
    ax.set_yticks(list(range(ytick_st, int(ax.get_ylim()[1]), step)))
    ax.invert_yaxis()
    #plt.legend(frameon=False, fontsize=20)
    #plt.show()
    plt.savefig(os.path.join(dset_folder, 'comparison_path_plot_true_calc_refined_clip_2_collective_iter_2_nn_12_sw_1e-3_1e-2.pdf'), transparent=True)

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    plot_path(ax, pos_baseline, 'Nominal', linestyle=(0, (5, 5)), color='gray', alpha=0.3)
    plot_path(ax, pos_predicted, 'Predicted', marker='.')
    plot_path(ax, pos_true, 'True', linestyle='dashed', marker='v')
    plot_path(ax, pos_refined, 'Refined', linestyle='dotted', marker='s', color='#3cf536')
    h = ax.get_ylim()[1] - ax.get_ylim()[0]
    w = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.set_ylim(-h / 10, h / 10)
    ax.set_xlim(-w / 10, w / 10)
    ax.set_aspect('equal')
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.invert_yaxis()
    #plt.legend(frameon=False, fontsize=20)
    #plt.show()
    plt.savefig(os.path.join(dset_folder, 'comparison_path_plot_thumbnail.pdf'), transparent=True)
